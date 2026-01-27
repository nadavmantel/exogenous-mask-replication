"""
================================================================================
GB ELECTRICITY MODEL: Complete Single-File Implementation
================================================================================

This is a complete, self-contained model that:
1. Loads real GB data from PyPSA-GB (or generates calibrated synthetic data)
2. Builds PyPSA networks for capacity expansion
3. Compares exogenous vs endogenous technology learning
4. Runs policy scenarios
5. Generates analysis and visualizations

DATA SOURCES (when PyPSA-GB available):
- ESPENI: Half-hourly GB demand (Wilson et al., 2021)
- Atlite/ERA5: Weather-based renewable profiles (Hofmann et al., 2021)
- National Grid FES: Future scenarios (National Grid ESO)

USAGE:
    Press play on your favorite IDE. At the bottom of this repo, in main(), under 'if _main_', there are a few different
    modes: normal, battery_sweep, solar_battery_sweep, and sensitivity.
    Normal mode runs on each of the stated scenarios under POLOCIES and outputs to gb_model_results.csv. Currently the
    renewables profile is that of 2019. Not the capacity, but the capacity factors.

    Battery_sweep uses only a battery subsidy that starts at the chosen amount, is set for the 2025-2027 investment
    cycle, and ends in the 2029 investment cycle. The sweep goes over a set of subsidies starting and ending by user
    inputs. The demand and renewable profiles are set by the normal model code. The sweep runs through similar functions
    as normal, but ignores the policies.

    Solar_Battery_sweep does a similar sweep to battery sweep but in a 2d grid.

    Sensitivity actually does a binary search to find the threshold amount for batteries using a set of different
    learning rates. This is, as should be obvious, a sensitivity analysis of the scenarios using +-2% on learning rates
    for batteries.


    All the assumptions that went into this script can be found in gb_model_assumptions.md


Author: Nadav Mantel
Date: December 2025

================================================================================
"""

import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import argparse
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')


# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to PyPSA-GB repository (set to None to use synthetic data)
# Clone from: https://github.com/andrewlyden/PyPSA-GB
PYPSA_GB_PATH = "./"  # e.g., "./PyPSA-GB" or "/home/user/PyPSA-GB"

# Model time settings
PERIODS = list(range(2025, 2051, 2))  # Every 2 years: 2025, 2027, 2029, ..., 2049
HOURS_PER_YEAR = 8760
DISCOUNT_RATE = 0.07

# Weather year for renewable/demand profiles
WEATHER_YEAR = 2019  # Use 2010-2020 for ESPENI data availability

# Historical installed capacities (MW) for capacity factor calculation
# These are used when loading real data to convert MW output to capacity factors
INSTALLED_CAPACITY_2019 = {
    'offshore_wind': 9700,   # UK offshore wind capacity end of 2019
    'onshore_wind': 13500,   # UK onshore wind capacity end of 2019
    'solar': 13000,          # UK solar PV capacity end of 2019
}

# GB 2024 existing capacity (GW)
EXISTING_CAPACITY = {
    "gas_ccgt": 36.0,
    "gas_peaker": 5.0,
    "nuclear": 5.0,
    "solar": 16.0,
    "onshore_wind": 15.0,
    "offshore_wind": 15.0,
    "battery": 3.0,
}

# ============================================================================
# UK SHARE OF GLOBAL MARKET (for endogenous learning)
# ============================================================================
# 
# KEY ASSUMPTION: UK deployment is proportional to global deployment.
# When UK builds X GW, we assume the world builds X / UK_SHARE GW.
#
# This "constant market share" assumption implies:
# - UK decisions reflect global technology trends
# - Cost reductions depend on global deployment, not UK alone
# - UK neither free-rides nor over-contributes to learning
#
# Alternative interpretations:
# - Higher share = UK is a "technology leader" (deployment drives global costs)
# - Lower share = UK is a "technology follower" (takes global costs as given)
#
# Sources: IRENA (2023), BNEF, IEA for market share estimates

UK_GLOBAL_SHARE = {
    "solar": 0.009,          # IEA-PVPS "Snapshot of Global PV Markets 2025", global cumulative capacity reached approximately 2.2 TW (2,200 GW) by the end of 2024. The UK's capacity as of early 2025 stands at roughly 18 GW (per DESNZ Energy Trends).
    "onshore_wind": 0.015,   # The Global Wind Energy Council (GWEC) 2025 Report indicates that global onshore wind capacity has surpassed 1 TW (1,000 GW). The UK has roughly 16 GW of operational onshore wind
    "offshore_wind": 0.18,  # Global offshore capacity is roughly 90–95 GW. With 16 GW currently operational and a massive pipeline (the "ScotWind" and "Round 4" projects), the UK accounts for nearly one-fifth of global installations.Source: RenewableUK Industrial Growth Plan (2024) and the ETC (Energy Transitions Commission)
    "battery": 0.045,        # UAccording to the Faraday Institution (2024/25 report), the UK’s capacity of ~7.4 GW / 12.9 GWh represents a significant portion of the global stationary storage market (approx. 164 GW globally).
    "gas_ccgt": 0.02,       # Mature tech, share doesn't matter (LR=0)
    "gas_peaker": 0.02,
    "nuclear": 0.01,        # UK ~1% of global nuclear
}

#Unlimited Capacity
MAX_BUILD_CAPACITY = {
    "solar": 1000,  # ~2.5 GW/year - UK did 4 GW in 2015
    "onshore_wind": 1000,  # ~1 GW/year - constrained by planning
    "offshore_wind": 1000,  # ~2.5 GW/year - UK's max was ~2.5 GW
    "battery": 1000,  # # ~1.5 GW/year - growing fast
    "gas_ccgt": 1000,  # ~1 GW/year - limited by planning/politics
    "gas_peaker": 1000, # Smaller plants, faster to build
    "nuclear": 1000, # HPC is 3.2 GW over 10+ years!
}
# ============================================================================
# TECHNOLOGY DEFINITIONS
# ============================================================================

@dataclass
class Technology:
    """Technology parameters for GB."""
    name: str
    capital_cost: float      # £/kW (2025)
    marginal_cost: float     # £/MWh
    learning_rate: float     # Per doubling of cumulative capacity
    global_capacity_2024: float  # GW
    max_build_per_period: float  # GW per 2-year period (realistic constraint)
    capacity_factor: float   # For dispatchable (max for variable)
    co2_intensity: float     # tCO2/MWh
    lifetime: int
    is_variable: bool


# Realistic build rates based on UK industry capacity and supply chains
# Sources: CCC, National Grid FES, ORE Catapult, industry reports

TECHNOLOGIES = {
    "gas_ccgt": Technology(
        name="gas_ccgt", capital_cost=750, marginal_cost=50,
        learning_rate=0.0, global_capacity_2024=2000,
        max_build_per_period=MAX_BUILD_CAPACITY["gas_ccgt"],  # ~1 GW/year - limited by planning/politics
        capacity_factor=0.85, co2_intensity=0.34, lifetime=25, is_variable=False
    ),
    "gas_peaker": Technology(
        name="gas_peaker", capital_cost=500, marginal_cost=80,
        learning_rate=0.0, global_capacity_2024=500,
        max_build_per_period=MAX_BUILD_CAPACITY["gas_peaker"],  # Smaller plants, faster to build
        capacity_factor=0.10, co2_intensity=0.45, lifetime=25, is_variable=False
    ),
    "nuclear": Technology(
        name="nuclear", capital_cost=8000, marginal_cost=10,
        learning_rate=0.0, global_capacity_2024=400,
        max_build_per_period=MAX_BUILD_CAPACITY["nuclear"],  # HPC is 3.2 GW over 10+ years!
        capacity_factor=0.85, co2_intensity=0.0, lifetime=60, is_variable=False
    ),
    "solar": Technology(
        name="solar", capital_cost=450, marginal_cost=0,
        learning_rate=0.20, global_capacity_2024=1800,
        max_build_per_period=MAX_BUILD_CAPACITY["solar"],  # ~2.5 GW/year - UK did 4 GW in 2015
        capacity_factor=0.11, co2_intensity=0.0, lifetime=25, is_variable=True
    ),
    "onshore_wind": Technology(
        name="onshore_wind", capital_cost=950, marginal_cost=0,
        learning_rate=0.13, global_capacity_2024=1000,
        max_build_per_period=MAX_BUILD_CAPACITY["onshore_wind"],  # ~1 GW/year - constrained by planning
        capacity_factor=0.26, co2_intensity=0.0, lifetime=25, is_variable=True
    ),
    "offshore_wind": Technology(
        name="offshore_wind", capital_cost=1800, marginal_cost=0,
        learning_rate=0.13, global_capacity_2024=75,
        max_build_per_period=MAX_BUILD_CAPACITY["offshore_wind"],  # ~2.5 GW/year - UK's max was ~2.5 GW in 2022
        capacity_factor=0.40, co2_intensity=0.0, lifetime=25, is_variable=True
    ),
    "battery": Technology(
        name="battery", capital_cost=600, marginal_cost=3,
        learning_rate=0.25, global_capacity_2024=300,
        max_build_per_period=MAX_BUILD_CAPACITY["battery"],  # ~1.5 GW/year - growing fast
        capacity_factor=1.0, co2_intensity=0.0, lifetime=15, is_variable=False
    ),
}


# ============================================================================
# POLICY SCENARIOS
# ============================================================================

@dataclass  
class Policy:
    """Policy scenario definition."""
    name: str
    carbon_price: Dict[int, float]  # £/tCO2 by year
    renewable_subsidy: Dict[str, Dict[int, float]]  # £/kW by tech and year


POLICIES = {
    "baseline": Policy(
        name="Current Policy",
        carbon_price={
            2025: 50, 2027: 62, 2029: 74,
            2031: 86, 2033: 98, 2035: 110,
            2037: 122, 2039: 134, 2041: 146,
            2043: 158, 2045: 170, 2047: 182, 2049: 194
        },
        renewable_subsidy={
            "offshore_wind": {
                2025: 100, 2027: 92, 2029: 84,
                2031: 76, 2033: 68, 2035: 60,
                2037: 52, 2039: 44, 2041: 36,
                2043: 28, 2045: 20, 2047: 12, 2049: 4
            },
            "onshore_wind": {
                2025: 50, 2027: 46, 2029: 42,
                2031: 38, 2033: 34, 2035: 30,
                2037: 26, 2039: 22, 2041: 18,
                2043: 14, 2045: 10, 2047: 6, 2049: 2
            },
            "solar": {
                2025: 50, 2027: 46, 2029: 42,
                2031: 38, 2033: 34, 2035: 30,
                2037: 26, 2039: 22, 2041: 18,
                2043: 14, 2045: 10, 2047: 6, 2049: 2
            },
        }
    ),
    "carbon_price_only": Policy(
        name="Carbon Tax Only",
        carbon_price={
            2025: 650, 2027: 623, 2029: 596, 2031: 569, 2033: 542, 2035: 515, 2037: 488,
            2039: 460, 2041: 433, 2043: 406, 2045: 379, 2047: 352, 2049: 325},
        renewable_subsidy={},
    ),
    "battery_solar": Policy(
        name="Battery + Solar",
        carbon_price={},
        renewable_subsidy={
            "battery": {
                2025: 480, 2027: 480, 2029: 240, 2031: 0, 2033: 0, 2035: 0,
                2037: 0, 2039: 0, 2041: 0, 2043: 0, 2045: 0, 2047: 0, 2049: 0
            },

            "solar": {
                2025: 105, 2027: 105, 2029: 52, 2031: 0,
                2033: 0, 2035: 0, 2037: 0, 2039: 0, 2041: 0, 2043: 0, 2045: 0, 2047: 0, 2049: 0
            },
        }
    ),
    "battery_heavy": Policy(
        name="Battery Heavy",
        carbon_price={},
        renewable_subsidy={
            "battery": {2025: 490, 2027: 490, 2029: 490, 2031: 0,
                2033: 0, 2035: 0, 2037: 0, 2039: 0, 2041: 0, 2043: 0, 2045: 0, 2047: 0, 2049: 0},
        }
    ),
}


# ============================================================================
# DATA LOADING
# ============================================================================

class GBDataLoader:
    """Load GB electricity data from PyPSA-GB or generate synthetic."""
    
    def __init__(self, pypsa_gb_path: str = None):
        self.pypsa_gb_path = Path(pypsa_gb_path) if pypsa_gb_path else None
        self.data = None
        
    def load(self, year: int = 2019) -> Dict[str, np.ndarray]:
        """Load all required data."""
        
        print(f"\nLoading GB data for weather year {year}...")
        
        # Try uploaded files first
        uploads_path = Path("/mnt/user-data/uploads")
        if uploads_path.exists() and any(uploads_path.glob("*.csv")):
            data = self.load_from_uploaded_files(str(uploads_path))
            if data and len(data) >= 4:
                self.data = data
                return data
        
        # Try PyPSA-GB repository
        if self.pypsa_gb_path and self.pypsa_gb_path.exists():
            data = self._load_from_pypsa_gb(year)
            if data:
                self.data = data
                return data
        
        # Fall back to synthetic
        print("  Using synthetic data (calibrated to GB statistics)")
        self.data = self._generate_synthetic(year)
        return self.data

    def load_from_uploaded_files(self, uploads_path: str = "/mnt/user-data/uploads") -> Dict[str, np.ndarray]:
        """
        Load data from uploaded PyPSA-GB Atlite output files.
        
        Expected files:
        - espeni_example.csv (or espeni.csv): Half-hourly demand
        - Wind_Offshore_2019.csv: Hourly offshore wind MW by farm
        - Wind_Onshore_2019_example.csv: Hourly onshore wind MW by farm  
        - PV_2019_*.csv: Hourly solar MW by site (multiple files)
        
        These files contain MW output per generator - we sum and normalize to capacity factors.
        """
        from pathlib import Path
        uploads = Path(uploads_path)
        data = {}
        
        print(f"  Loading from uploaded PyPSA-GB files...")
        
        # ---- ESPENI DEMAND ----
        espeni_files = list(uploads.glob("espeni*.csv"))
        if espeni_files:
            print(f"    Demand: {espeni_files[0].name}")
            df = pd.read_csv(espeni_files[0], low_memory=False)
            
            if 'POWER_ESPENI_MW' in df.columns:
                demand = df['POWER_ESPENI_MW'].values
                # Convert half-hourly to hourly by averaging pairs
                if len(demand) > 8760 * 2:
                    # Filter to 2019 if multiple years
                    if 'ELEXM_utc' in df.columns:
                        df['datetime'] = pd.to_datetime(df['ELEXM_utc'])
                        df_2019 = df[df['datetime'].dt.year == 2019]
                        demand = df_2019['POWER_ESPENI_MW'].values
                
                # Average half-hourly to hourly
                    # if len(demand) > 10000:
                    demand = demand[:17520].reshape(-1, 2).mean(axis=1)
                
                # Convert MW to GW
                data['demand'] = demand[:8760] / 1000
                print(f"      Mean: {data['demand'].mean():.1f} GW, Peak: {data['demand'].max():.1f} GW")
        
        # ---- OFFSHORE WIND ----
        offshore_files = list(uploads.glob("*[Oo]ffshore*2019*.csv"))
        if offshore_files:
            print(f"    Offshore: {offshore_files[0].name}")
            df = pd.read_csv(offshore_files[0], index_col=0)
            total_mw = df.sum(axis=1).values
            
            # Extend if less than 8760 hours
            if len(total_mw) < 8760:
                total_mw = np.tile(total_mw, 8760 // len(total_mw) + 1)[:8760]
            
            # Convert to capacity factor using actual 2019 UK installed capacity
            installed_capacity_mw = INSTALLED_CAPACITY_2019.get('offshore_wind', 9700)
            cf = total_mw[:8760] / installed_capacity_mw
            # Clip to [0, 1] in case of data issues
            data['offshore_wind'] = np.clip(cf, 0, 1)
            print(f"      CF: {data['offshore_wind'].mean()*100:.1f}% (from real 2019 weather)")
        
        # ---- ONSHORE WIND ----
        onshore_files = list(uploads.glob("*[Oo]nshore*2019*.csv"))
        if onshore_files:
            print(f"    Onshore: {onshore_files[0].name}")
            df = pd.read_csv(onshore_files[0], index_col=0, low_memory=False)
            
            # Sum across all farms, handling NaN
            total_mw = df.sum(axis=1, skipna=True).values
            
            # Check how much valid data we have
            valid_mask = ~np.isnan(total_mw) & (total_mw > 0)
            valid_count = valid_mask.sum()
            
            if valid_count < 1000:  # Not enough valid data
                print(f"      Warning: Only {valid_count} valid hours, using synthetic")
                synthetic = self._generate_synthetic(2019)
                data['onshore_wind'] = synthetic['onshore_wind']
            else:
                # Use valid data and extend/repeat if needed
                valid_mw = total_mw[valid_mask]
                if len(valid_mw) < 8760:
                    total_mw = np.tile(valid_mw, 8760 // len(valid_mw) + 1)[:8760]
                else:
                    total_mw = valid_mw[:8760]
                
                # Normalize to capacity factor, target ~26.5% mean CF
                capacity = total_mw.max()
                if capacity > 0:
                    cf = total_mw / capacity
                    cf = cf * (0.265 / max(cf.mean(), 0.01))
                    data['onshore_wind'] = np.clip(cf, 0, 1)
            print(f"      CF: {data['onshore_wind'].mean()*100:.1f}%")
        
        # ---- SOLAR PV ----
        # PV files are split by quarter: PV_2019_1 (Q1), PV_2019_2 (Q2), etc.
        # Some quarters may only have example versions
        # Strategy: use full version if available, otherwise example
        pv_files_to_use = []
        for q in ['1', '2', '3', '4']:
            full = list(uploads.glob(f"PV_2019_{q}.csv"))
            example = list(uploads.glob(f"PV_2019_{q}_example.csv"))
            if full:
                pv_files_to_use.extend(full)
            elif example:
                pv_files_to_use.extend(example)
        
        if pv_files_to_use:
            print(f"    Solar: {len(pv_files_to_use)} PV files (Q1-Q4)")
            all_pv = []
            for f in sorted(pv_files_to_use):
                try:
                    df = pd.read_csv(f, index_col=0, low_memory=False)
                    row_sums = df.sum(axis=1, skipna=True).values
                    # Only include valid (non-NaN, non-zero) values
                    valid = row_sums[~np.isnan(row_sums)]
                    if len(valid) > 0:
                        all_pv.append(valid)
                except Exception as e:
                    print(f"      Warning: Could not read {f.name}: {e}")
            
            if all_pv:
                # Concatenate all PV data
                total_mw = np.concatenate(all_pv)
                
                # Extend if less than 8760 hours
                if len(total_mw) < 8760:
                    total_mw = np.tile(total_mw, 8760 // len(total_mw) + 1)[:8760]
                else:
                    total_mw = total_mw[:8760]
                
                # Convert to capacity factor using actual 2019 UK installed capacity
                installed_capacity_mw = INSTALLED_CAPACITY_2019.get('solar', 13000)
                cf = total_mw / installed_capacity_mw
                # Clip to [0, 1] in case of data issues
                data['solar'] = np.clip(cf, 0, 1)
                print(f"      CF: {data['solar'].mean()*100:.1f}% (from real 2019 weather)")
        
        # Check completeness
        required = ['demand', 'solar', 'onshore_wind', 'offshore_wind']
        missing = [k for k in required if k not in data]
        
        if missing:
            print(f"    Warning: Missing {missing}, using synthetic")
            synthetic = self._generate_synthetic(2019)
            for k in missing:
                data[k] = synthetic[k]
        
        data['source'] = 'PyPSA-GB Atlite (uploaded files)'
        return data
    
    def _load_from_pypsa_gb(self, year: int) -> Optional[Dict[str, np.ndarray]]:
        """Load from PyPSA-GB repository."""
        
        data = {}
        
        # ESPENI demand
        espeni_path = self.pypsa_gb_path / "data" / "demand" / "espeni.csv"
        if espeni_path.exists():
            print(f"  Loading ESPENI demand from {espeni_path}")
            try:
                df = pd.read_csv(espeni_path, index_col=0, low_memory=False)
                
                # Try to convert index to datetime
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    print(f"    Warning: Could not parse dates in index")
                
                # Find demand column
                demand_cols = [c for c in df.columns if 'demand' in c.lower() or 'espeni' in c.lower() or 'mw' in c.lower()]
                if demand_cols:
                    demand_col = demand_cols[0]
                else:
                    # Use first numeric column
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    demand_col = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]
                
                print(f"    Using column: {demand_col}")
                
                # Filter by year if index is datetime
                if hasattr(df.index, 'year'):
                    year_data = df[df.index.year == year][demand_col].dropna()
                else:
                    # Just use all data
                    year_data = df[demand_col].dropna()
                
                if len(year_data) == 0:
                    print(f"    Warning: No data found for year {year}, using all available data")
                    year_data = df[demand_col].dropna()
                
                # Convert to numpy array
                demand_values = year_data.values.astype(float)
                
                # Resample if half-hourly (more than 8760 values)
                if len(demand_values) > 8760 * 1.5:
                    # Average pairs for half-hourly data
                    demand_values = demand_values[:len(demand_values)//2*2].reshape(-1, 2).mean(axis=1)
                
                # Ensure we have 8760 values
                if len(demand_values) < 8760:
                    # Tile to extend
                    demand_values = np.tile(demand_values, 8760 // len(demand_values) + 1)[:8760]
                else:
                    demand_values = demand_values[:8760]
                
                # Convert to GW if in MW (values > 1000 suggest MW)
                if np.nanmean(demand_values) > 1000:
                    demand_values = demand_values / 1000
                    
                data['demand'] = demand_values
                print(f"    Mean: {np.nanmean(data['demand']):.1f} GW, Peak: {np.nanmax(data['demand']):.1f} GW")
            except Exception as e:
                print(f"    Error loading ESPENI: {e}")
        
        # Renewable profiles from Atlite
        # PyPSA-GB stores profiles in subdirectories like:
        # data/renewables/atlite/outputs/PV/pv_profile_2019.csv
        # data/renewables/atlite/outputs/Wind_Onshore/onshore_profile_2019.csv
        atlite_path = self.pypsa_gb_path / "data" / "renewables" / "atlite" / "outputs"
        
        tech_configs = [
            ('solar', ['PV', 'Solar'], ['pv', 'solar']),
            ('onshore_wind', ['Wind_Onshore', 'Onshore'], ['onshore', 'wind_onshore']),
            ('offshore_wind', ['Wind_Offshore', 'Offshore'], ['offshore', 'wind_offshore'])
        ]
        
        for tech, dir_patterns, file_patterns in tech_configs:
            found = False
            
            # First, try to find subdirectories
            for dir_pattern in dir_patterns:
                tech_dir = atlite_path / dir_pattern
                if tech_dir.exists() and tech_dir.is_dir():
                    # Look for CSV files inside
                    csv_files = list(tech_dir.glob("*.csv"))
                    # Prefer files with the year
                    year_files = [f for f in csv_files if str(year) in f.name]
                    if year_files:
                        csv_files = year_files
                    
                    if csv_files:
                        f = csv_files[0]
                        print(f"  Loading {tech} from {f}")
                        try:
                            profile = self._load_profile_csv(f, year)
                            if profile is not None and len(profile) > 0:
                                data[tech] = profile
                                print(f"    CF: {np.nanmean(data[tech])*100:.1f}%")
                                found = True
                                break
                        except Exception as e:
                            print(f"    Error: {e}")
                            continue
            
            if found:
                continue
                
            # Fallback: look for files directly in atlite_path
            if atlite_path.exists():
                for f in atlite_path.glob("*.csv"):
                    fname = f.name.lower()
                    if any(p in fname for p in file_patterns):
                        print(f"  Loading {tech} from {f}")
                        try:
                            profile = self._load_profile_csv(f, year)
                            if profile is not None and len(profile) > 0:
                                data[tech] = profile
                                print(f"    CF: {np.nanmean(data[tech])*100:.1f}%")
                                found = True
                                break
                        except Exception as e:
                            print(f"    Error: {e}")
                            continue
        
        # Check if we got all data
        required = ['demand', 'solar', 'onshore_wind', 'offshore_wind']
        missing = [k for k in required if k not in data]
        
        if missing:
            print(f"  Missing data: {missing}")
            return None
            
        data['source'] = 'PyPSA-GB (ESPENI + Atlite/ERA5)'
        return data

    def _load_profile_csv(self, filepath, year: int) -> Optional[np.ndarray]:
        """Load a capacity factor profile from CSV file.

        Handles two data formats:
        1. Single column with capacity factors (values 0-1)
        2. Multiple columns with MW output per generator (wind farms, solar sites)
           → Sum all columns and normalize by max to get capacity factor
        """
        df = pd.read_csv(filepath, index_col=0, low_memory=False)

        # Try to parse index as datetime
        try:
            df.index = pd.to_datetime(df.index)
        except:
            pass

        # Check for standard capacity factor column names first
        if 'capacity_factor' in df.columns:
            profile = df['capacity_factor'].values.astype(float)
        elif 'p_max_pu' in df.columns:
            profile = df['p_max_pu'].values.astype(float)
        elif any(c.lower() == 'cf' for c in df.columns):
            col = [c for c in df.columns if c.lower() == 'cf'][0]
            profile = df[col].values.astype(float)
        else:
            # No standard CF column - this is likely MW data per generator
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) == 0:
                print(f"    Warning: No numeric columns in {filepath}")
                return None

            if len(numeric_cols) > 3:
                # Multiple columns = MW per generator (wind farms, solar sites, etc.)
                # SUM all columns to get total output, then normalize
                print(f"    Summing {len(numeric_cols)} generators")
                total_output = df[numeric_cols].sum(axis=1).fillna(0).values.astype(float)

                # Normalize by max output (≈ installed capacity) to get CF
                max_output = np.nanmax(total_output)
                if max_output > 0:
                    profile = total_output / max_output
                else:
                    print(f"    Warning: Max output is 0")
                    return None
            else:
                # Single or few columns - use first one
                profile = df[numeric_cols[0]].values.astype(float)

                # Normalize if needed
                max_val = np.nanmax(profile)
                if max_val > 1.0:
                    profile = profile / max_val

        # Filter by year if datetime index
        if hasattr(df.index, 'year'):
            year_mask = pd.Series(df.index.year == year)
            if year_mask.sum() > 0 and year_mask.sum() < len(profile):
                profile = profile[year_mask.values]

        # Handle half-hourly data (>8760 values)
        if len(profile) > 8760 * 1.5:
            profile = profile[:len(profile) // 2 * 2].reshape(-1, 2).mean(axis=1)

        # Ensure exactly 8760 values
        if len(profile) < 8760:
            if len(profile) > 100:
                profile = np.tile(profile, 8760 // len(profile) + 1)[:8760]
            else:
                print(f"    Warning: Only {len(profile)} values")
                return None
        else:
            profile = profile[:8760]

        # Clip to valid range [0, 1]
        profile = np.clip(profile, 0, 1)

        return profile

    def _generate_synthetic(self, year: int, seed: int = 42) -> Dict[str, np.ndarray]:
        """Generate synthetic GB profiles calibrated to published statistics."""
        
        np.random.seed(seed + year)
        hours = 8760
        t = np.arange(hours)
        hour_of_day = t % 24
        day_of_year = (t // 24) % 365
        day_of_week = (t // 24) % 7
        
        # DEMAND - Calibrated to ESPENI/National Grid
        base = 32
        seasonal = 8 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
        morning = 4 * np.exp(-((hour_of_day - 8.5) ** 2) / 4)
        evening = 6 * np.exp(-((hour_of_day - 17.5) ** 2) / 4)
        night = -5 * np.exp(-((hour_of_day - 3.5) ** 2) / 6)
        weekend = 1 - 0.10 * (day_of_week >= 5).astype(float)
        
        demand = (base + seasonal + morning + evening + night) * weekend
        noise = np.random.randn(hours)
        for i in range(1, hours):
            noise[i] = 0.9 * noise[i-1] + 0.1 * noise[i]
        demand = np.clip(demand * (1 + 0.03 * noise), 20, 60)
        
        # SOLAR - 10.5% CF (DUKES)
        declination = 23.45 * np.sin(2 * np.pi * (day_of_year - 81) / 365)
        cos_ha = np.clip(-np.tan(np.radians(52)) * np.tan(np.radians(declination)), -1, 1)
        sunrise = 12 - np.degrees(np.arccos(cos_ha)) / 15
        sunset = 12 + np.degrees(np.arccos(cos_ha)) / 15
        
        solar = np.zeros(hours)
        for i in range(hours):
            h, d = hour_of_day[i], day_of_year[i]
            if sunrise[d] < h < sunset[d]:
                t_norm = (h - sunrise[d]) / (sunset[d] - sunrise[d])
                solar[i] = np.sin(np.pi * t_norm) ** 1.2 * (0.4 + 0.6 * np.sin(np.pi * (d + 10) / 365))
        
        cloud = np.random.rand(hours)
        for i in range(1, hours):
            cloud[i] = 0.85 * cloud[i-1] + 0.15 * np.random.rand()
        solar = np.clip(solar * (0.3 + 0.7 * cloud) * (0.105 / max(0.01, (solar * (0.3 + 0.7 * cloud)).mean())), 0, 1)
        
        # WIND - Weather-driven with autocorrelation
        seasonal_wind = 0.28 + 0.10 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
        seasonal_wind = np.repeat(seasonal_wind[:365], 24)[:hours]
        
        n_days = hours // 24 + 1
        daily_weather = np.random.weibull(2.0, n_days)
        for i in range(1, n_days):
            daily_weather[i] = 0.65 * daily_weather[i-1] + 0.35 * daily_weather[i]
        weather = np.interp(t, np.arange(n_days) * 24, daily_weather)
        weather = (weather - weather.min()) / (weather.max() - weather.min() + 0.01)
        
        onshore = np.clip(seasonal_wind * (0.25 + 0.75 * weather) + 0.05 * np.random.randn(hours), 0.02, 0.95)
        onshore = np.clip(onshore * (0.265 / onshore.mean()), 0.02, 0.95)
        
        seasonal_off = 0.42 + 0.06 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
        seasonal_off = np.repeat(seasonal_off[:365], 24)[:hours]
        offshore = np.clip(seasonal_off * (0.35 + 0.65 * weather) + 0.03 * np.random.randn(hours), 0.05, 0.98)
        offshore = np.clip(offshore * (0.40 / offshore.mean()), 0.05, 0.98)
        
        print(f"    Demand: {demand.mean():.1f} GW mean, {demand.max():.1f} GW peak")
        print(f"    Solar CF: {solar.mean()*100:.1f}%, Onshore: {onshore.mean()*100:.1f}%, Offshore: {offshore.mean()*100:.1f}%")
        
        return {
            'demand': demand,
            'solar': solar,
            'onshore_wind': onshore,
            'offshore_wind': offshore,
            'source': 'Synthetic (calibrated to DUKES/ESPENI)'
        }


# ============================================================================
# LEARNING CURVES
# ============================================================================

def annuity_factor(lifetime: int, rate: float) -> float:
    """Calculate annuity factor for capital cost annualization."""
    if rate == 0:
        return 1 / lifetime
    return rate / (1 - (1 + rate) ** (-lifetime))


def wright_law_cost(C0: float, K: float, K0: float, learning_rate: float) -> float:
    """Calculate cost using Wright's Law: C = C0 * (K/K0)^(-alpha)."""
    if learning_rate <= 0 or K <= K0:
        return C0
    alpha = -np.log2(1 - learning_rate)
    return C0 * (K / K0) ** (-alpha)


def get_technology_cost(tech: Technology, year: int, cumulative_global: float,
                        learning_type: str) -> float:
    """Get technology cost based on learning type."""
    
    if learning_type == "exogenous":
        # EXOGENOUS: Costs follow predetermined global trajectory
        # UK deployment doesn't affect costs - we're a "price taker"
        # Global additions based on IEA/IRENA projections
        global_growth_rates = {
            "solar": 400,           # GW/year - based on IEA NZE scenario
            "onshore_wind": 100,    # GW/year
            "offshore_wind": 30,    # GW/year  
            "battery": 200,         # GW/year - rapid growth expected
            "gas_ccgt": 20,         # GW/year - some growth in developing world
            "gas_peaker": 5,
            "nuclear": 5,           # GW/year - modest growth
        }
        years_from_2024 = year - 2025
        growth = global_growth_rates.get(tech.name, 10)
        projected_global = tech.global_capacity_2024 + growth * years_from_2024
        return wright_law_cost(tech.capital_cost, projected_global, 
                               tech.global_capacity_2024, tech.learning_rate)
    else:
        # ENDOGENOUS: Costs depend on actual cumulative deployment
        # UK deployment drives global deployment (via UK_GLOBAL_SHARE)
        # This creates feedback: cheaper tech → more deployment → even cheaper
        return wright_law_cost(tech.capital_cost, cumulative_global,
                               tech.global_capacity_2024, tech.learning_rate)


def get_carbon_price(policy: Policy, year: int) -> float:
    """Get carbon price for a given year, interpolating if needed."""
    if not policy.carbon_price:
        return 0.0
    # If exact year exists, return it
    if year in policy.carbon_price:
        return policy.carbon_price[year]

    # Otherwise interpolate between nearest years
    years = sorted(policy.carbon_price.keys())
    if len(years) == 0:
        return 0.0
    # Before first year
    if year <= years[0]:
        return policy.carbon_price[years[0]]

    # After last year
    if year >= years[-1]:
        return policy.carbon_price[years[-1]]

    # Find surrounding years and interpolate
    for i, y in enumerate(years):
        if y > year:
            y0, y1 = years[i - 1], y
            p0, p1 = policy.carbon_price[y0], policy.carbon_price[y1]
            # Linear interpolation
            return p0 + (p1 - p0) * (year - y0) / (y1 - y0)

    return 0.0
# ============================================================================
# PyPSA NETWORK BUILDER
# ============================================================================

def build_network(
    year: int,
    data: Dict[str, np.ndarray],
    costs: Dict[str, float],
    policy: Policy,
    existing: Dict[str, float],
    sampling: int = 6
) -> pypsa.Network:
    """Build PyPSA network for capacity expansion optimization."""
    
    n = pypsa.Network()
    
    # Time setup - sample every N hours for speed
    hours = len(data['demand'])
    sampled_hours = hours // sampling
    snapshots = pd.date_range(f"{year}-01-01", periods=sampled_hours, freq=f"{sampling}h")
    n.set_snapshots(snapshots)
    n.snapshot_weightings.loc[:, :] = sampling
    
    # Single bus (copper plate)
    n.add("Bus", "GB")
    
    # Demand (GW to MW)
    sampled_demand = data['demand'][::sampling][:sampled_hours] * 1000
    n.add("Load", "demand", bus="GB", p_set=sampled_demand)
    
    # Carbon price for this year
    carbon_price = get_carbon_price(policy, year)
    
    # ---- GENERATORS ----
    
    for tech_name in ["gas_ccgt", "gas_peaker", "nuclear"]:
        tech = TECHNOLOGIES[tech_name]
        existing_cap = existing.get(tech_name, 0)
        # Max this period = existing + what can be built in one period
        max_cap = existing_cap + tech.max_build_per_period
        
        # Apply carbon price to gas
        mc = tech.marginal_cost
        if "gas" in tech_name:
            print(
                f"          Carbon Price: £{carbon_price}")
            mc += carbon_price * tech.co2_intensity

        
        n.add("Generator", tech_name,
            bus="GB",
            carrier=tech_name,
            p_nom_extendable=True,
            p_nom_min=existing_cap * 1000,
            p_nom_max=max_cap * 1000,
            capital_cost=costs[tech_name] * annuity_factor(tech.lifetime, DISCOUNT_RATE) * 1000,
            marginal_cost=mc,
            p_max_pu=tech.capacity_factor if tech_name == "nuclear" else 1.0,
        )
    
    # Variable renewables
    for tech_name, profile_key in [("solar", "solar"), ("onshore_wind", "onshore_wind"), ("offshore_wind", "offshore_wind")]:
        tech = TECHNOLOGIES[tech_name]
        existing_cap = existing.get(tech_name, 0)
        # Max this period = existing + what can be built in one period
        max_cap = existing_cap + tech.max_build_per_period
        
        # Apply subsidy
        subsidy = policy.renewable_subsidy.get(tech_name, {}).get(year, 0)
        effective_cost = max(0, costs[tech_name] - subsidy)
        
        profile = data[profile_key][::sampling][:sampled_hours]
        
        n.add("Generator", tech_name,
            bus="GB",
            carrier=tech_name,
            p_nom_extendable=True,
            p_nom_min=existing_cap * 1000,
            p_nom_max=max_cap * 1000,
            capital_cost=effective_cost * annuity_factor(tech.lifetime, DISCOUNT_RATE) * 1000,
            marginal_cost=0,
            p_max_pu=profile,
        )
    
    # Battery storage
    tech = TECHNOLOGIES["battery"]
    existing_bat = existing.get("battery", 0)
    max_bat = existing_bat + tech.max_build_per_period

    battery_subsidy = policy.renewable_subsidy.get("battery", {}).get(year, 0)
    effective_battery_cost = max(0, costs["battery"] - battery_subsidy)
    if battery_subsidy > 0:
        print(
            f"    Battery: £{costs['battery']:.0f}/kW - £{battery_subsidy:.0f} subsidy = £{effective_battery_cost:.0f}/kW")

    n.add("StorageUnit", "battery",
        bus="GB",
        carrier="battery",
        p_nom_extendable=True,
        p_nom_min=existing_bat * 1000,
        p_nom_max=max_bat * 1000,
        capital_cost=effective_battery_cost * annuity_factor(tech.lifetime, DISCOUNT_RATE) * 1000,
        marginal_cost=tech.marginal_cost,
        max_hours=2,
        efficiency_store=0.90,
        efficiency_dispatch=0.90,
        cyclic_state_of_charge=True,
    )
    
    return n


# ============================================================================
# SIMULATION ENGINE
# ============================================================================

def run_scenario(
    learning_type: str,
    policy: Policy,
    data_loader: GBDataLoader,
    weather_year: int = 2019
) -> pd.DataFrame:
    """Run myopic capacity expansion for one scenario."""
    
    # Load data
    data = data_loader.load(weather_year)
    
    # Initialize tracking
    existing = EXISTING_CAPACITY.copy()
    cumulative_global = {tech: TECHNOLOGIES[tech].global_capacity_2024 for tech in TECHNOLOGIES}
    
    results = []
    
    for period in PERIODS:
        # Scale demand for growth (1% per year from 2025)
        growth_factor = 1.01 ** (period - 2025)
        period_data = data.copy()
        period_data['demand'] = data['demand'] * growth_factor
        
        # Get costs
        costs = {}
        for tech_name, tech in TECHNOLOGIES.items():
            costs[tech_name] = get_technology_cost(tech, period, cumulative_global[tech_name], learning_type)
        if period in [2025, 2027,2029,2031,2033, 2035, 2050]:
            print(f"  {learning_type} costs in {period}:")
            print(f"    Solar: £{costs['solar']:.0f}/kW")
            print(f"    Offshore: £{costs['offshore_wind']:.0f}/kW")
            print(f"    Onshore: £{costs['onshore_wind']:.0f}/kW")
            # DEBUG: Show gas cost with carbon price

        # Build and optimize

        n = build_network(period, period_data, costs, policy, existing)
        # DEBUG: Check if model makes sense
        print(f"\n  DEBUG for {period}:")
        print(f"    Peak demand: {n.loads_t.p_set.max().max() / 1000:.1f} GW")
        print(f"    Total generator capacity: {n.generators.p_nom_max.sum() / 1000:.1f} GW")
        carbon_price = get_carbon_price(policy, period)  # You need this function
        gas_tech = TECHNOLOGIES["gas_ccgt"]

        # Effective marginal cost = fuel/O&M + carbon price × CO2 intensity
        effective_gas_cost = gas_tech.marginal_cost + carbon_price * gas_tech.co2_intensity



        # Check each generator's max possible output
        for gen in n.generators.index:
            p_nom = n.generators.p_nom_max[gen]
            if gen in n.generators_t.p_max_pu.columns:
                max_pu = n.generators_t.p_max_pu[gen].max()
                mean_pu = n.generators_t.p_max_pu[gen].mean()
            else:
                max_pu = 1.0
                mean_pu = 1.0
            print(f"    {gen}: {p_nom / 1000:.1f} GW, max_pu={max_pu:.2f}, mean_pu={mean_pu:.2f}")
        status = n.optimize(solver_name="highs", solver_options={"output_flag": False})
        print(f"\n  Optimal capacities for {period}:")
        for gen in n.generators.index:
            p_nom_opt = n.generators.loc[gen, 'p_nom_opt'] / 1000  # GW
            p_nom_min = n.generators.loc[gen, 'p_nom_min'] / 1000
            p_nom_max = n.generators.loc[gen, 'p_nom_max'] / 1000
            print(f"    {gen}: {p_nom_opt:.1f} GW (min={p_nom_min:.1f}, max={p_nom_max:.1f})")
        if status[0] != "ok":
            print(f"  Warning: Optimization failed for {period}")
            continue
        
        # Extract results
        new_capacity = {}
        for gen in n.generators.index:
            new_cap = n.generators.p_nom_opt[gen] / 1000 - existing.get(gen, 0)
            new_capacity[gen] = max(0, new_cap)
            existing[gen] = n.generators.p_nom_opt[gen] / 1000
        
        if "battery" in n.storage_units.index:
            new_capacity["battery"] = max(0, n.storage_units.p_nom_opt["battery"] / 1000 - existing.get("battery", 0))
            existing["battery"] = n.storage_units.p_nom_opt["battery"] / 1000
        
        # Update global cumulative capacity based on UK deployment
        # KEY ASSUMPTION: UK deployment is proportional to global deployment
        # If UK builds ΔK, global builds ΔK / UK_SHARE
        for tech_name in TECHNOLOGIES:
            if tech_name in new_capacity and new_capacity[tech_name] > 0:
                uk_share = UK_GLOBAL_SHARE.get(tech_name, 0.02)
                global_addition = new_capacity[tech_name] / uk_share
                cumulative_global[tech_name] += global_addition
        
        # Calculate emissions
        gas_gen = sum(n.generators_t.p.get(g, pd.Series([0])).sum() for g in ["gas_ccgt", "gas_peaker"]) * 6 / 1e6
        emissions = gas_gen * 0.37  # Average intensity
        subsidy_spent = 0.0

        # Renewable subsidies
        for tech_name in ["solar", "onshore_wind", "offshore_wind"]:
            new_capacity_mw = n.generators.loc[tech_name, 'p_nom_opt'] - n.generators.loc[tech_name, 'p_nom_min']
            new_capacity_gw = new_capacity_mw / 1000
            subsidy_rate = policy.renewable_subsidy.get(tech_name, {}).get(period, 0)
            tech_subsidy = new_capacity_gw * subsidy_rate * 1000  # £m (GW * £/kW * 1000)
            subsidy_spent += tech_subsidy

        # Battery subsidy
        new_battery_mw = n.storage_units.loc['battery', 'p_nom_opt'] - n.storage_units.loc['battery', 'p_nom_min']
        new_battery_gw = new_battery_mw / 1000
        battery_subsidy_rate = policy.renewable_subsidy.get("battery", {}).get(period, 0)
        battery_subsidy = new_battery_gw * battery_subsidy_rate * 1000  # £m
        subsidy_spent += battery_subsidy

        subsidy_spent_bn = subsidy_spent / 1000
        # Store results
        results.append({
            "period": period,
            "learning_type": learning_type,
            "policy": policy.name,
            "solar_gw": existing.get("solar", 0),
            "onshore_gw": existing.get("onshore_wind", 0),
            "offshore_gw": existing.get("offshore_wind", 0),
            "gas_gw": existing.get("gas_ccgt", 0) + existing.get("gas_peaker", 0),
            "nuclear_gw": existing.get("nuclear", 0),
            "battery_gw": existing.get("battery", 0),
            "solar_cost": costs["solar"],
            "offshore_cost": costs["offshore_wind"],
            "battery_cost": costs["battery"],
            "emissions_mt": emissions,
            "system_cost_bn": n.objective / 1e9,
            "subsidy_spent_bn": subsidy_spent_bn,
        })
    
    return pd.DataFrame(results)


def run_all_scenarios(data_loader: GBDataLoader, weather_year: int = 2019) -> pd.DataFrame:
    """Run all combinations of learning types and policies."""
    
    all_results = []
    
    scenarios = [(lt, p) for lt in ["exogenous", "endogenous"] for p in POLICIES.values()]
    total = len(scenarios)
    
    for i, (learning_type, policy) in enumerate(scenarios):
        print(f"\n[{i+1}/{total}] {learning_type.upper()} + {policy.name}")
        df = run_scenario(learning_type, policy, data_loader, weather_year)
        all_results.append(df)
    
    return pd.concat(all_results, ignore_index=True)


# ============================================================================
# ANALYSIS & VISUALIZATION
# ============================================================================

def analyze_results(results: pd.DataFrame) -> pd.DataFrame:
    """Calculate summary metrics."""
    
    summary = results.groupby(["learning_type", "policy"]).agg({
        "solar_gw": "last",
        "onshore_gw": "last", 
        "offshore_gw": "last",
        "gas_gw": "last",
        "battery_gw": "last",
        "emissions_mt": "sum",
        "system_cost_bn": "sum",
        "subsidy_spent_bn": "sum",
    }).reset_index()
    
    summary["total_renewable_gw"] = summary["solar_gw"] + summary["onshore_gw"] + summary["offshore_gw"]
    
    return summary


def plot_results(results: pd.DataFrame, summary: pd.DataFrame, save_prefix: str = "gb_model"):
    """Generate visualization plots."""
    
    # Plot 1: Capacity evolution
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    policy = "Current Policy"
    techs = [
        ("offshore_gw", "Offshore Wind", "tab:blue"),
        ("onshore_gw", "Onshore Wind", "tab:cyan"),
        ("solar_gw", "Solar PV", "tab:orange"),
        ("battery_gw", "Battery", "tab:green"),
        ("gas_gw", "Gas", "tab:gray"),
        ("nuclear_gw", "Nuclear", "tab:purple"),
    ]
    
    for ax, (col, name, color) in zip(axes.flatten(), techs):
        for lt, style in [("exogenous", "--"), ("endogenous", "-")]:
            data = results[(results["learning_type"] == lt) & (results["policy"] == policy)]
            ax.plot(data["period"], data[col], linestyle=style, marker="o",
                   linewidth=2, color=color, label=lt.capitalize())
        ax.set_xlabel("Year")
        ax.set_ylabel("Capacity (GW)")
        ax.set_title(name, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f"GB Capacity Evolution: {policy}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_capacity.png", dpi=300, bbox_inches="tight")
    print(f"  Saved: {save_prefix}_capacity.png")
    
    # Plot 2: Policy comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    policies = [p for p in summary["policy"].unique() if p != "No Policy"]
    x = np.arange(len(policies))
    width = 0.35
    
    exog = summary[summary["learning_type"] == "exogenous"].set_index("policy")
    endog = summary[summary["learning_type"] == "endogenous"].set_index("policy")
    
    # Emissions
    ax = axes[0]
    ax.bar(x - width/2, [exog.loc[p, "emissions_mt"] for p in policies], width, label="Exogenous", color="tab:blue", alpha=0.7)
    ax.bar(x + width/2, [endog.loc[p, "emissions_mt"] for p in policies], width, label="Endogenous", color="tab:orange", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Cumulative Emissions (Mt CO2)")
    ax.set_title("Total Emissions 2025-2050", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    # Renewable capacity
    ax = axes[1]
    ax.bar(x - width/2, [exog.loc[p, "total_renewable_gw"] for p in policies], width, label="Exogenous", color="tab:blue", alpha=0.7)
    ax.bar(x + width/2, [endog.loc[p, "total_renewable_gw"] for p in policies], width, label="Endogenous", color="tab:orange", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Capacity (GW)")
    ax.set_title("Total Renewable Capacity 2050", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    # System cost
    ax = axes[2]
    ax.bar(x - width/2, [exog.loc[p, "system_cost_bn"] for p in policies], width, label="Exogenous", color="tab:blue", alpha=0.7)
    ax.bar(x + width/2, [endog.loc[p, "system_cost_bn"] for p in policies], width, label="Endogenous", color="tab:orange", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Cost (£ Billion)")
    ax.set_title("Total System Cost", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    fig.suptitle("Policy Comparison: Exogenous vs Endogenous Learning", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_policy.png", dpi=300, bbox_inches="tight")
    print(f"  Saved: {save_prefix}_policy.png")
    
    # Plot 3: Cost trajectories
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    for ax, (col, name, color) in zip(axes, [("solar_cost", "Solar", "tab:orange"), 
                                              ("offshore_cost", "Offshore Wind", "tab:blue"),
                                              ("battery_cost", "Battery", "tab:green")]):
        for lt, style in [("exogenous", "--"), ("endogenous", "-")]:
            data = results[(results["learning_type"] == lt) & (results["policy"] == policy)]
            ax.plot(data["period"], data[col], linestyle=style, marker="o", linewidth=2, color=color, label=lt.capitalize())
        ax.set_xlabel("Year")
        ax.set_ylabel("Cost (£/kW)")
        ax.set_title(f"{name} Cost", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle("Technology Cost Trajectories", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_costs.png", dpi=300, bbox_inches="tight")
    print(f"  Saved: {save_prefix}_costs.png")
    
    plt.close('all')


def find_minimum_solar_battery_subsidy(
        data_loader: GBDataLoader,
        weather_year: int = 2019,
        battery_range: tuple = (0, 600, 100),  # (start, stop, step) in £/kW
        solar_range: tuple = (0, 300, 50),  # (start, stop, step) in £/kW
        target_gas_growth: float = 0.5,
        learning_type: str = "endogenous",
        taper_to_zero_by: int = 2045
) -> pd.DataFrame:
    """
    Find minimum combined solar + battery subsidy to eliminate gas growth.
    Tests all combinations in a grid search.
    """

    print("\n" + "=" * 70)
    print("SOLAR + BATTERY SUBSIDY GRID SEARCH")
    print(f"Finding minimum combined subsidy to eliminate gas growth ({learning_type})")
    print(f"Subsidies taper to zero by {taper_to_zero_by}")
    print("=" * 70)

    def create_tapering_subsidy(initial_subsidy: float, taper_end_year: int) -> Dict[int, float]:
        """Create a subsidy schedule that tapers from initial value to zero."""
        subsidy_schedule = {}
        taper_start = 2027  # Start tapering after this year
        taper_end_year = 2029

        for year in PERIODS:
            if year >= taper_end_year:
                subsidy_schedule[year] = 0
            elif year <= taper_start:
                subsidy_schedule[year] = initial_subsidy
            else:
                # Linear taper from taper_start to taper_end_year
                years_into_taper = year - taper_start
                years_total = taper_end_year - taper_start
                subsidy_schedule[year] = initial_subsidy * (1 - years_into_taper / years_total)

        return subsidy_schedule

    # Generate all combinations
    bat_start, bat_stop, bat_step = battery_range
    sol_start, sol_stop, sol_step = solar_range

    battery_levels = list(range(bat_start, bat_stop + 1, bat_step))
    solar_levels = list(range(sol_start, sol_stop + 1, sol_step))

    total_combos = len(battery_levels) * len(solar_levels)
    print(f"\nTesting {len(battery_levels)} battery x {len(solar_levels)} solar = {total_combos} combinations")

    results = []
    combo_num = 0

    for bat_subsidy in battery_levels:
        for sol_subsidy in solar_levels:
            combo_num += 1
            print(f"\n[{combo_num}/{total_combos}] Battery: £{bat_subsidy}/kW, Solar: £{sol_subsidy}/kW")

            # Create tapering schedules
            battery_schedule = create_tapering_subsidy(bat_subsidy, taper_to_zero_by)
            solar_schedule = create_tapering_subsidy(sol_subsidy, taper_to_zero_by)

            # Create policy
            test_policy = Policy(
                name=f"Bat{bat_subsidy}_Sol{sol_subsidy}",
                # carbon_price={y: 50 + (y - 2025) * 6 for y in PERIODS},
                carbon_price={},
                renewable_subsidy={
                    "battery": battery_schedule,
                    "solar": solar_schedule,
                }
            )

            # Run scenario
            df = run_scenario(learning_type, test_policy, data_loader, weather_year)

            # Extract results
            gas_growth = df.iloc[-1]['gas_gw'] - df.iloc[0]['gas_gw']
            bat_growth = df.iloc[-1]['battery_gw'] - df.iloc[0]['battery_gw']
            sol_growth = df.iloc[-1]['solar_gw'] - df.iloc[0]['solar_gw']

            # Calculate total initial subsidy (simple metric for comparison)
            total_initial_subsidy = bat_subsidy + sol_subsidy
            subsidy_spent = 0.0

            total_subsidy_cost_bn = 0

            prev_bat = df.iloc[0]['battery_gw']
            prev_sol = df.iloc[0]['solar_gw']

            for i, row in df.iterrows():
                if i == 0:
                    continue  # Skip first period (no new builds to count)

                new_bat = max(0, row['battery_gw'] - prev_bat)
                new_sol = max(0, row['solar_gw'] - prev_sol)

                # £/kW × GW / 1000 = £bn
                total_subsidy_cost_bn += battery_schedule.get(row['period'], 0) * new_bat / 1000
                total_subsidy_cost_bn += solar_schedule.get(row['period'], 0) * new_sol / 1000

                prev_bat = row['battery_gw']
                prev_sol = row['solar_gw']
            results.append({
                'battery_initial': bat_subsidy,
                'solar_initial': sol_subsidy,
                'total_initial_subsidy': total_initial_subsidy,
                'gas_growth_gw': gas_growth,
                'battery_growth_gw': bat_growth,
                'solar_growth_gw': sol_growth,
                'gas_2049_gw': df.iloc[-1]['gas_gw'],
                'battery_2049_gw': df.iloc[-1]['battery_gw'],
                'solar_2049_gw': df.iloc[-1]['solar_gw'],
                'total_emissions_mt': df['emissions_mt'].sum(),
                'total_system_cost_bn': df['system_cost_bn'].sum(),
                'total_subsidy_cost_bn': total_subsidy_cost_bn,
                'meets_target': gas_growth <= target_gas_growth,


            })

            status = "✓" if gas_growth <= target_gas_growth else "✗"
            print(f"    Gas growth: {gas_growth:+.1f} GW {status}")

    results_df = pd.DataFrame(results)

    # Find minimum subsidy combinations that meet target
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    successful = results_df[results_df['meets_target'] == True].copy()

    if len(successful) > 0:
        # Find minimum total subsidy
        min_idx = successful['total_initial_subsidy'].idxmin()
        best = successful.loc[min_idx]

        print(f"\n*** MINIMUM COMBINED SUBSIDY TO ELIMINATE GAS GROWTH ***")
        print(f"    Battery: £{best['battery_initial']:.0f}/kW (initial)")
        print(f"    Solar:   £{best['solar_initial']:.0f}/kW (initial)")
        print(f"    Total:   £{best['total_initial_subsidy']:.0f}/kW")
        print(f"    Gas growth: {best['gas_growth_gw']:+.1f} GW")
        print(f"    Battery 2049: {best['battery_2049_gw']:.1f} GW")
        print(f"    Solar 2049: {best['solar_2049_gw']:.1f} GW")


        # Show all successful combinations sorted by total subsidy
        print(f"\nAll successful combinations (sorted by total subsidy):")
        successful_sorted = successful.sort_values('total_initial_subsidy')
        print(successful_sorted[['battery_initial', 'solar_initial', 'total_initial_subsidy',
                                 'gas_growth_gw', 'battery_2049_gw', 'solar_2049_gw','total_subsidy_cost_bn']].to_string(index=False))
    else:
        print(f"\n*** No combination achieved ≤{target_gas_growth} GW gas growth ***")
        print("*** Try increasing subsidy ranges ***")

    # Generate heatmap
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Reshape for heatmap
    pivot_gas = results_df.pivot(index='battery_initial', columns='solar_initial', values='gas_growth_gw')
    pivot_bat = results_df.pivot(index='battery_initial', columns='solar_initial', values='battery_2049_gw')
    pivot_sol = results_df.pivot(index='battery_initial', columns='solar_initial', values='solar_2049_gw')

    # Panel 1: Gas growth heatmap
    ax = axes[0]
    im = ax.imshow(pivot_gas.values, cmap='RdYlGn_r', aspect='auto', origin='lower')
    ax.set_xticks(range(len(solar_levels)))
    ax.set_xticklabels(solar_levels)
    ax.set_yticks(range(len(battery_levels)))
    ax.set_yticklabels(battery_levels)
    ax.set_xlabel('Solar Initial Subsidy (£/kW)')
    ax.set_ylabel('Battery Initial Subsidy (£/kW)')
    ax.set_title('A. Gas Growth (GW)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='GW')

    # Add contour line at target
    cs = ax.contour(pivot_gas.values, levels=[target_gas_growth], colors='black', linewidths=2)
    ax.clabel(cs, fmt=f'{target_gas_growth} GW')

    # Panel 2: Battery 2049 capacity
    ax = axes[1]
    im = ax.imshow(pivot_bat.values, cmap='Blues', aspect='auto', origin='lower')
    ax.set_xticks(range(len(solar_levels)))
    ax.set_xticklabels(solar_levels)
    ax.set_yticks(range(len(battery_levels)))
    ax.set_yticklabels(battery_levels)
    ax.set_xlabel('Solar Initial Subsidy (£/kW)')
    ax.set_ylabel('Battery Initial Subsidy (£/kW)')
    ax.set_title('B. Battery 2049 Capacity (GW)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='GW')

    # Panel 3: Solar 2049 capacity
    ax = axes[2]
    im = ax.imshow(pivot_sol.values, cmap='Oranges', aspect='auto', origin='lower')
    ax.set_xticks(range(len(solar_levels)))
    ax.set_xticklabels(solar_levels)
    ax.set_yticks(range(len(battery_levels)))
    ax.set_yticklabels(battery_levels)
    ax.set_xlabel('Solar Initial Subsidy (£/kW)')
    ax.set_ylabel('Battery Initial Subsidy (£/kW)')
    ax.set_title('C. Solar 2049 Capacity (GW)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='GW')

    plt.suptitle(
        f'Solar + Battery Subsidy Grid Search ({learning_type.capitalize()})\nSubsidies taper to £0 by {taper_to_zero_by}',
        fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(f'solar_battery_subsidy_grid_{learning_type}.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: solar_battery_subsidy_grid_{learning_type}.png")

    results_df.to_csv(f'solar_battery_subsidy_grid_{learning_type}.csv', index=False)
    print(f"Saved: solar_battery_subsidy_grid_{learning_type}.csv")

    return results_df


def find_minimum_carbon_tax(
        data_loader: GBDataLoader,
        weather_year: int = 2019,
        price_range: tuple = (0, 500, 25),  # (start, stop, step) in £/tCO2
        target_gas_growth: float = 0.5,
        learning_type: str = "endogenous",
) -> pd.DataFrame:
    """
    Find minimum carbon tax to eliminate gas growth.
    Tests three trajectory types:
    1. Constant: Same price throughout
    2. Front-loaded (Head): Starts high, tapers to 50% by 2049
    3. Back-loaded (Tail): Starts at 50%, increases to full by 2049
    """

    print("\n" + "=" * 70)
    print("CARBON TAX SWEEP")
    print(f"Finding minimum carbon tax to eliminate gas growth ({learning_type})")
    print("=" * 70)

    def create_constant_carbon(price: float) -> Dict[int, float]:
        """Constant carbon price throughout."""
        return {year: price for year in PERIODS}

    def create_frontloaded_carbon(max_price: float) -> Dict[int, float]:
        """Starts at max_price, tapers to 50% by 2049."""
        schedule = {}
        start_year = PERIODS[0]
        end_year = PERIODS[-1]

        for year in PERIODS:
            progress = (year - start_year) / (end_year - start_year)
            # Linear taper from 100% to 50%
            multiplier = 1.0 - (0.5 * progress)
            schedule[year] = max_price * multiplier

        return schedule

    def create_backloaded_carbon(max_price: float) -> Dict[int, float]:
        """Starts at 50% of max, increases to max by 2049."""
        schedule = {}
        start_year = PERIODS[0]
        end_year = PERIODS[-1]

        for year in PERIODS:
            progress = (year - start_year) / (end_year - start_year)
            # Linear increase from 50% to 100%
            multiplier = 0.5 + (0.5 * progress)
            schedule[year] = max_price * multiplier

        return schedule

    # Show example schedules
    print("\nExample carbon price schedules (for £200/tCO2 parameter):")
    print("-" * 60)
    print(f"{'Year':<8} {'Constant':<12} {'Front-loaded':<14} {'Back-loaded':<12}")
    print("-" * 60)
    example_const = create_constant_carbon(200)
    example_front = create_frontloaded_carbon(200)
    example_back = create_backloaded_carbon(200)
    for year in PERIODS:
        print(f"{year:<8} £{example_const[year]:<11.0f} £{example_front[year]:<13.0f} £{example_back[year]:<11.0f}")
    print("-" * 60)

    # Calculate average carbon price for comparison
    def avg_carbon_price(schedule: Dict[int, float]) -> float:
        return sum(schedule.values()) / len(schedule)

    start, stop, step = price_range
    price_levels = list(range(start, stop + 1, step))

    trajectory_types = [
        ("constant", create_constant_carbon),
        ("front_loaded", create_frontloaded_carbon),
        ("back_loaded", create_backloaded_carbon),
    ]

    total_runs = len(price_levels) * len(trajectory_types)
    print(f"\nTesting {len(price_levels)} price levels × {len(trajectory_types)} trajectories = {total_runs} runs")

    results = []
    run_num = 0

    for price_param in price_levels:
        for traj_name, traj_func in trajectory_types:
            run_num += 1

            carbon_schedule = traj_func(price_param)
            avg_price = avg_carbon_price(carbon_schedule)

            print(f"\n[{run_num}/{total_runs}] {traj_name}: £{price_param}/tCO2 (avg: £{avg_price:.0f}/tCO2)")

            # Create policy with this carbon schedule and NO renewable subsidies
            test_policy = Policy(
                name=f"Carbon_{traj_name}_{price_param}",
                carbon_price=carbon_schedule,
                renewable_subsidy={}  # No subsidies - carbon only
            )

            # Run scenario
            df = run_scenario(learning_type, test_policy, data_loader, weather_year)

            # Extract results
            gas_growth = df.iloc[-1]['gas_gw'] - df.iloc[0]['gas_gw']
            bat_growth = df.iloc[-1]['battery_gw'] - df.iloc[0]['battery_gw']
            solar_growth = df.iloc[-1]['solar_gw'] - df.iloc[0]['solar_gw']
            offshore_growth = df.iloc[-1]['offshore_gw'] - df.iloc[0]['offshore_gw']

            results.append({
                'trajectory': traj_name,
                'price_parameter': price_param,
                'avg_carbon_price': avg_price,
                'carbon_2025': carbon_schedule[2025],
                'carbon_2049': carbon_schedule[2049],
                'gas_growth_gw': gas_growth,
                'battery_growth_gw': bat_growth,
                'solar_growth_gw': solar_growth,
                'offshore_growth_gw': offshore_growth,
                'gas_2049_gw': df.iloc[-1]['gas_gw'],
                'battery_2049_gw': df.iloc[-1]['battery_gw'],
                'solar_2049_gw': df.iloc[-1]['solar_gw'],
                'offshore_2049_gw': df.iloc[-1]['offshore_gw'],
                'total_emissions_mt': df['emissions_mt'].sum(),
                'total_system_cost_bn': df['system_cost_bn'].sum(),
                'meets_target': gas_growth <= target_gas_growth,
            })

            status = "✓" if gas_growth <= target_gas_growth else "✗"
            print(f"    Gas growth: {gas_growth:+.1f} GW {status}")

    results_df = pd.DataFrame(results)

    # Find minimum for each trajectory type
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nMinimum carbon price to achieve ≤{target_gas_growth} GW gas growth:")
    print("-" * 70)

    best_overall = None
    best_overall_avg = float('inf')

    for traj_name in ["constant", "front_loaded", "back_loaded"]:
        traj_results = results_df[results_df['trajectory'] == traj_name]
        successful = traj_results[traj_results['meets_target'] == True]

        if len(successful) > 0:
            # Find minimum by average price
            min_idx = successful['avg_carbon_price'].idxmin()
            best = successful.loc[min_idx]

            print(f"\n{traj_name.upper()}:")
            print(f"    Price parameter: £{best['price_parameter']:.0f}/tCO2")
            print(f"    2025 price: £{best['carbon_2025']:.0f}/tCO2")
            print(f"    2049 price: £{best['carbon_2049']:.0f}/tCO2")
            print(f"    Average price: £{best['avg_carbon_price']:.0f}/tCO2")
            print(f"    Gas growth: {best['gas_growth_gw']:+.1f} GW")

            if best['avg_carbon_price'] < best_overall_avg:
                best_overall = best
                best_overall_avg = best['avg_carbon_price']
        else:
            print(f"\n{traj_name.upper()}: No price level achieved target")

    if best_overall is not None:
        print(f"\n{'=' * 70}")
        print(f"*** BEST OVERALL: {best_overall['trajectory'].upper()} ***")
        print(f"    Average carbon price: £{best_overall['avg_carbon_price']:.0f}/tCO2")
        print(f"    2025: £{best_overall['carbon_2025']:.0f}/tCO2 → 2049: £{best_overall['carbon_2049']:.0f}/tCO2")
        print(f"{'=' * 70}")

    # Generate plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {'constant': 'blue', 'front_loaded': 'green', 'back_loaded': 'red'}
    markers = {'constant': 'o', 'front_loaded': '^', 'back_loaded': 's'}

    # Panel 1: Gas growth vs average carbon price
    ax = axes[0, 0]
    for traj_name in trajectory_types:
        traj_data = results_df[results_df['trajectory'] == traj_name[0]]
        ax.plot(traj_data['avg_carbon_price'], traj_data['gas_growth_gw'],
                color=colors[traj_name[0]], marker=markers[traj_name[0]],
                linewidth=2, markersize=6, label=traj_name[0].replace('_', ' ').title())
    ax.axhline(y=target_gas_growth, color='black', linestyle='--', alpha=0.5, label=f'Target ({target_gas_growth} GW)')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Average Carbon Price (£/tCO2)', fontsize=11)
    ax.set_ylabel('Gas Capacity Growth (GW)', fontsize=11)
    ax.set_title('A. Gas Growth vs Carbon Price', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Battery growth vs average carbon price
    ax = axes[0, 1]
    for traj_name in trajectory_types:
        traj_data = results_df[results_df['trajectory'] == traj_name[0]]
        ax.plot(traj_data['avg_carbon_price'], traj_data['battery_growth_gw'],
                color=colors[traj_name[0]], marker=markers[traj_name[0]],
                linewidth=2, markersize=6, label=traj_name[0].replace('_', ' ').title())
    ax.set_xlabel('Average Carbon Price (£/tCO2)', fontsize=11)
    ax.set_ylabel('Battery Capacity Growth (GW)', fontsize=11)
    ax.set_title('B. Battery Growth vs Carbon Price', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Emissions vs average carbon price
    ax = axes[1, 0]
    for traj_name in trajectory_types:
        traj_data = results_df[results_df['trajectory'] == traj_name[0]]
        ax.plot(traj_data['avg_carbon_price'], traj_data['total_emissions_mt'],
                color=colors[traj_name[0]], marker=markers[traj_name[0]],
                linewidth=2, markersize=6, label=traj_name[0].replace('_', ' ').title())
    ax.set_xlabel('Average Carbon Price (£/tCO2)', fontsize=11)
    ax.set_ylabel('Total Emissions (MtCO2)', fontsize=11)
    ax.set_title('C. Total Emissions vs Carbon Price', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: System cost vs average carbon price
    ax = axes[1, 1]
    for traj_name in trajectory_types:
        traj_data = results_df[results_df['trajectory'] == traj_name[0]]
        ax.plot(traj_data['avg_carbon_price'], traj_data['total_system_cost_bn'],
                color=colors[traj_name[0]], marker=markers[traj_name[0]],
                linewidth=2, markersize=6, label=traj_name[0].replace('_', ' ').title())
    ax.set_xlabel('Average Carbon Price (£/tCO2)', fontsize=11)
    ax.set_ylabel('Total System Cost (£bn)', fontsize=11)
    ax.set_title('D. System Cost vs Carbon Price', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Carbon Tax Sweep ({learning_type.capitalize()} Learning)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'carbon_tax_sweep_{learning_type}.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: carbon_tax_sweep_{learning_type}.png")

    results_df.to_csv(f'carbon_tax_sweep_{learning_type}.csv', index=False)
    print(f"Saved: carbon_tax_sweep_{learning_type}.csv")

    return results_df

def find_minimum_battery_subsidy(
        data_loader: GBDataLoader,
        weather_year: int = 2019,
        subsidy_range: tuple = (300, 400, 5),  # (start, stop, step) for INITIAL subsidy
        target_gas_growth: float = 0.5,
        learning_type: str = "endogenous",
        taper_to_zero_by: int = 2045  # Year when subsidy reaches zero
) -> pd.DataFrame:
    """
    Find minimum battery subsidy to eliminate gas growth.
    Subsidy tapers down linearly from initial value to zero.
    """

    print("\n" + "=" * 70)
    print("BATTERY SUBSIDY SWEEP (TAPERING)")
    print(f"Finding minimum initial subsidy to eliminate gas growth ({learning_type})")
    print(f"Subsidies taper to zero by {taper_to_zero_by}")
    print("=" * 70)

    def create_tapering_subsidy(initial_subsidy: float, taper_end_year: int) -> Dict[int, float]:
        """Create a subsidy schedule that tapers from initial value to zero."""
        subsidy_schedule = {}
        start_year = PERIODS[0]  # 2025
        taper_start = 2027
        taper_end_year = 2029
        for year in PERIODS:
            if year >= taper_end_year:
                subsidy_schedule[year] = 0
            elif year <= taper_start:
                subsidy_schedule[year] = initial_subsidy
            else:
                # Linear taper from taper_start to taper_end_year
                years_into_taper = year - taper_start
                years_total = taper_end_year - taper_start
                subsidy_schedule[year] = initial_subsidy * (1 - years_into_taper / years_total)
        return subsidy_schedule

    start, stop, step = subsidy_range
    subsidy_levels = list(range(start, stop + 1, step))
    results = []

    for initial_subsidy in subsidy_levels:
        print(f"\n--- Testing initial battery subsidy: £{initial_subsidy}/kW (tapering) ---")

        # Create tapering subsidy schedule
        battery_subsidy_schedule = create_tapering_subsidy(initial_subsidy, taper_to_zero_by)

        # Show the schedule
        print(f"    Schedule: ", end="")
        sample_years = [2025, 2035, 2045, 2049]
        for y in sample_years:
            if y in battery_subsidy_schedule:
                print(f"{y}:£{battery_subsidy_schedule[y]:.0f}  ", end="")
        print()

        # Create policy with tapering subsidy
        test_policy = Policy(
            name=f"Battery_Subsidy_{initial_subsidy}",
            # carbon_price={y: 50 + (y - 2025) * 6 for y in PERIODS},
            carbon_price = {}, # Use only battery subsidy
            renewable_subsidy={
                "battery": battery_subsidy_schedule
            }
        )

        # Run scenario
        df = run_scenario(learning_type, test_policy, data_loader, weather_year)

        # Extract results
        total_subsidy_cost_bn = 0
        gas_start = df.iloc[0]['gas_gw']
        gas_end = df.iloc[-1]['gas_gw']
        gas_growth = gas_end - gas_start

        bat_start = df.iloc[0]['battery_gw']
        bat_end = df.iloc[-1]['battery_gw']
        bat_growth = bat_end - bat_start

        total_emissions = df['emissions_mt'].sum()
        total_cost = df['system_cost_bn'].sum()
        prev_bat = df.iloc[0]['battery_gw']
        for i, row in df.iterrows():
            if i == 0:
                continue  # Skip first period (no new builds to count)

            new_bat = max(0, row['battery_gw'] - prev_bat)

            # £/kW × GW / 1000 = £bn
            total_subsidy_cost_bn += battery_subsidy_schedule.get(row['period'], 0) * new_bat / 1000

            prev_bat = row['battery_gw']
        # Calculate total subsidy cost (rough estimate)
        # Sum of (subsidy * capacity built that year) - simplified


        results.append({
            'initial_subsidy_per_kw': initial_subsidy,
            'gas_start_gw': gas_start,
            'gas_end_gw': gas_end,
            'gas_growth_gw': gas_growth,
            'battery_start_gw': bat_start,
            'battery_end_gw': bat_end,
            'battery_growth_gw': bat_growth,
            'total_emissions_mt': total_emissions,
            'total_system_cost_bn': total_cost,
            'total_subsidy_cost_bn': total_subsidy_cost_bn,
        })

        print(f"    Gas: {gas_start:.1f} → {gas_end:.1f} GW (Δ = {gas_growth:+.1f} GW)")
        print(f"    Battery: {bat_start:.1f} → {bat_end:.1f} GW (Δ = {bat_growth:+.1f} GW)")

    results_df = pd.DataFrame(results)

    # Find minimum subsidy
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(results_df.to_string(index=False))

    min_subsidy = None
    for _, row in results_df.iterrows():
        if row['gas_growth_gw'] <= target_gas_growth:
            min_subsidy = row['initial_subsidy_per_kw']
            print(f"\n*** MINIMUM INITIAL SUBSIDY FOR ≤{target_gas_growth} GW GAS GROWTH: £{min_subsidy}/kW ***")
            print(f"*** (tapering to £0 by {taper_to_zero_by}) ***")
            break

    if min_subsidy is None:
        print(f"\n*** No subsidy level achieved ≤{target_gas_growth} GW gas growth ***")
        print(f"*** Try increasing max subsidy beyond £{stop}/kW ***")

    # Generate plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Gas vs Battery growth
    ax = axes[0]
    ax.plot(results_df['initial_subsidy_per_kw'], results_df['gas_growth_gw'],
            'r-o', linewidth=2, markersize=6, label='Gas Growth')
    ax.plot(results_df['initial_subsidy_per_kw'], results_df['battery_growth_gw'],
            'b-o', linewidth=2, markersize=6, label='Battery Growth')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.axhline(y=target_gas_growth, color='red', linestyle=':', alpha=0.5)
    if min_subsidy:
        ax.axvline(x=min_subsidy, color='green', linestyle='--', alpha=0.7,
                   label=f'Min subsidy (£{min_subsidy}/kW)')
    ax.set_xlabel('Initial Battery Subsidy (£/kW)', fontsize=11)
    ax.set_ylabel('Capacity Growth 2025-2049 (GW)', fontsize=11)
    ax.set_title('A. Capacity Growth vs Initial Subsidy', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Final capacities
    ax = axes[1]
    ax.plot(results_df['initial_subsidy_per_kw'], results_df['gas_end_gw'],
            'r-o', linewidth=2, markersize=6, label='Gas (2049)')
    ax.plot(results_df['initial_subsidy_per_kw'], results_df['battery_end_gw'],
            'b-o', linewidth=2, markersize=6, label='Battery (2049)')
    ax.set_xlabel('Initial Battery Subsidy (£/kW)', fontsize=11)
    ax.set_ylabel('Final Capacity (GW)', fontsize=11)
    ax.set_title('B. 2049 Capacity vs Initial Subsidy', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Emissions and cost
    ax = axes[2]
    ax2 = ax.twinx()
    line1 = ax.plot(results_df['initial_subsidy_per_kw'], results_df['total_emissions_mt'],
                    'g-o', linewidth=2, markersize=6, label='Total Emissions')
    line2 = ax2.plot(results_df['initial_subsidy_per_kw'], results_df['total_system_cost_bn'],
                     'm-s', linewidth=2, markersize=6, label='System Cost')
    ax.set_xlabel('Initial Battery Subsidy (£/kW)', fontsize=11)
    ax.set_ylabel('Total Emissions (MtCO2)', fontsize=11, color='green')
    ax2.set_ylabel('Total System Cost (£bn)', fontsize=11, color='purple')
    ax.set_title('C. Emissions & Cost vs Initial Subsidy', fontweight='bold')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Battery Subsidy Sweep - Tapering to £0 by {taper_to_zero_by} ({learning_type.capitalize()})',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'battery_subsidy_sweep_{learning_type}.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: battery_subsidy_sweep_{learning_type}.png")

    results_df.to_csv(f'battery_subsidy_sweep_{learning_type}.csv', index=False)
    print(f"Saved: battery_subsidy_sweep_{learning_type}.csv")

    return results_df


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================

def find_threshold_binary_search(
        data_loader: GBDataLoader,
        weather_year: int,
        learning_rate: float,
        demand_growth: float = 0.01,
        search_range: Tuple[float, float] = (450, 550),
        tolerance: float = 1.0,
        taper_to_zero_by: int = 2029
) -> Dict:
    """
    Use binary search to find the subsidy threshold for a given learning rate.

    The threshold is defined as the minimum subsidy that achieves zero or negative
    gas capacity growth by 2049.

    Args:
        data_loader: GBDataLoader instance
        weather_year: Year for demand/renewable profiles
        learning_rate: Battery learning rate to test (e.g., 0.18 for 18%)
        demand_growth: Annual demand growth rate (default 0.01 = 1%)
        search_range: (min, max) subsidy range to search
        tolerance: Stop when range narrows to this (£/kW)
        taper_to_zero_by: Year when subsidy reaches zero

    Returns:
        Dictionary with threshold and search metadata
    """
    # Temporarily modify the battery learning rate
    original_lr = TECHNOLOGIES["battery"].learning_rate
    TECHNOLOGIES["battery"].learning_rate = learning_rate

    def create_tapering_subsidy(initial: float) -> Dict[int, float]:
        schedule = {}
        for year in PERIODS:
            if year >= taper_to_zero_by:
                schedule[year] = 0
            elif year <= 2027:
                schedule[year] = initial
            else:
                progress = (year - 2027) / (taper_to_zero_by - 2027)
                schedule[year] = initial * (1 - progress)
        return schedule

    def test_subsidy(subsidy: float) -> Tuple[bool, float, float]:
        """Test if a subsidy level achieves zero gas growth. Returns (success, gas_growth, battery_end)."""
        test_policy = Policy(
            name=f"Threshold_Test_{subsidy:.1f}",
            carbon_price={},
            renewable_subsidy={"battery": create_tapering_subsidy(subsidy)}
        )

        # Modify demand growth temporarily
        original_periods_data = None

        # Run with custom demand growth
        data = data_loader.load(weather_year)
        existing = EXISTING_CAPACITY.copy()
        cumulative_global = {tech: TECHNOLOGIES[tech].global_capacity_2024 for tech in TECHNOLOGIES}

        final_gas = 0
        final_battery = 0
        initial_gas = existing["gas_ccgt"] + existing["gas_peaker"]

        for period in PERIODS:
            # Scale demand with custom growth rate
            growth_factor = (1 + demand_growth) ** (period - 2025)
            period_data = {k: v * growth_factor if k == 'demand' else v for k, v in data.items()}

            costs = {tech: get_technology_cost(TECHNOLOGIES[tech], period,
                                               cumulative_global[tech], "endogenous")
                     for tech in TECHNOLOGIES}

            n = build_network(period, period_data, costs, test_policy, existing)
            status = n.optimize(solver_name="highs", solver_options={"output_flag": False})

            if status[0] != "ok":
                continue

            # Update capacities
            new_capacity = {}
            for gen in n.generators.index:
                new_cap = n.generators.p_nom_opt[gen] / 1000 - existing.get(gen, 0)
                new_capacity[gen] = max(0, new_cap)

            bat_cap = n.storage_units.p_nom_opt["battery"] / 1000
            new_capacity["battery"] = max(0, bat_cap - existing.get("battery", 0))

            for tech in new_capacity:
                uk_share = UK_GLOBAL_SHARE.get(tech, 0.02)
                global_addition = new_capacity[tech] / uk_share
                cumulative_global[tech] += global_addition

            for gen in n.generators.index:
                existing[gen] = n.generators.p_nom_opt[gen] / 1000
            existing["battery"] = n.storage_units.p_nom_opt["battery"] / 1000

            final_gas = existing["gas_ccgt"] + existing["gas_peaker"]
            final_battery = existing["battery"]

        gas_growth = final_gas - initial_gas
        return (gas_growth <= 0, gas_growth, final_battery)

    # Binary search
    low, high = search_range
    iterations = 0
    max_iterations = 20

    # First verify that the range brackets the threshold
    low_success, low_gas, _ = test_subsidy(low)
    high_success, high_gas, high_battery = test_subsidy(high)

    if low_success:
        print(f"    Warning: Lower bound £{low}/kW already achieves zero gas growth")
        TECHNOLOGIES["battery"].learning_rate = original_lr
        return {
            "learning_rate": learning_rate,
            "demand_growth": demand_growth,
            "threshold": low,
            "threshold_found": True,
            "note": "threshold_below_range"
        }

    if not high_success:
        print(f"    Warning: Upper bound £{high}/kW does not achieve zero gas growth")
        TECHNOLOGIES["battery"].learning_rate = original_lr
        return {
            "learning_rate": learning_rate,
            "demand_growth": demand_growth,
            "threshold": high,
            "threshold_found": False,
            "note": "threshold_above_range"
        }

    # Binary search for threshold
    while (high - low) > tolerance and iterations < max_iterations:
        mid = (low + high) / 2
        success, gas_growth, battery_end = test_subsidy(mid)

        if success:
            high = mid
        else:
            low = mid

        iterations += 1

    threshold = (low + high) / 2

    # Restore original learning rate
    TECHNOLOGIES["battery"].learning_rate = original_lr

    return {
        "learning_rate": learning_rate,
        "learning_rate_pct": learning_rate * 100,
        "demand_growth": demand_growth,
        "demand_growth_pct": demand_growth * 100,
        "threshold": threshold,
        "threshold_found": True,
        "iterations": iterations,
        "final_range": (low, high),
    }


def run_sensitivity_analysis(
        data_loader: GBDataLoader,
        weather_year: int = 2019,
        learning_rates: Optional[List[float]] = None,
        demand_growths: Optional[List[float]] = None,
        baseline_learning_rate: float = 0.18,
        baseline_demand_growth: float = 0.01,
) -> pd.DataFrame:
    """
    Run sensitivity analysis on battery subsidy threshold.

    Tests how the threshold varies with:
    1. Battery learning rate (default: baseline ±2 percentage points)
    2. Demand growth rate (default: baseline and half)

    Args:
        data_loader: GBDataLoader instance
        weather_year: Year for demand/renewable profiles
        learning_rates: List of learning rates to test (default: [0.16, 0.18, 0.20, 0.22])
        demand_growths: List of demand growth rates (default: [0.005, 0.01])
        baseline_learning_rate: Reference learning rate for comparison
        baseline_demand_growth: Reference demand growth for comparison

    Returns:
        DataFrame with sensitivity analysis results
    """
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS: Battery Subsidy Threshold")
    print("=" * 70)

    if learning_rates is None:
        # Default: baseline ±2 percentage points in 1pp increments
        learning_rates = [
            baseline_learning_rate - 0.02,
            baseline_learning_rate - 0.01,
            baseline_learning_rate,
            baseline_learning_rate + 0.01,
            baseline_learning_rate + 0.02,
        ]

    if demand_growths is None:
        # Default: baseline and half
        demand_growths = [baseline_demand_growth / 2, baseline_demand_growth]

    results = []
    total_runs = len(learning_rates) * len(demand_growths)
    run_num = 0

    # First find baseline threshold
    print(f"\n--- Finding baseline threshold ---")
    print(f"    Learning rate: {baseline_learning_rate * 100:.0f}%, Demand growth: {baseline_demand_growth * 100:.1f}%")

    baseline_result = find_threshold_binary_search(
        data_loader, weather_year,
        learning_rate=baseline_learning_rate,
        demand_growth=baseline_demand_growth,
        search_range=(450, 550)
    )
    baseline_threshold = baseline_result["threshold"]
    print(f"    Baseline threshold: £{baseline_threshold:.1f}/kW")

    baseline_result["is_baseline"] = True
    baseline_result["threshold_shift"] = 0
    baseline_result["threshold_shift_pct"] = 0
    results.append(baseline_result)

    # Now test variations
    for lr in learning_rates:
        for dg in demand_growths:
            # Skip if this is the baseline (already computed)
            if abs(lr - baseline_learning_rate) < 0.001 and abs(dg - baseline_demand_growth) < 0.0001:
                continue

            run_num += 1
            print(f"\n--- Run {run_num}/{total_runs - 1} ---")
            print(f"    Learning rate: {lr * 100:.0f}%, Demand growth: {dg * 100:.1f}%")

            # Adjust search range based on expected direction
            # Higher learning rate = lower threshold (costs fall faster)
            # Lower demand growth = higher threshold (less need for batteries)
            lr_effect = (baseline_learning_rate - lr) * 1500  # rough £/kW per 1% LR
            dg_effect = (dg - baseline_demand_growth) / baseline_demand_growth * baseline_threshold * 0.15
            expected_shift = lr_effect + dg_effect

            search_center = baseline_threshold + expected_shift
            search_range = (max(400, search_center - 80), min(600, search_center + 80))

            result = find_threshold_binary_search(
                data_loader, weather_year,
                learning_rate=lr,
                demand_growth=dg,
                search_range=search_range
            )

            if result["threshold_found"]:
                result["threshold_shift"] = result["threshold"] - baseline_threshold
                result["threshold_shift_pct"] = (result["threshold"] - baseline_threshold) / baseline_threshold * 100
                print(f"    Threshold: £{result['threshold']:.1f}/kW (shift: £{result['threshold_shift']:+.1f}/kW)")
            else:
                result["threshold_shift"] = None
                result["threshold_shift_pct"] = None
                print(f"    Threshold not found in range")

            result["is_baseline"] = False
            results.append(result)

    results_df = pd.DataFrame(results)

    # Summary statistics
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"\nBaseline threshold: £{baseline_threshold:.1f}/kW")
    print(f"  (Learning rate: {baseline_learning_rate * 100:.0f}%, Demand growth: {baseline_demand_growth * 100:.1f}%)")

    # Learning rate sensitivity (holding demand growth at baseline)
    lr_results = results_df[
        (abs(results_df["demand_growth"] - baseline_demand_growth) < 0.0001) &
        (results_df["threshold_found"] == True)
        ].sort_values("learning_rate")

    if len(lr_results) > 1:
        lr_range = lr_results["learning_rate"].max() - lr_results["learning_rate"].min()
        threshold_range = lr_results["threshold"].max() - lr_results["threshold"].min()
        sensitivity_per_pp = threshold_range / (lr_range * 100) if lr_range > 0 else 0

        print(f"\nLearning rate sensitivity:")
        print(
            f"  Range tested: {lr_results['learning_rate'].min() * 100:.0f}% - {lr_results['learning_rate'].max() * 100:.0f}%")
        print(f"  Threshold range: £{lr_results['threshold'].min():.1f} - £{lr_results['threshold'].max():.1f}/kW")
        print(f"  Sensitivity: ~£{abs(sensitivity_per_pp):.0f}/kW per percentage point")

    # Demand growth sensitivity (holding learning rate at baseline)
    dg_results = results_df[
        (abs(results_df["learning_rate"] - baseline_learning_rate) < 0.001) &
        (results_df["threshold_found"] == True)
        ].sort_values("demand_growth")

    if len(dg_results) > 1:
        baseline_dg_threshold = dg_results[dg_results["demand_growth"] == baseline_demand_growth]["threshold"].values
        half_dg_threshold = dg_results[dg_results["demand_growth"] == baseline_demand_growth / 2]["threshold"].values

        if len(baseline_dg_threshold) > 0 and len(half_dg_threshold) > 0:
            pct_shift = (half_dg_threshold[0] - baseline_dg_threshold[0]) / baseline_dg_threshold[0] * 100
            print(f"\nDemand growth sensitivity:")
            print(f"  Halving demand growth shifts threshold by {pct_shift:+.1f}%")

    # Save results
    output_file = "sensitivity_analysis_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Generate robustness statement
    print("\n" + "-" * 70)
    print("SUGGESTED ROBUSTNESS STATEMENT FOR PAPER:")
    print("-" * 70)

    if len(lr_results) > 1:
        lr_min = lr_results["learning_rate"].min() * 100
        lr_max = lr_results["learning_rate"].max() * 100
        t_min = lr_results["threshold"].min()
        t_max = lr_results["threshold"].max()
        t_range = t_max - t_min

        print(f"\n\"Varying the battery learning rate within empirically plausible bounds")
        print(f"({lr_min:.0f}%–{lr_max:.0f}%) shifts the threshold by approximately ±£{t_range / 2:.0f}/kW")
        print(f"while preserving its existence.\"")

    if len(dg_results) > 1 and len(baseline_dg_threshold) > 0 and len(half_dg_threshold) > 0:
        print(f"\n\"Halving demand growth shifts the threshold upward by approximately {abs(pct_shift):.0f}%.\"")

    return results_df


# ============================================================================
# MAIN
# ============================================================================

def main(pypsa_gb_path: str = None, weather_year: int = None):
    """Run complete GB electricity model."""
    
    # Use config defaults if not specified
    if pypsa_gb_path is None:
        pypsa_gb_path = PYPSA_GB_PATH
    if weather_year is None:
        weather_year = WEATHER_YEAR
    
    print("="*70)
    print("GB ELECTRICITY MODEL")
    print("Exogenous vs Endogenous Technology Learning")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Periods: {PERIODS[0]} to {PERIODS[-1]} (every {PERIODS[1]-PERIODS[0]} years)")
    print(f"  Weather year: {weather_year}")
    print(f"  Data source: {'PyPSA-GB' if pypsa_gb_path else 'Synthetic'}")
    
    # Initialize data loader
    data_loader = GBDataLoader(pypsa_gb_path)
    
    # Run all scenarios
    print("\nRunning scenarios...")
    results = run_all_scenarios(data_loader, weather_year)
    
    # Analyze
    print("\nAnalyzing results...")
    summary = analyze_results(results)
    
    # Print key results
    print("\n" + "="*70)
    print(f"KEY RESULTS - {PERIODS[-1]} Capacity (GW)")
    print("="*70)
    
    last_period = PERIODS[-1]
    for policy in ["Current Policy", "High Carbon Price", "Offshore Focus"]:
        print(f"\n{policy}:")
        for lt in ["exogenous", "endogenous"]:
            row = results[(results["period"] == last_period) & (results["learning_type"] == lt) & (results["policy"] == policy)]
            if len(row) > 0:
                row = row.iloc[0]
                print(f"  {lt:12s}: Offshore={row['offshore_gw']:.1f}, Solar={row['solar_gw']:.1f}, "
                      f"Gas={row['gas_gw']:.1f}, Battery={row['battery_gw']:.1f}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_results(results, summary)
    
    # Save data
    results.to_csv("gb_model_results.csv", index=False)
    summary.to_csv("gb_model_summary.csv", index=False)
    print("\nSaved: gb_model_results.csv, gb_model_summary.csv")
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    
    return results, summary


if __name__ == "__main__":
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    RUN_MODE = "sensitivity"  # Options: "normal", "battery_sweep", "solar_battery_sweep"

    # For battery-only sweep
    BATTERY_SWEEP_RANGE = (490, 540, 1)

    # For solar + battery sweep
    BATTERY_RANGE = (485, 495, 1)  # Coarser for speed
    SOLAR_RANGE = (85, 105, 1)

    TAPER_END_YEAR = 2031
    # =========================================================================

    if RUN_MODE == "solar_battery_sweep":
        print("\n" + "=" * 70)
        print("RUNNING SOLAR + BATTERY SUBSIDY GRID SEARCH")
        print("=" * 70)

        data_loader = GBDataLoader(PYPSA_GB_PATH)
        results = find_minimum_solar_battery_subsidy(
            data_loader,
            weather_year=WEATHER_YEAR,
            battery_range=BATTERY_RANGE,
            solar_range=SOLAR_RANGE,
            learning_type="endoogenous",
            taper_to_zero_by=TAPER_END_YEAR
        )

    elif RUN_MODE == "battery_sweep":
        data_loader = GBDataLoader(PYPSA_GB_PATH)
        results = find_minimum_battery_subsidy(
            data_loader,
            weather_year=WEATHER_YEAR,
            subsidy_range=BATTERY_SWEEP_RANGE,
            learning_type="endogenous",
            taper_to_zero_by=TAPER_END_YEAR
        )
    elif RUN_MODE == "carbon_sweep":

        CARBON_RANGE = (500, 1000, 25)  # Test £0 to £500/tCO2 in £25 steps

        data_loader = GBDataLoader(PYPSA_GB_PATH)
        results = find_minimum_carbon_tax(
            data_loader,
            weather_year=WEATHER_YEAR,
            price_range=CARBON_RANGE,
            learning_type="endogenous"  # or "exogenous" to compare!
        )
    elif RUN_MODE == "sensitivity":
        # Run sensitivity analysis for robustness claims
        # Note: The paper cites 18% battery learning rate, but the model default is 25%
        # Adjust baseline_learning_rate to match your paper's assumptions
        data_loader = GBDataLoader(PYPSA_GB_PATH)
        run_sensitivity_analysis(
            data_loader,
            weather_year=WEATHER_YEAR,
            baseline_learning_rate=0.25,  # As cited in paper
            baseline_demand_growth=0.01,  # 1% annual growth
            learning_rates=[0.23, 0.24, 0.25, 0.26, 0.27],  # ±2pp
            demand_growths=[0.005, 0.01],  # Half and baseline
        )

    else:
        results, summary = main()
