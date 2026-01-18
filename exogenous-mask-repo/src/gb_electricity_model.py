#!/usr/bin/env python3
"""
GB Electricity Model: Exogenous vs Endogenous Technology Learning

A capacity expansion model of the Great Britain electricity system that compares
exogenous and endogenous technology learning formulations. This model accompanies
the paper "The Exogenous Mask: How Cost Assumptions Hide Sensitive Intervention 
Points in Energy Transition Policy".

Key features:
- Myopic capacity expansion over 2025-2049 (2-year intervals)
- Single-node (copper-plate) representation of GB
- Wright's Law learning curves with empirically calibrated rates
- Configurable policy instruments: carbon price, technology subsidies

Usage:
    python gb_electricity_model.py --mode normal
    python gb_electricity_model.py --mode battery_sweep
    python gb_electricity_model.py --mode solar_battery_sweep

Data sources:
    - PyPSA-GB (Lyden et al., 2022) for demand and generation profiles
    - Falls back to calibrated synthetic data if PyPSA-GB unavailable

Author: Nadav Mantel
License: MIT
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


# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to PyPSA-GB repository (set to None to use synthetic data)
PYPSA_GB_PATH: Optional[str] = None  # e.g., "./PyPSA-GB"

# Model time settings
PERIODS = list(range(2025, 2051, 2))  # Every 2 years
HOURS_PER_YEAR = 8760
DISCOUNT_RATE = 0.07
WEATHER_YEAR = 2019

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

# Historical installed capacities (MW) for capacity factor calculation
INSTALLED_CAPACITY_2019 = {
    'offshore_wind': 9700,
    'onshore_wind': 13500,
    'solar': 13000,
}

# UK share of global market (for endogenous learning)
# UK deployment contributes to global deployment via: global = uk / share
UK_GLOBAL_SHARE = {
    "solar": 0.02,          # UK ~2% of global solar
    "onshore_wind": 0.02,   # UK ~2% of global onshore wind
    "offshore_wind": 0.05,  # UK ~5% of global offshore (major player)
    "battery": 0.02,        # UK ~2% of global battery storage
    "gas_ccgt": 0.02,       # Mature tech, share doesn't matter (LR=0)
    "gas_peaker": 0.02,
    "nuclear": 0.01,        # UK ~1% of global nuclear
}

# Maximum build capacity per period (GW) - set high to not constrain
MAX_BUILD_CAPACITY = {tech: 1000 for tech in EXISTING_CAPACITY}


# =============================================================================
# TECHNOLOGY DEFINITIONS
# =============================================================================

@dataclass
class Technology:
    """Technology parameters for capacity expansion."""
    name: str
    capital_cost: float       # £/kW (2024 baseline)
    marginal_cost: float      # £/MWh
    learning_rate: float      # Per doubling of cumulative capacity
    global_capacity_2024: float  # GW
    max_build_per_period: float  # GW per 2-year period
    capacity_factor: float    # For dispatchable; max for variable
    co2_intensity: float      # tCO2/MWh
    lifetime: int             # Years
    is_variable: bool         # True for solar/wind


TECHNOLOGIES = {
    "gas_ccgt": Technology(
        name="gas_ccgt", capital_cost=750, marginal_cost=50,
        learning_rate=0.0, global_capacity_2024=2000,
        max_build_per_period=MAX_BUILD_CAPACITY["gas_ccgt"],
        capacity_factor=0.85, co2_intensity=0.34, lifetime=25, is_variable=False
    ),
    "gas_peaker": Technology(
        name="gas_peaker", capital_cost=500, marginal_cost=80,
        learning_rate=0.0, global_capacity_2024=500,
        max_build_per_period=MAX_BUILD_CAPACITY["gas_peaker"],
        capacity_factor=0.10, co2_intensity=0.45, lifetime=25, is_variable=False
    ),
    "nuclear": Technology(
        name="nuclear", capital_cost=8000, marginal_cost=10,
        learning_rate=0.0, global_capacity_2024=400,
        max_build_per_period=MAX_BUILD_CAPACITY["nuclear"],
        capacity_factor=0.85, co2_intensity=0.0, lifetime=60, is_variable=False
    ),
    "solar": Technology(
        name="solar", capital_cost=450, marginal_cost=0,
        learning_rate=0.20, global_capacity_2024=1800,
        max_build_per_period=MAX_BUILD_CAPACITY["solar"],
        capacity_factor=0.11, co2_intensity=0.0, lifetime=25, is_variable=True
    ),
    "onshore_wind": Technology(
        name="onshore_wind", capital_cost=950, marginal_cost=0,
        learning_rate=0.08, global_capacity_2024=1000,
        max_build_per_period=MAX_BUILD_CAPACITY["onshore_wind"],
        capacity_factor=0.26, co2_intensity=0.0, lifetime=25, is_variable=True
    ),
    "offshore_wind": Technology(
        name="offshore_wind", capital_cost=1800, marginal_cost=0,
        learning_rate=0.12, global_capacity_2024=75,
        max_build_per_period=MAX_BUILD_CAPACITY["offshore_wind"],
        capacity_factor=0.40, co2_intensity=0.0, lifetime=25, is_variable=True
    ),
    "battery": Technology(
        name="battery", capital_cost=600, marginal_cost=3,
        learning_rate=0.18, global_capacity_2024=300,
        max_build_per_period=MAX_BUILD_CAPACITY["battery"],
        capacity_factor=1.0, co2_intensity=0.0, lifetime=15, is_variable=False
    ),
}


# =============================================================================
# POLICY DEFINITIONS
# =============================================================================

@dataclass
class Policy:
    """Policy scenario definition."""
    name: str
    carbon_price: Dict[int, float]           # £/tCO2 by year
    renewable_subsidy: Dict[str, Dict[int, float]]  # £/kW by tech and year


POLICIES = {
    "baseline": Policy(
        name="Current Policy",
        carbon_price={
            2025: 50, 2027: 62, 2029: 74, 2031: 86, 2033: 98, 2035: 110,
            2037: 122, 2039: 134, 2041: 146, 2043: 158, 2045: 170, 
            2047: 182, 2049: 194
        },
        renewable_subsidy={
            "offshore_wind": {
                2025: 100, 2027: 92, 2029: 84, 2031: 76, 2033: 68, 2035: 60,
                2037: 52, 2039: 44, 2041: 36, 2043: 28, 2045: 20, 2047: 12, 2049: 4
            },
            "onshore_wind": {
                2025: 50, 2027: 46, 2029: 42, 2031: 38, 2033: 34, 2035: 30,
                2037: 26, 2039: 22, 2041: 18, 2043: 14, 2045: 10, 2047: 6, 2049: 2
            },
            "solar": {
                2025: 50, 2027: 46, 2029: 42, 2031: 38, 2033: 34, 2035: 30,
                2037: 26, 2039: 22, 2041: 18, 2043: 14, 2045: 10, 2047: 6, 2049: 2
            },
        }
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
                2025: 105, 2027: 105, 2029: 52, 2031: 0, 2033: 0, 2035: 0,
                2037: 0, 2039: 0, 2041: 0, 2043: 0, 2045: 0, 2047: 0, 2049: 0
            },
        }
    ),
}


# =============================================================================
# DATA LOADING
# =============================================================================

class GBDataLoader:
    """Load GB electricity data from PyPSA-GB or generate synthetic profiles."""
    
    def __init__(self, pypsa_gb_path: Optional[str] = None):
        self.pypsa_gb_path = Path(pypsa_gb_path) if pypsa_gb_path else None
        self.data = None
    
    def load(self, year: int = 2019) -> Dict[str, np.ndarray]:
        """Load all required data profiles."""
        print(f"\nLoading GB data for weather year {year}...")
        
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
    
    def _load_from_pypsa_gb(self, year: int) -> Optional[Dict[str, np.ndarray]]:
        """Load from PyPSA-GB repository structure."""
        data = {}
        
        # ESPENI demand
        espeni_path = self.pypsa_gb_path / "data" / "demand" / "espeni.csv"
        if espeni_path.exists():
            print(f"  Loading ESPENI demand from {espeni_path}")
            try:
                df = pd.read_csv(espeni_path, index_col=0, low_memory=False)
                df.index = pd.to_datetime(df.index, errors='coerce')
                
                # Find demand column
                demand_cols = [c for c in df.columns if 'demand' in c.lower() or 'mw' in c.lower()]
                demand_col = demand_cols[0] if demand_cols else df.select_dtypes(np.number).columns[0]
                
                year_data = df[df.index.year == year][demand_col].dropna().values.astype(float)
                
                # Resample if half-hourly
                if len(year_data) > HOURS_PER_YEAR * 1.5:
                    year_data = year_data[:len(year_data)//2*2].reshape(-1, 2).mean(axis=1)
                
                year_data = year_data[:HOURS_PER_YEAR]
                if len(year_data) < HOURS_PER_YEAR:
                    year_data = np.tile(year_data, HOURS_PER_YEAR // len(year_data) + 1)[:HOURS_PER_YEAR]
                
                # Convert to GW if in MW
                if np.mean(year_data) > 1000:
                    year_data = year_data / 1000
                
                data['demand'] = year_data
                print(f"    Mean: {data['demand'].mean():.1f} GW, Peak: {data['demand'].max():.1f} GW")
            except Exception as e:
                print(f"    Error loading ESPENI: {e}")
        
        # Renewable profiles
        atlite_path = self.pypsa_gb_path / "data" / "renewables" / "atlite" / "outputs"
        for tech, patterns in [('solar', ['PV', 'Solar']), 
                               ('onshore_wind', ['Wind_Onshore', 'Onshore']),
                               ('offshore_wind', ['Wind_Offshore', 'Offshore'])]:
            for pattern in patterns:
                tech_path = atlite_path / pattern
                if tech_path.exists():
                    csv_files = list(tech_path.glob(f"*{year}*.csv")) or list(tech_path.glob("*.csv"))
                    if csv_files:
                        profile = self._load_profile_csv(csv_files[0], year)
                        if profile is not None:
                            data[tech] = profile
                            print(f"  {tech}: CF = {profile.mean()*100:.1f}%")
                            break
        
        required = ['demand', 'solar', 'onshore_wind', 'offshore_wind']
        if all(k in data for k in required):
            data['source'] = 'PyPSA-GB (ESPENI + Atlite/ERA5)'
            return data
        return None
    
    def _load_profile_csv(self, filepath: Path, year: int) -> Optional[np.ndarray]:
        """Load a capacity factor profile from CSV."""
        df = pd.read_csv(filepath, index_col=0, low_memory=False)
        
        # Check for CF column or sum generators
        if 'capacity_factor' in df.columns:
            profile = df['capacity_factor'].values.astype(float)
        else:
            numeric_cols = df.select_dtypes(np.number).columns
            if len(numeric_cols) > 3:
                total = df[numeric_cols].sum(axis=1).fillna(0).values.astype(float)
                profile = total / np.nanmax(total) if np.nanmax(total) > 0 else total
            else:
                profile = df[numeric_cols[0]].values.astype(float)
                if np.nanmax(profile) > 1:
                    profile = profile / np.nanmax(profile)
        
        # Ensure 8760 values
        if len(profile) > HOURS_PER_YEAR * 1.5:
            profile = profile[:len(profile)//2*2].reshape(-1, 2).mean(axis=1)
        if len(profile) < HOURS_PER_YEAR:
            profile = np.tile(profile, HOURS_PER_YEAR // len(profile) + 1)[:HOURS_PER_YEAR]
        else:
            profile = profile[:HOURS_PER_YEAR]
        
        return np.clip(profile, 0, 1)
    
    def _generate_synthetic(self, year: int, seed: int = 42) -> Dict[str, np.ndarray]:
        """Generate synthetic GB profiles calibrated to published statistics."""
        np.random.seed(seed + year)
        hours = HOURS_PER_YEAR
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
        
        # SOLAR - Target 10.5% CF (DUKES)
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
        solar = np.clip(solar * (0.3 + 0.7 * cloud), 0, 1)
        solar = solar * (0.105 / max(0.01, solar.mean()))
        
        # WIND - Weather-driven with autocorrelation
        n_days = hours // 24 + 1
        daily_weather = np.random.weibull(2.0, n_days)
        for i in range(1, n_days):
            daily_weather[i] = 0.65 * daily_weather[i-1] + 0.35 * daily_weather[i]
        weather = np.interp(t, np.arange(n_days) * 24, daily_weather)
        weather = (weather - weather.min()) / (weather.max() - weather.min() + 0.01)
        
        seasonal_on = 0.28 + 0.10 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
        seasonal_on = np.repeat(seasonal_on[:365], 24)[:hours]
        onshore = np.clip(seasonal_on * (0.25 + 0.75 * weather) + 0.05 * np.random.randn(hours), 0.02, 0.95)
        onshore = onshore * (0.265 / onshore.mean())
        
        seasonal_off = 0.42 + 0.06 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
        seasonal_off = np.repeat(seasonal_off[:365], 24)[:hours]
        offshore = np.clip(seasonal_off * (0.35 + 0.65 * weather) + 0.03 * np.random.randn(hours), 0.05, 0.98)
        offshore = offshore * (0.40 / offshore.mean())
        
        print(f"    Demand: {demand.mean():.1f} GW mean, {demand.max():.1f} GW peak")
        print(f"    Solar CF: {solar.mean()*100:.1f}%, Onshore: {onshore.mean()*100:.1f}%, Offshore: {offshore.mean()*100:.1f}%")
        
        return {
            'demand': demand,
            'solar': np.clip(solar, 0, 1),
            'onshore_wind': np.clip(onshore, 0, 1),
            'offshore_wind': np.clip(offshore, 0, 1),
            'source': 'Synthetic (calibrated to DUKES/ESPENI)'
        }


# =============================================================================
# LEARNING CURVES
# =============================================================================

def annuity_factor(lifetime: int, rate: float) -> float:
    """Calculate annuity factor for capital cost annualization."""
    if rate == 0:
        return 1 / lifetime
    return rate / (1 - (1 + rate) ** (-lifetime))


def wright_law_cost(C0: float, K: float, K0: float, learning_rate: float) -> float:
    """
    Calculate cost using Wright's Law: C = C0 * (K/K0)^(-alpha)
    
    Args:
        C0: Initial cost
        K: Current cumulative capacity
        K0: Initial cumulative capacity
        learning_rate: Cost reduction per doubling (e.g., 0.2 = 20%)
    
    Returns:
        Current cost after learning
    """
    if learning_rate <= 0 or K <= K0:
        return C0
    alpha = -np.log2(1 - learning_rate)
    return C0 * (K / K0) ** (-alpha)


def get_technology_cost(tech: Technology, year: int, cumulative_global: float,
                        learning_type: str) -> float:
    """
    Get technology cost based on learning formulation.
    
    Args:
        tech: Technology object
        year: Current year
        cumulative_global: Cumulative global capacity (for endogenous)
        learning_type: 'exogenous' or 'endogenous'
    
    Returns:
        Capital cost in £/kW
    """
    if learning_type == "exogenous":
        # Costs follow predetermined global trajectory (IEA/IRENA projections)
        global_growth_rates = {
            "solar": 400, "onshore_wind": 100, "offshore_wind": 30,
            "battery": 200, "gas_ccgt": 20, "gas_peaker": 5, "nuclear": 5,
        }
        years_from_start = year - 2025
        growth = global_growth_rates.get(tech.name, 10)
        projected_global = tech.global_capacity_2024 + growth * years_from_start
        return wright_law_cost(tech.capital_cost, projected_global, 
                               tech.global_capacity_2024, tech.learning_rate)
    else:
        # Costs depend on actual cumulative deployment
        return wright_law_cost(tech.capital_cost, cumulative_global,
                               tech.global_capacity_2024, tech.learning_rate)


def get_carbon_price(policy: Policy, year: int) -> float:
    """Get carbon price for a given year, interpolating if needed."""
    if not policy.carbon_price:
        return 0.0
    if year in policy.carbon_price:
        return policy.carbon_price[year]
    
    # Interpolate
    years = sorted(policy.carbon_price.keys())
    if year < years[0]:
        return policy.carbon_price[years[0]]
    if year > years[-1]:
        return policy.carbon_price[years[-1]]
    
    for i in range(len(years) - 1):
        if years[i] <= year < years[i+1]:
            t = (year - years[i]) / (years[i+1] - years[i])
            return policy.carbon_price[years[i]] * (1-t) + policy.carbon_price[years[i+1]] * t
    return 0.0


# =============================================================================
# PyPSA NETWORK BUILDER
# =============================================================================

def build_network(
    year: int,
    data: Dict[str, np.ndarray],
    costs: Dict[str, float],
    policy: Policy,
    existing: Dict[str, float],
    sampling: int = 6
) -> pypsa.Network:
    """
    Build PyPSA network for capacity expansion optimization.
    
    Args:
        year: Planning year
        data: Dictionary with demand and renewable profiles
        costs: Dictionary of technology costs
        policy: Policy object with carbon price and subsidies
        existing: Dictionary of existing capacities
        sampling: Hour sampling factor (6 = every 6th hour)
    
    Returns:
        Configured PyPSA Network
    """
    n = pypsa.Network()
    
    # Time setup
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
    
    carbon_price = get_carbon_price(policy, year)
    
    # Dispatchable generators
    for tech_name in ["gas_ccgt", "gas_peaker", "nuclear"]:
        tech = TECHNOLOGIES[tech_name]
        existing_cap = existing.get(tech_name, 0)
        max_cap = existing_cap + tech.max_build_per_period
        
        mc = tech.marginal_cost
        if "gas" in tech_name:
            mc += carbon_price * tech.co2_intensity
        
        n.add("Generator", tech_name,
            bus="GB", carrier=tech_name,
            p_nom_extendable=True,
            p_nom_min=existing_cap * 1000,
            p_nom_max=max_cap * 1000,
            capital_cost=costs[tech_name] * annuity_factor(tech.lifetime, DISCOUNT_RATE) * 1000,
            marginal_cost=mc,
            p_max_pu=tech.capacity_factor if tech_name == "nuclear" else 1.0,
        )
    
    # Variable renewables
    for tech_name, profile_key in [("solar", "solar"), 
                                    ("onshore_wind", "onshore_wind"), 
                                    ("offshore_wind", "offshore_wind")]:
        tech = TECHNOLOGIES[tech_name]
        existing_cap = existing.get(tech_name, 0)
        max_cap = existing_cap + tech.max_build_per_period
        
        subsidy = policy.renewable_subsidy.get(tech_name, {}).get(year, 0)
        effective_cost = max(0, costs[tech_name] - subsidy)
        
        profile = data[profile_key][::sampling][:sampled_hours]
        
        n.add("Generator", tech_name,
            bus="GB", carrier=tech_name,
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
    
    n.add("StorageUnit", "battery",
        bus="GB", carrier="battery",
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


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

def run_scenario(
    learning_type: str,
    policy: Policy,
    data_loader: GBDataLoader,
    weather_year: int = 2019,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Run myopic capacity expansion for one scenario.
    
    Args:
        learning_type: 'exogenous' or 'endogenous'
        policy: Policy object
        data_loader: GBDataLoader instance
        weather_year: Year for demand/renewable profiles
        verbose: Print detailed progress
    
    Returns:
        DataFrame with results for each period
    """
    data = data_loader.load(weather_year)
    
    existing = EXISTING_CAPACITY.copy()
    cumulative_global = {tech: TECHNOLOGIES[tech].global_capacity_2024 for tech in TECHNOLOGIES}
    
    results = []
    
    for period in PERIODS:
        # Scale demand for growth (1% per year)
        growth_factor = 1.01 ** (period - 2025)
        period_data = {k: v * growth_factor if k == 'demand' else v for k, v in data.items()}
        
        # Get costs
        costs = {tech: get_technology_cost(TECHNOLOGIES[tech], period, 
                                           cumulative_global[tech], learning_type) 
                 for tech in TECHNOLOGIES}
        
        if verbose and period in [2025, 2035, 2049]:
            print(f"  {learning_type} costs in {period}: Solar £{costs['solar']:.0f}, "
                  f"Battery £{costs['battery']:.0f}, Offshore £{costs['offshore_wind']:.0f}")
        
        # Build and optimize
        n = build_network(period, period_data, costs, policy, existing)
        status = n.optimize(solver_name="highs", solver_options={"output_flag": False})
        
        if status[0] != "ok":
            print(f"  Warning: Optimization failed for {period}")
            continue
        
        # Extract results
        new_capacity = {}
        for gen in n.generators.index:
            new_cap = n.generators.p_nom_opt[gen] / 1000 - existing.get(gen, 0)
            new_capacity[gen] = max(0, new_cap)
        
        bat_cap = n.storage_units.p_nom_opt["battery"] / 1000
        new_capacity["battery"] = max(0, bat_cap - existing.get("battery", 0))
        
        # Update cumulative global (endogenous only matters here)
        for tech in new_capacity:
            uk_share = UK_GLOBAL_SHARE.get(tech, 0.02)
            global_addition = new_capacity[tech] / uk_share
            cumulative_global[tech] += global_addition
        
        # Update existing for next period
        for gen in n.generators.index:
            existing[gen] = n.generators.p_nom_opt[gen] / 1000
        existing["battery"] = n.storage_units.p_nom_opt["battery"] / 1000
        
        # Calculate emissions and costs
        gen_p = n.generators_t.p.sum() * n.snapshot_weightings.generators.iloc[0]
        emissions = sum(gen_p[g] * TECHNOLOGIES[g].co2_intensity / 1000 
                       for g in n.generators.index if "gas" in g)
        system_cost = n.objective / 1e9
        
        results.append({
            "period": period,
            "learning_type": learning_type,
            "policy": policy.name,
            "solar_gw": existing["solar"],
            "onshore_gw": existing["onshore_wind"],
            "offshore_gw": existing["offshore_wind"],
            "gas_gw": existing["gas_ccgt"] + existing["gas_peaker"],
            "battery_gw": existing["battery"],
            "nuclear_gw": existing["nuclear"],
            "emissions_mt": emissions,
            "system_cost_bn": system_cost,
            "solar_cost": costs["solar"],
            "battery_cost": costs["battery"],
        })
    
    return pd.DataFrame(results)


def run_all_scenarios(data_loader: GBDataLoader, weather_year: int = 2019) -> pd.DataFrame:
    """Run exogenous and endogenous scenarios for all policies."""
    all_results = []
    
    for learning_type in ["exogenous", "endogenous"]:
        for policy_key, policy in POLICIES.items():
            print(f"\nRunning {learning_type} / {policy.name}...")
            df = run_scenario(learning_type, policy, data_loader, weather_year, verbose=True)
            all_results.append(df)
    
    return pd.concat(all_results, ignore_index=True)


# =============================================================================
# SUBSIDY SWEEPS
# =============================================================================

def find_minimum_battery_subsidy(
    data_loader: GBDataLoader,
    weather_year: int = 2019,
    subsidy_range: tuple = (480, 520, 1),
    learning_type: str = "endogenous",
    taper_to_zero_by: int = 2031
) -> pd.DataFrame:
    """
    Sweep battery subsidies to find threshold for zero gas growth.
    
    Args:
        data_loader: GBDataLoader instance
        weather_year: Year for demand/renewable profiles
        subsidy_range: (start, stop, step) for initial subsidy in £/kW
        learning_type: 'exogenous' or 'endogenous'
        taper_to_zero_by: Year when subsidy reaches zero
    
    Returns:
        DataFrame with sweep results
    """
    print("\n" + "=" * 70)
    print("BATTERY SUBSIDY SWEEP")
    print(f"Learning type: {learning_type}, Taper to zero by: {taper_to_zero_by}")
    print("=" * 70)
    
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
    
    start, stop, step = subsidy_range
    results = []
    
    for initial_subsidy in range(start, stop + 1, step):
        print(f"\n  Testing £{initial_subsidy}/kW...")
        
        battery_schedule = create_tapering_subsidy(initial_subsidy)
        
        test_policy = Policy(
            name=f"Battery_Subsidy_{initial_subsidy}",
            carbon_price={},
            renewable_subsidy={"battery": battery_schedule}
        )
        
        df = run_scenario(learning_type, test_policy, data_loader, weather_year)
        
        gas_growth = df.iloc[-1]['gas_gw'] - df.iloc[0]['gas_gw']
        bat_growth = df.iloc[-1]['battery_gw'] - df.iloc[0]['battery_gw']
        
        # Calculate subsidy expenditure
        total_subsidy_bn = 0
        prev_bat = df.iloc[0]['battery_gw']
        for _, row in df.iterrows():
            new_bat = max(0, row['battery_gw'] - prev_bat)
            total_subsidy_bn += battery_schedule.get(row['period'], 0) * new_bat / 1000
            prev_bat = row['battery_gw']
        
        results.append({
            'initial_subsidy_per_kw': initial_subsidy,
            'gas_growth_gw': gas_growth,
            'battery_growth_gw': bat_growth,
            'gas_end_gw': df.iloc[-1]['gas_gw'],
            'battery_end_gw': df.iloc[-1]['battery_gw'],
            'total_emissions_mt': df['emissions_mt'].sum(),
            'total_system_cost_bn': df['system_cost_bn'].sum(),
            'total_subsidy_cost_bn': total_subsidy_bn,
        })
        
        print(f"    Gas growth: {gas_growth:+.1f} GW, Battery: {bat_growth:+.1f} GW")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'battery_subsidy_sweep_{learning_type}.csv', index=False)
    print(f"\nSaved: battery_subsidy_sweep_{learning_type}.csv")
    
    return results_df


def find_minimum_solar_battery_subsidy(
    data_loader: GBDataLoader,
    weather_year: int = 2019,
    battery_range: tuple = (450, 500, 5),
    solar_range: tuple = (100, 120, 5),
    learning_type: str = "endogenous",
    taper_to_zero_by: int = 2031
) -> pd.DataFrame:
    """
    Grid search over solar and battery subsidies.
    
    Args:
        data_loader: GBDataLoader instance
        weather_year: Year for demand/renewable profiles
        battery_range: (start, stop, step) for battery subsidy
        solar_range: (start, stop, step) for solar subsidy
        learning_type: 'exogenous' or 'endogenous'
        taper_to_zero_by: Year when subsidies reach zero
    
    Returns:
        DataFrame with grid search results
    """
    print("\n" + "=" * 70)
    print("SOLAR + BATTERY SUBSIDY GRID SEARCH")
    print("=" * 70)
    
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
    
    bat_start, bat_stop, bat_step = battery_range
    sol_start, sol_stop, sol_step = solar_range
    
    battery_levels = list(range(bat_start, bat_stop + 1, bat_step))
    solar_levels = list(range(sol_start, sol_stop + 1, sol_step))
    
    total_runs = len(battery_levels) * len(solar_levels)
    print(f"Testing {len(battery_levels)} × {len(solar_levels)} = {total_runs} combinations")
    
    results = []
    run_num = 0
    
    for bat_sub in battery_levels:
        for sol_sub in solar_levels:
            run_num += 1
            print(f"\n[{run_num}/{total_runs}] Battery: £{bat_sub}/kW, Solar: £{sol_sub}/kW")
            
            test_policy = Policy(
                name=f"BatSol_{bat_sub}_{sol_sub}",
                carbon_price={},
                renewable_subsidy={
                    "battery": create_tapering_subsidy(bat_sub),
                    "solar": create_tapering_subsidy(sol_sub),
                }
            )
            
            df = run_scenario(learning_type, test_policy, data_loader, weather_year)
            
            gas_growth = df.iloc[-1]['gas_gw'] - df.iloc[0]['gas_gw']
            bat_growth = df.iloc[-1]['battery_gw'] - df.iloc[0]['battery_gw']
            sol_growth = df.iloc[-1]['solar_gw'] - df.iloc[0]['solar_gw']
            
            # Calculate subsidy expenditure
            total_subsidy_bn = 0
            prev_bat, prev_sol = df.iloc[0]['battery_gw'], df.iloc[0]['solar_gw']
            bat_schedule = create_tapering_subsidy(bat_sub)
            sol_schedule = create_tapering_subsidy(sol_sub)
            
            for _, row in df.iterrows():
                new_bat = max(0, row['battery_gw'] - prev_bat)
                new_sol = max(0, row['solar_gw'] - prev_sol)
                total_subsidy_bn += bat_schedule.get(row['period'], 0) * new_bat / 1000
                total_subsidy_bn += sol_schedule.get(row['period'], 0) * new_sol / 1000
                prev_bat, prev_sol = row['battery_gw'], row['solar_gw']
            
            results.append({
                'battery_initial': bat_sub,
                'solar_initial': sol_sub,
                'gas_growth_gw': gas_growth,
                'battery_growth_gw': bat_growth,
                'solar_growth_gw': sol_growth,
                'gas_end_gw': df.iloc[-1]['gas_gw'],
                'battery_end_gw': df.iloc[-1]['battery_gw'],
                'solar_end_gw': df.iloc[-1]['solar_gw'],
                'total_emissions_mt': df['emissions_mt'].sum(),
                'total_system_cost_bn': df['system_cost_bn'].sum(),
                'total_subsidy_cost_bn': total_subsidy_bn,
            })
            
            status = "✓" if gas_growth <= 0 else "✗"
            print(f"    Gas: {gas_growth:+.1f} GW {status}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'solar_battery_subsidy_grid_{learning_type}.csv', index=False)
    print(f"\nSaved: solar_battery_subsidy_grid_{learning_type}.csv")
    
    return results_df


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description="GB Electricity Model: Exogenous vs Endogenous Learning"
    )
    parser.add_argument(
        "--mode", 
        choices=["normal", "battery_sweep", "solar_battery_sweep"],
        default="normal",
        help="Run mode (default: normal)"
    )
    parser.add_argument(
        "--pypsa-gb-path",
        type=str,
        default=None,
        help="Path to PyPSA-GB repository (optional)"
    )
    parser.add_argument(
        "--weather-year",
        type=int,
        default=2019,
        help="Weather year for profiles (default: 2019)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("GB ELECTRICITY MODEL")
    print("Exogenous vs Endogenous Technology Learning")
    print("=" * 70)
    
    data_loader = GBDataLoader(args.pypsa_gb_path or PYPSA_GB_PATH)
    
    if args.mode == "normal":
        results = run_all_scenarios(data_loader, args.weather_year)
        results.to_csv("gb_model_results.csv", index=False)
        print("\nSaved: gb_model_results.csv")
        
    elif args.mode == "battery_sweep":
        find_minimum_battery_subsidy(
            data_loader,
            weather_year=args.weather_year,
            subsidy_range=(480, 520, 1),
            learning_type="endogenous",
            taper_to_zero_by=2031
        )
        
    elif args.mode == "solar_battery_sweep":
        find_minimum_solar_battery_subsidy(
            data_loader,
            weather_year=args.weather_year,
            battery_range=(450, 500, 5),
            solar_range=(100, 120, 5),
            learning_type="endogenous",
            taper_to_zero_by=2031
        )
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
