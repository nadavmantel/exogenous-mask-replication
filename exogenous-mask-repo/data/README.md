# Data Sources

This model can operate with either **synthetic data** (default) or **real data** from PyPSA-GB.

## Option A: Synthetic Data (Default)

When no PyPSA-GB data is available, the model automatically generates calibrated synthetic profiles based on published GB statistics:

- **Demand**: Calibrated to ESPENI/National Grid data
  - Mean: ~32 GW, Peak: ~54 GW
  - Includes diurnal, seasonal, and weekend patterns

- **Solar PV**: Target 10.5% capacity factor (DUKES)
  - Accounts for UK latitude (52°N)
  - Includes cloud cover autocorrelation

- **Onshore Wind**: Target 26.5% capacity factor
  - Weather-driven with multi-day autocorrelation

- **Offshore Wind**: Target 40% capacity factor
  - Higher and more consistent than onshore

The synthetic data is suitable for reproducing the **qualitative findings** of the paper but may differ slightly from runs using real weather data.

## Option B: PyPSA-GB Data (Recommended)

For higher fidelity results, use real demand and generation data from PyPSA-GB.

### Step 1: Clone PyPSA-GB

```bash
git clone https://github.com/andrewlyden/PyPSA-GB.git
```

### Step 2: Download Data

Follow the PyPSA-GB documentation to download the required datasets:
- **ESPENI demand data**: Available from Sheffield Solar/National Grid
- **Atlite renewable profiles**: Generated from ERA5 reanalysis data

The relevant files are:
```
PyPSA-GB/
├── data/
│   ├── demand/
│   │   └── espeni.csv                 # Half-hourly GB demand
│   └── renewables/
│       └── atlite/
│           └── outputs/
│               ├── PV/                # Solar capacity factors
│               ├── Wind_Onshore/      # Onshore wind profiles
│               └── Wind_Offshore/     # Offshore wind profiles
```

### Step 3: Configure the Model

Set the path in `src/gb_electricity_model.py`:

```python
PYPSA_GB_PATH = "./PyPSA-GB"  # or absolute path
```

Or pass it via command line:
```bash
python src/gb_electricity_model.py --pypsa-gb-path ./PyPSA-GB
```

## Data Format

### Demand (ESPENI)
- Half-hourly MW values
- Column: `POWER_ESPENI_MW` or similar
- Multiple years available; model filters by `WEATHER_YEAR` (default: 2019)

### Renewable Profiles
The model can handle two formats:

1. **Capacity factors** (0-1 range):
   - Column: `capacity_factor` or `p_max_pu`

2. **MW output per generator** (sum of columns):
   - Multiple columns representing individual wind farms or solar sites
   - Model sums and normalizes by installed capacity

## References

- Wilson, G., et al. (2021). ESPENI dataset. Sheffield Solar.
- Hofmann, M., et al. (2021). Atlite: A light-weight library for computing renewable power potentials. *Journal of Open Source Software*.
- Lyden, A., et al. (2022). PyPSA-GB: An open-source model of the GB electricity system. GitHub repository.

## Note on Weather Years

The default weather year is 2019. This affects:
- Demand patterns (winter/summer peaks)
- Wind availability (storm events)
- Solar irradiance (cloud patterns)

For sensitivity analysis, try different weather years (2015-2020 typically available).
