# The Exogenous Mask: Replication Code

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPSA](https://img.shields.io/badge/PyPSA-0.26+-green.svg)](https://pypsa.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Replication code for **"The Exogenous Mask: How Cost Assumptions Hide Sensitive Intervention Points in Energy Transition Policy"** by Nadav Mantel.

## Abstract

Energy system models commonly represent technological cost reductions as exogenous functions of time. This modelling choice removes the feedback between deployment and costs that characterizes learning-by-doing. This repository demonstrates how exogenous assumptions can mask **sensitive intervention points (SIPs)** and non-linear system dynamics in future energy pathway models. Using an endogenous learning capacity expansion model of the Great Britain power system, we show that battery subsidies exhibit a sharp deployment threshold: a <1% change in subsidy magnitude produces order-of-magnitude differences in outcomes. These dynamics are entirely absent in the same model with exogenous learning.

## Repository Structure

```
exogenous-mask-replication/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── LICENSE                             # MIT License
│
├── src/
│   └── gb_electricity_model.py         # Main capacity expansion model
│
├── figures/
│   ├── figure1_integration_barrier.py  # Figure 1: Solar-battery integration barrier
│   └── figure2_threshold_dynamics.py   # Figure 2: Subsidy threshold dynamics
│
├── data/
│   └── README.md                       # Instructions for obtaining PyPSA-GB data
│
└── results/                            # Generated outputs (created on first run)
    ├── gb_model_results.csv
    ├── battery_subsidy_sweep_endogenous.csv
    └── solar_battery_subsidy_grid_endogenous.csv
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/[username]/exogenous-mask-replication.git
cd exogenous-mask-replication

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Model

The model can run in three modes:

```bash
# Mode 1: Basic comparison (exogenous vs endogenous learning)
python src/gb_electricity_model.py --mode normal

# Mode 2: Battery subsidy sweep (finds threshold)
python src/gb_electricity_model.py --mode battery_sweep

# Mode 3: Solar + battery coordinated subsidy grid search
python src/gb_electricity_model.py --mode solar_battery_sweep
```

### 3. Generate Figures

After running the model sweeps:

```bash
# Figure 1: Solar-battery integration barrier
python figures/figure1_integration_barrier.py

# Figure 2: Threshold dynamics and coordinated subsidies
python figures/figure2_threshold_dynamics.py
```

## Data Sources

The model uses demand and renewable generation profiles from **PyPSA-GB** (Lyden et al., 2022). The code can operate in two modes:

### Option A: Use Synthetic Data (Default)
If PyPSA-GB data is not available, the model generates calibrated synthetic profiles based on published GB statistics (DUKES, National Grid ESO). This is suitable for reproducing the qualitative findings.

### Option B: Use Real Data (Recommended)
For higher fidelity results, download PyPSA-GB data:

1. Clone the PyPSA-GB repository:
   ```bash
   git clone https://github.com/andrewlyden/PyPSA-GB.git
   ```

2. Set the path in `gb_electricity_model.py`:
   ```python
   PYPSA_GB_PATH = "./PyPSA-GB"
   ```

**Data files used:**
- `espeni.csv` — Half-hourly GB electricity demand (ESPENI dataset)
- `Wind_Offshore_2019.csv` — Hourly offshore wind generation
- `Wind_Onshore_2019.csv` — Hourly onshore wind generation  
- `PV_2019.csv` — Hourly solar PV generation

## Model Description

### Overview

A stylised capacity expansion model of the GB electricity system solved myopically over 2025–2049 in two-year intervals. GB is represented as a single "copper-plate" node with no transmission constraints.

### Technologies

| Technology | Capital Cost (2024) | Learning Rate | Initial Capacity |
|------------|---------------------|---------------|------------------|
| Solar PV | £450/kW | 20% | 16 GW |
| Offshore Wind | £1,800/kW | 12% | 15 GW |
| Onshore Wind | £950/kW | 8% | 15 GW |
| Battery (2h) | £600/kW | 18% | 3 GW |
| Gas CCGT | £750/kW | 0% | 36 GW |
| Gas Peaker | £500/kW | 0% | 5 GW |
| Nuclear | £8,000/kW | 0% | 5 GW |

### Learning Formulations

**Exogenous learning:** Technology costs follow predetermined global trajectories based on IEA/IRENA projections. UK deployment does not affect costs.

**Endogenous learning:** Costs follow Wright's Law:

$$C(K) = C_0 \cdot \left(\frac{K}{K_0}\right)^{-\alpha}$$

where $\alpha = -\log_2(1 - LR)$ and UK deployment contributes to global cumulative capacity via assumed market shares.

### Policy Scenarios

The model tests three policy instruments:
1. **Carbon price** — Applied to gas generation marginal costs
2. **Battery-only subsidy** — Direct capital cost reduction for storage
3. **Coordinated solar-battery subsidy** — Joint subsidies for solar PV and battery storage

## Key Results

### Finding 1: Integration Barrier

Under endogenous learning with current policy, batteries remain at ~3 GW while solar reaches only 41 GW by 2049. Under exogenous assumptions with identical policy, batteries reach 65 GW and solar 90 GW. The absence of early battery deployment creates an "integration barrier" that prevents solar from achieving sufficient value to drive continued deployment.

### Finding 2: Sharp Subsidy Thresholds

Battery subsidies exhibit a discontinuous threshold: between £511–512/kW, a 0.2% change in subsidy magnitude produces order-of-magnitude differences in deployment. This SIP is structurally absent from exogenous formulations.

### Finding 3: Coordinated Subsidy Efficiency

Coordinated solar-battery subsidies achieve zero gas growth at approximately **78% of the cost** of battery-only subsidies—a 22% savings. This reflects compounding returns from joint learning feedbacks.

## Configuration

Key parameters can be modified in `gb_electricity_model.py`:

```python
# Time periods
PERIODS = list(range(2025, 2051, 2))  # Every 2 years

# UK share of global market (affects endogenous learning)
UK_GLOBAL_SHARE = {
    "solar": 0.02,          # UK ~2% of global solar
    "offshore_wind": 0.05,  # UK ~5% (major player)
    "battery": 0.02,        # UK ~2% of global storage
}

# Sweep parameters (adjust for resolution vs. speed)
BATTERY_SWEEP_RANGE = (480, 520, 1)      # £/kW: start, stop, step
SOLAR_RANGE = (100, 120, 1)              # For grid search
BATTERY_RANGE = (450, 500, 1)            # For grid search
```

## Output Files

| File | Description |
|------|-------------|
| `gb_model_results.csv` | Period-by-period capacity and costs for all scenarios |
| `battery_subsidy_sweep_endogenous.csv` | Battery-only subsidy sweep results |
| `solar_battery_subsidy_grid_endogenous.csv` | Coordinated subsidy grid search |
| `figure1_integration_barrier.png/pdf` | Figure 1 |
| `figure2_threshold_dynamics.png/pdf` | Figure 2 |

## Computational Requirements

- **Basic run:** ~5 minutes on a standard laptop
- **Battery sweep (40 levels):** ~30 minutes
- **Solar-battery grid (20×20):** ~4–6 hours

The model uses the HiGHS solver (open-source, included with PyPSA).

## Limitations

This model deliberately abstracts from many real-world features:
- No transmission constraints (copper-plate)
- No build-rate limits
- No demand-side flexibility
- No foresight about future costs
- Simplified technology representations

These simplifications amplify dynamic mechanisms. **Results should be interpreted as illustrative of mechanisms, not as quantitative policy guidance.**

## Citation

If you use this code, please cite:

```bibtex
@article{mantel2025exogenous,
  title={The Exogenous Mask: How Cost Assumptions Hide Sensitive 
         Intervention Points in Energy Transition Policy},
  author={Mantel, Nadav},
  journal={[Journal]},
  year={2025}
}
```

## Dependencies

- Python ≥ 3.10
- PyPSA ≥ 0.26
- pandas ≥ 2.0
- numpy ≥ 1.24
- matplotlib ≥ 3.7
- HiGHS solver (bundled with PyPSA)

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- PyPSA-GB (Lyden et al., 2022) for demand and generation data
- PyPSA development team for the power system modelling framework

## Contact

For questions or issues, please open a GitHub issue or contact [your email].
