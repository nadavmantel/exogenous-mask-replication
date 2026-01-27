"""
Figure 2: Endogenous vs Exogenous Model Comparison
4-panel layout showing the difference in threshold dynamics and policy surfaces
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# =============================================================================
# CSV FILE PATHS
# =============================================================================
BATTERY_SWEEP_EXO = "battery_subsidy_sweep_exogenous.csv"
BATTERY_SWEEP_ENDO = "battery_subsidy_sweep_endogenous.csv"
GRID_EXO = "solar_battery_subsidy_grid_exogenous.csv"
GRID_ENDO = "solar_battery_subsidy_grid_endoogenous.csv"

OUTPUT_PATH = "figure2_endo_vs_exo.png"
# =============================================================================


def load_data():
    """Load all CSV files."""
    battery_exo = pd.read_csv(BATTERY_SWEEP_EXO)
    battery_endo = pd.read_csv(BATTERY_SWEEP_ENDO)
    grid_exo = pd.read_csv(GRID_EXO)
    grid_endo = pd.read_csv(GRID_ENDO)
    
    # Clean column names
    for df in [battery_exo, battery_endo, grid_exo, grid_endo]:
        df.columns = df.columns.str.strip()
    
    return battery_exo, battery_endo, grid_exo, grid_endo


def create_figure(battery_exo, battery_endo, grid_exo, grid_endo):
    """Create 4-panel comparison figure. Top row: Exogenous, Bottom row: Endogenous."""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Colors
    threshold_color = '#E63946'  # Red
    deploy_color = '#2A9D8F'     # Teal
    low_color = '#F4A261'        # Orange
    
    # Custom colormap for heatmaps
    colors = ['#F4A261', '#FFFFFF', '#2A9D8F']
    cmap = LinearSegmentedColormap.from_list('battery_growth', colors)
    
    # ==========================================================================
    # PANEL (a): Battery-only - EXOGENOUS (smooth curve) - TOP LEFT
    # ==========================================================================
    ax1 = axes[0, 0]
    
    x_exo = battery_exo['initial_subsidy_per_kw']
    y_exo = battery_exo['battery_end_gw']
    gas_exo = battery_exo['gas_growth_gw']
    
    ax1.plot(x_exo, y_exo, 'o-', color=deploy_color, linewidth=2, markersize=4)
    
    # Find threshold and add horizontal dashed line at zero gas growth capacity
    zero_gas_mask = gas_exo <= 0
    if zero_gas_mask.any():
        threshold_sub = x_exo[zero_gas_mask].min()
        threshold_battery = battery_exo[battery_exo['initial_subsidy_per_kw'] == threshold_sub]['battery_end_gw'].values[0]
        ax1.axhline(y=threshold_battery, color=threshold_color, linestyle='--', linewidth=2, alpha=0.7)
        ax1.text(x_exo.max() - 2, threshold_battery + 30, f'{threshold_battery:.0f} GW\n(zero gas growth)', 
                fontsize=9, ha='right', color=threshold_color)
    
    ax1.annotate('Gradual increase\n(no feedback)', 
                xy=(395, 150), fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.set_xlabel('Battery subsidy (£/kW)', fontsize=11)
    ax1.set_ylabel('Battery capacity in 2049 (GW)', fontsize=11)
    ax1.set_title('(a) Battery-only subsidy — Exogenous', fontsize=12, fontweight='bold')
    ax1.set_ylim(bottom=0)
    
    # ==========================================================================
    # PANEL (b): Coordinated surface - EXOGENOUS (vertical contour) - TOP RIGHT
    # ==========================================================================
    ax2 = axes[0, 1]
    
    pivot_battery_exo = grid_exo.pivot(
        index='solar_initial', 
        columns='battery_initial', 
        values='battery_growth_gw'
    )
    pivot_gas_exo = grid_exo.pivot(
        index='solar_initial', 
        columns='battery_initial', 
        values='gas_growth_gw'
    )
    
    im1 = ax2.imshow(pivot_battery_exo.values, cmap=cmap, aspect='auto', origin='lower',
                     extent=[pivot_battery_exo.columns.min(), pivot_battery_exo.columns.max(),
                            pivot_battery_exo.index.min(), pivot_battery_exo.index.max()])
    
    # Add contour at gas_growth = 0
    X_exo, Y_exo = np.meshgrid(pivot_gas_exo.columns, pivot_gas_exo.index)
    try:
        contour_exo = ax2.contour(X_exo, Y_exo, pivot_gas_exo.values, levels=[0], 
                                  colors=[threshold_color], linewidths=2.5)
        ax2.clabel(contour_exo, fmt='0 GW\ngas growth', fontsize=8)
    except:
        pass
    
    # Mark cheapest zero-gas point
    successful_exo = grid_exo[grid_exo['gas_growth_gw'] <= 0].copy()
    if len(successful_exo) > 0:
        best_exo = successful_exo.loc[successful_exo['total_subsidy_cost_bn'].idxmin()]
        ax2.plot(best_exo['battery_initial'], best_exo['solar_initial'], 
                '*', color='white', markersize=15, markeredgecolor='black', markeredgewidth=1.5)
        ax2.annotate(f"£{int(best_exo['battery_initial'])} + £{int(best_exo['solar_initial'])}", 
                    xy=(best_exo['battery_initial'], best_exo['solar_initial']),
                    xytext=(30, 10), textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='black'))
    
    ax2.annotate('Vertical boundary:\nsolar has no effect', 
                xy=(420, 60), fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.colorbar(im1, ax=ax2, label='Battery growth (GW)')
    ax2.set_xlabel('Battery subsidy (£/kW)', fontsize=11)
    ax2.set_ylabel('Solar subsidy (£/kW)', fontsize=11)
    ax2.set_title('(b) Coordinated subsidy — Exogenous', fontsize=12, fontweight='bold')
    
    # ==========================================================================
    # PANEL (c): Battery-only - ENDOGENOUS (sharp threshold) - BOTTOM LEFT
    # ==========================================================================
    ax3 = axes[1, 0]
    
    x_endo = battery_endo['initial_subsidy_per_kw']
    y_endo = battery_endo['battery_end_gw']
    gas_endo = battery_endo['gas_growth_gw']
    
    ax3.plot(x_endo, y_endo, 'o-', color=deploy_color, linewidth=2, markersize=4)
    
    # Find threshold
    zero_gas_mask_endo = gas_endo <= 0
    if zero_gas_mask_endo.any():
        threshold_sub_endo = x_endo[zero_gas_mask_endo].min()
        below_threshold_endo = x_endo[x_endo < threshold_sub_endo].max()
        ax3.axvspan(below_threshold_endo, threshold_sub_endo, alpha=0.3, color=threshold_color,
                   label=f'Threshold\n(£{below_threshold_endo:.0f}–{threshold_sub_endo:.0f}/kW)')
        
        # Mark the threshold battery capacity
        threshold_battery = battery_endo[battery_endo['initial_subsidy_per_kw'] == threshold_sub_endo]['battery_end_gw'].values[0]
        ax3.axhline(y=threshold_battery, color=threshold_color, linestyle=':', alpha=0.5)
    
    ax3.annotate('Sharp threshold\n(feedback loop)', 
                xy=(520, 400), fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax3.set_xlabel('Battery subsidy (£/kW)', fontsize=11)
    ax3.set_ylabel('Battery capacity in 2049 (GW)', fontsize=11)
    ax3.set_title('(c) Battery-only subsidy — Endogenous', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.set_ylim(bottom=0)
    
    # ==========================================================================
    # PANEL (d): Coordinated surface - ENDOGENOUS (sloped contour) - BOTTOM RIGHT
    # ==========================================================================
    ax4 = axes[1, 1]
    
    pivot_battery_endo = grid_endo.pivot(
        index='solar_initial', 
        columns='battery_initial', 
        values='battery_growth_gw'
    )
    pivot_gas_endo = grid_endo.pivot(
        index='solar_initial', 
        columns='battery_initial', 
        values='gas_growth_gw'
    )
    
    im2 = ax4.imshow(pivot_battery_endo.values, cmap=cmap, aspect='auto', origin='lower',
                     extent=[pivot_battery_endo.columns.min(), pivot_battery_endo.columns.max(),
                            pivot_battery_endo.index.min(), pivot_battery_endo.index.max()])
    
    # Add contour at gas_growth = 0
    X_endo, Y_endo = np.meshgrid(pivot_gas_endo.columns, pivot_gas_endo.index)
    try:
        contour_endo = ax4.contour(X_endo, Y_endo, pivot_gas_endo.values, levels=[0], 
                                   colors=[threshold_color], linewidths=2.5)
        ax4.clabel(contour_endo, fmt='0 GW\ngas growth', fontsize=8)
    except:
        pass
    
    # Mark cheapest zero-gas point
    successful_endo = grid_endo[grid_endo['gas_growth_gw'] <= 0].copy()
    if len(successful_endo) > 0:
        best_endo = successful_endo.loc[successful_endo['total_subsidy_cost_bn'].idxmin()]
        ax4.plot(best_endo['battery_initial'], best_endo['solar_initial'], 
                '*', color='white', markersize=15, markeredgecolor='black', markeredgewidth=1.5)
        ax4.annotate(f"£{int(best_endo['battery_initial'])} + £{int(best_endo['solar_initial'])}", 
                    xy=(best_endo['battery_initial'], best_endo['solar_initial']),
                    xytext=(-50, 15), textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='black'))
    
    ax4.annotate('Sloped boundary:\npolicy complementarity', 
                xy=(492, 100), fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.colorbar(im2, ax=ax4, label='Battery growth (GW)')
    ax4.set_xlabel('Battery subsidy (£/kW)', fontsize=11)
    ax4.set_ylabel('Solar subsidy (£/kW)', fontsize=11)
    ax4.set_title('(d) Coordinated subsidy — Endogenous', fontsize=12, fontweight='bold')
    
    # ==========================================================================
    # Add row labels on the left side
    # ==========================================================================
    fig.text(0.02, 0.72, 'Exogenous\nLearning', fontsize=12, fontweight='bold', 
             ha='center', va='center', rotation=90)
    fig.text(0.02, 0.28, 'Endogenous\nLearning', fontsize=12, fontweight='bold', 
             ha='center', va='center', rotation=90)
    
    # ==========================================================================
    # Print summary statistics
    # ==========================================================================
    print("\n" + "="*60)
    print("SUMMARY: Exogenous vs Endogenous Comparison")
    print("="*60)
    
    # Exogenous
    if len(successful_exo) > 0:
        print(f"\nEXOGENOUS - Cheapest zero gas growth:")
        print(f"  Battery: £{best_exo['battery_initial']:.0f}/kW")
        print(f"  Solar: £{best_exo['solar_initial']:.0f}/kW")
        print(f"  Total cost: £{best_exo['total_subsidy_cost_bn']:.1f}bn")
    
    # Endogenous
    if len(successful_endo) > 0:
        print(f"\nENDOGENOUS - Cheapest zero gas growth:")
        print(f"  Battery: £{best_endo['battery_initial']:.0f}/kW")
        print(f"  Solar: £{best_endo['solar_initial']:.0f}/kW")
        print(f"  Total cost: £{best_endo['total_subsidy_cost_bn']:.1f}bn")
    
    # Battery-only thresholds
    bat_exo_zero = battery_exo[battery_exo['gas_growth_gw'] <= 0]
    bat_endo_zero = battery_endo[battery_endo['gas_growth_gw'] <= 0]
    
    if len(bat_exo_zero) > 0:
        print(f"\nEXOGENOUS battery-only threshold: £{bat_exo_zero['initial_subsidy_per_kw'].min():.0f}/kW")
    if len(bat_endo_zero) > 0:
        print(f"ENDOGENOUS battery-only threshold: £{bat_endo_zero['initial_subsidy_per_kw'].min():.0f}/kW")
    
    # ==========================================================================
    # Save
    # ==========================================================================
    plt.tight_layout()
    plt.subplots_adjust(left=0.08)
    
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_PATH.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Saved: {OUTPUT_PATH.replace('.png', '.pdf')}")
    
    plt.show()
    
    return fig


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("Loading data...")
    battery_exo, battery_endo, grid_exo, grid_endo = load_data()
    
    print(f"\nExogenous battery sweep: {len(battery_exo)} rows")
    print(f"Endogenous battery sweep: {len(battery_endo)} rows")
    print(f"Exogenous grid: {len(grid_exo)} rows")
    print(f"Endogenous grid: {len(grid_endo)} rows")
    
    create_figure(battery_exo, battery_endo, grid_exo, grid_endo)
