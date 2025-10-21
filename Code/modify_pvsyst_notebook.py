#!/usr/bin/env python3
"""
Modify PVsyst notebook to implement scale-then-clip workflow.
"""

import json
from pathlib import Path

# Read the notebook
notebook_path = Path('25_09_05_Data_visualiser_matching_inv.ipynb')
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

print(f"Loaded notebook with {len(notebook['cells'])} cells")

# ============================================================================
# STEP 1: Add documentation cell before cell 16
# ============================================================================

documentation_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## ⚠️ IMPORTANT METHODOLOGY UPDATE\n",
        "\n",
        "This notebook implements the **scale-then-clip workflow** for inverter power clipping analysis:\n",
        "\n",
        "#### What Changed:\n",
        "- **Previous approach (clip-then-scale)**: `scale × Σ(clip(hourly))` - Clipped hourly power BEFORE aggregation\n",
        "- **New approach (scale-then-clip)**: `Σ(clip(scale × hourly))` - Scales hourly power FIRST, then clips, then aggregates\n",
        "\n",
        "#### Why This Matters:\n",
        "The order of operations significantly affects the optimization results because:\n",
        "1. **Physical accuracy**: Inverter clipping occurs at instantaneous power levels (hourly), not daily energy totals\n",
        "2. **Scale factor correctness**: Scaling before clipping ensures the optimized factor reflects true system performance\n",
        "3. **Consistency**: Matches the methodology in `batch_pvsyst_evaluation_scaled_clip.py` and SunSolve notebook\n",
        "\n",
        "#### Implementation Details:\n",
        "1. Hourly simulation power is kept unclipped initially\n",
        "2. During optimization: `hourly_power × scale → clip() → resample('h').mean() → calculate MBE`\n",
        "3. After optimization: Apply the same scale-then-clip workflow to full dataset\n",
        "\n",
        "#### Mathematical Impact:\n",
        "- Clip-then-scale: `S × Σ(min(P, C))` where S=scale, P=power, C=clip threshold\n",
        "- Scale-then-clip: `Σ(min(S×P, C))` (physically correct)\n",
        "\n",
        "These produce different results when clipping occurs!\n"
    ]
}

# Insert documentation cell before cell 16
notebook['cells'].insert(16, documentation_cell)
print("Added documentation cell before optimization cell")

# ============================================================================
# STEP 2: Create new version of cell 16 (now cell 17) with scale-then-clip workflow
# ============================================================================

new_cell_17_source = '''# Define options at the beginning
apply_clipping = True  # Set to False to disable clipping
clipping_threshold = 2.392 # MW - power clipping threshold
target_mbe_tolerance = 1e-13  # Target absolute MBE to achieve

# First check if EArray needs conversion (avoid error if already numeric)
if pd.api.types.is_object_dtype(simulation_results_df['EArray']):
    # Only convert if it's still a string/object type
    simulation_results_df['EArray'] = pd.to_numeric(
        simulation_results_df['EArray'].str.replace(',', '.'),
        errors='coerce'
    )
    print("Converted EArray column from string to numeric")
else:
    print("EArray column is already numeric type:", simulation_results_df['EArray'].dtype)

# ============================================================================
# SCALE-THEN-CLIP WORKFLOW: Do NOT clip here!
# Clipping will be applied AFTER scaling during optimization
# ============================================================================
print("\\n⚠️ SCALE-THEN-CLIP WORKFLOW ACTIVE")
print("Power clipping will be applied AFTER scaling optimization")

# Now set the index and sort
if 'timestamp' in simulation_results_df.columns:
    simulation_results_df.set_index('timestamp', inplace=True)
    simulation_results_df.sort_index(inplace=True)
    print("Set timestamp as index")

# Convert Power (MW) to Energy (MWh) for measured 5-minute data
df['Energy_MWh'] = df[inverter] * (5/60)  # 5 minutes = 5/60 hours

# Resample measured data to HOURLY averages (to match simulation resolution)
hourly_actual_power = df[inverter].resample('h').mean()
print(f"Resampled actual power from {len(df)} 5-minute intervals to {len(hourly_actual_power)} hourly averages")

# ============================================================================
# CRITICAL: Create metrics_df with HOURLY data for scale-then-clip workflow
# ============================================================================
print("\\n=== DATA FILTERING FOR METRICS CALCULATION ===")
print("Step 1: Creating metrics dataframe with HOURLY data")
metrics_df = pd.DataFrame()
metrics_df['Actual'] = hourly_actual_power  # Hourly measured power (MW)
metrics_df['Simulated'] = simulation_results_df['EArray']  # Hourly simulated power (MW), NOT YET CLIPPED

# Filter to include only timestamps where both actual and simulated data exist
initial_rows = len(metrics_df)
metrics_df = metrics_df.dropna()
print(f"After filtering for matching timestamps: {len(metrics_df)} hourly data points (removed {initial_rows - len(metrics_df)} NaN values)")

# Load and apply maintenance-free days filter
print("\\nStep 2: Loading and applying maintenance-free days filter")
maintenance_free_days_file = r"C:\\Users\\z5183876\\OneDrive - UNSW\\Documents\\GitHub\\25_09_02_Bomen_bifacial_gain_2021\\Results\\remaining_dates_2021.txt"
try:
    with open(maintenance_free_days_file, 'r') as f:
        maintenance_free_days = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Loaded {len(maintenance_free_days)} maintenance-free days")

    # Convert maintenance-free days to datetime
    maintenance_free_dates = pd.to_datetime(maintenance_free_days)

    # Filter metrics to only include maintenance-free days
    metrics_df['date'] = metrics_df.index.date
    initial_rows = len(metrics_df)
    metrics_df = metrics_df[metrics_df['date'].isin([date.date() for date in maintenance_free_dates])]

    print(f"After filtering for maintenance-free days: {len(metrics_df)} hourly points (removed {initial_rows - len(metrics_df)} points)")

except Exception as e:
    print(f"Error loading maintenance-free days: {e}")
    print("Continuing with only the non-NaN data points")

# Print filtered data statistics
if len(metrics_df) > 0:
    print(f"\\nFiltered data date range: {metrics_df.index.min()} to {metrics_df.index.max()}")
    print(f"Number of unique days in filtered data: {len(metrics_df['date'].unique())}")
else:
    print("\\nWarning: No data points remain after filtering!")

# ============================================================================
# SCALE-THEN-CLIP WORKFLOW FUNCTIONS
# ============================================================================

def compute_hourly_scaled_clipped(hourly_power, scale_factor):
    """
    Apply scale-then-clip workflow to hourly power data.

    Args:
        hourly_power: Raw hourly power series (MW)
        scale_factor: Scaling factor to apply

    Returns:
        Hourly power after scaling and optional clipping (MW)
    """
    # Step 1: SCALE hourly power
    scaled_hourly = hourly_power * scale_factor

    # Step 2: CLIP scaled power (if enabled)
    if apply_clipping and clipping_threshold is not None:
        clipped_hourly = scaled_hourly.clip(upper=clipping_threshold)
        clipped_count = (scaled_hourly > clipping_threshold).sum()
        if clipped_count > 0:
            print(f"  Clipped {clipped_count} hours at {clipping_threshold} MW")
    else:
        clipped_hourly = scaled_hourly

    return clipped_hourly


def calculate_mbe_scaleThenClip(scale_factor):
    """
    Calculate Mean Bias Error using scale-then-clip workflow.

    This function applies scaling first, then clipping, then calculates MBE
    on the filtered hourly metrics data.

    Args:
        scale_factor: Scaling factor to apply

    Returns:
        Mean Bias Error (MBE) in MW
    """
    # Apply scale-then-clip to simulated hourly data
    scaled_clipped = compute_hourly_scaled_clipped(
        metrics_df['Simulated'],
        scale_factor
    )

    # Calculate MBE: mean(simulated - actual) at hourly resolution
    mbe = np.mean(scaled_clipped - metrics_df['Actual'])

    return mbe

print("\\n=== SCALING FACTOR OPTIMIZATION ===")
print(f"Target MBE tolerance: {target_mbe_tolerance:.2e}")


def find_optimal_scaling_factor(min_factor=0.5, max_factor=2.0, max_iterations=100):
    """
    Find optimal scaling factor using binary search to minimize MBE.

    Physical interpretation:
    - Positive MBE: Simulated values too high → decrease scale factor
    - Negative MBE: Simulated values too low → increase scale factor
    """
    print("\\nStarting binary search for optimal scaling factor...")
    print(f"Initial search range: [{min_factor:.8f}, {max_factor:.8f}]")

    iterations = 0
    best_factor = None
    best_mbe = float('inf')

    while iterations < max_iterations:
        iterations += 1
        mid_factor = (min_factor + max_factor) / 2

        # Calculate MBE for the current factor using scale-then-clip
        mbe = calculate_mbe_scaleThenClip(mid_factor)

        # Track the best factor found
        if abs(mbe) < abs(best_mbe):
            best_factor = mid_factor
            best_mbe = mbe

        print(f"Iteration {iterations}: Factor = {mid_factor:.10f}, MBE = {mbe:.10e}")

        # Check if we've reached the target precision
        if abs(mbe) < target_mbe_tolerance:
            print(f"✓ Converged! Found factor with MBE below tolerance.")
            return mid_factor, mbe, iterations

        # Adjust search range based on MBE sign
        if mbe > 0:  # MBE is positive, need to decrease factor
            max_factor = mid_factor
        else:  # MBE is negative, need to increase factor
            min_factor = mid_factor

        # Check if search range is too small (reached numerical precision limit)
        if max_factor - min_factor < 1e-15:
            print(f"! Reached numerical precision limit after {iterations} iterations.")
            print(f"  Best MBE achieved: {best_mbe:.12e}")
            return best_factor, best_mbe, iterations

    print(f"! Reached maximum iterations ({max_iterations}).")
    print(f"  Best MBE achieved: {best_mbe:.12e}")
    return best_factor, best_mbe, iterations

# Only proceed with optimization if we have data
if len(metrics_df) > 0:
    # Find the optimal scaling factor using scale-then-clip workflow
    optimal_factor, final_mbe, iterations = find_optimal_scaling_factor()

    print("\\n=== OPTIMIZATION RESULTS ===")
    print(f"Optimal scaling factor: {optimal_factor:.10f}")
    print(f"Final MBE: {final_mbe:.12e} MW")
    print(f"Found in {iterations} iterations")

    # Apply scale-then-clip with optimal factor
    print(f"\\nApplying optimal scale factor {optimal_factor:.10f} to PVsyst data...")
    print(f"Applying scale-then-clip with optimal factor {optimal_factor:.10f}...")

    # Use the helper function to apply scale-then-clip to hourly data
    metrics_df['Simulated_scaled'] = compute_hourly_scaled_clipped(
        metrics_df['Simulated'],
        optimal_factor
    )

    metrics_df['Simulated_display'] = metrics_df['Simulated_scaled']

    # Calculate error metrics with optimal scaling (using hourly filtered data)
    # 1. RMSE (Root Mean Square Error)
    rmse = np.sqrt(mean_squared_error(metrics_df['Actual'], metrics_df['Simulated_display']))

    # 2. CRMSE (Centralized Root Mean Square Error)
    actual_centered = metrics_df['Actual'] - metrics_df['Actual'].mean()
    simulated_centered = metrics_df['Simulated_display'] - metrics_df['Simulated_display'].mean()
    crmse = np.sqrt(np.mean((actual_centered - simulated_centered)**2))

    # 3. MBE (Mean Bias Error) - should be very close to zero
    mbe = np.mean(metrics_df['Simulated_display'] - metrics_df['Actual'])

    print("\\n=== ERROR METRICS WITH OPTIMAL SCALING (ON FILTERED DATA) ===")
    print(f"RMSE: {rmse:.4f} MW")
    print(f"CRMSE: {crmse:.4f} MW")
    print(f"MBE: {mbe:.12e} MW")

    # Calculate normalized RMSE
    nrmse = rmse / (metrics_df['Actual'].max() - metrics_df['Actual'].min())
    print(f"nRMSE: {nrmse:.4%} of actual range")

    # Verify if target was met
    if abs(mbe) < target_mbe_tolerance:
        print(f"✓ SUCCESS: |MBE| = {abs(mbe):.12e} < {target_mbe_tolerance:.2e}")
    else:
        print(f"! WARNING: |MBE| = {abs(mbe):.12e} > {target_mbe_tolerance:.2e}")
        print("  Target tolerance not achieved exactly due to numerical precision limits.")

    # ========================================================================
    # VISUALIZATION: Apply scale-then-clip to full dataset for plotting
    # ========================================================================
    plt.figure(figsize=long_hoz_figsize)

    # Apply scale-then-clip to ALL hourly simulation data
    all_hourly_scaled_clipped = compute_hourly_scaled_clipped(
        simulation_results_df['EArray'],
        optimal_factor
    )

    # Convert hourly power to daily energy for visualization
    all_daily_simulated_scaled = (all_hourly_scaled_clipped * 1.0).resample('D').sum()  # MWh/day
    all_daily_actual = df['Energy_MWh'].resample('D').sum()  # MWh/day

    earray_label = f'Simulated (Daily Energy, optimal scale: {optimal_factor:.6f})'
    if apply_clipping:
        earray_label += f', clipped at {clipping_threshold} MW'

    # Filter for 2021 data only
    year_2021_simulated = all_daily_simulated_scaled[all_daily_simulated_scaled.index.year == 2021]
    year_2021_actual = all_daily_actual[all_daily_actual.index.year == 2021]

    print(f"\\n=== 2021 DATA FILTERING ===")
    print(f"Total days in 2021 with simulation data: {len(year_2021_simulated)}")
    print(f"Total days in 2021 with actual data: {len(year_2021_actual)}")

    # Plot 2021 scaled simulation data
    plt.plot(year_2021_simulated.index, year_2021_simulated,
             linewidth=1.5, color='#ff7f0e', label=earray_label)

    # Plot 2021 actual energy data
    plt.plot(year_2021_actual.index, year_2021_actual,
             linewidth=1.5, color='#1f77b4', label='Actual (Daily Energy)')

    # Add labels and title
    plt.xlabel('Date', fontsize=axis_label_size)
    plt.ylabel('Energy (MWh/day)', fontsize=axis_label_size)

    # Add metrics to title
    plt.title(f'Optimally Scaled Simulation vs Actual Daily Energy (2021 full year)\\n'
              f'Scale: {optimal_factor:.6f}, RMSE: {rmse:.2f} MW\\n'
              f'CRMSE: {crmse:.2f} MW, MBE: {mbe:.2e} MW*',
              fontsize=title_size)

    # Add note about metrics
    plt.figtext(0.5, 0.01,
                '* Metrics calculated on hourly filtered data (maintenance-free days)',
                ha='center', fontsize=10, style='italic')

    # Add legend and formatting
    plt.legend(fontsize=text_size)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gcf().autofmt_xdate()
    plt.tight_layout()

    # Show plot
    plt.show()

    # Print statistics for 2021 data
    print(f"\\n=== STATISTICS WITH OPTIMAL SCALING (2021 DATA) ===")
    print(f"2021 simulation data points: {len(year_2021_simulated)}")
    print(f"2021 actual data points: {len(year_2021_actual)}")
    print(f"Scaled Daily Energy Min: {year_2021_simulated.min():.2f} MWh/day")
    print(f"Scaled Daily Energy Max: {year_2021_simulated.max():.2f} MWh/day")
    print(f"Scaled Daily Energy Mean: {year_2021_simulated.mean():.2f} MWh/day")
    print(f"Actual Daily Energy Min: {year_2021_actual.min():.2f} MWh/day")
    print(f"Actual Daily Energy Max: {year_2021_actual.max():.2f} MWh/day")
    print(f"Actual Daily Energy Mean: {year_2021_actual.mean():.2f} MWh/day")
else:
    print("\\nERROR: No data points remain after filtering. Cannot proceed with optimization.")
'''

# Update cell 17 (formerly cell 16) with new source
notebook['cells'][17]['source'] = new_cell_17_source.split('\n')

print("Updated optimization cell with scale-then-clip workflow")

# ============================================================================
# STEP 3: Save the modified notebook
# ============================================================================

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"\nSuccessfully modified notebook: {notebook_path}")
print(f"Total cells after modification: {len(notebook['cells'])}")
print("\nChanges applied:")
print("  1. Added documentation cell explaining scale-then-clip methodology")
print("  2. Modified optimization cell to implement scale-then-clip workflow")
print("  3. Replaced clip-then-scale with scale-then-clip in all relevant sections")
print("\nBackup file available: 25_09_05_Data_visualiser_matching_inv_BACKUP_clipThenScale_20251020_210310.ipynb")
