"""
Apply scale-then-clip workflow modifications to SunSolve notebook.
This script modifies cells 12 (hourly) and 16 (daily) to implement the scale-then-clip workflow.
"""

import json
import re
from pathlib import Path

# File paths
notebook_path = Path(__file__).parent / "25_10_08_Data_visualiser_matching_inv_SunSolve.ipynb"

print(f"Loading notebook: {notebook_path}")

# Load the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

print(f"Notebook loaded - Total cells: {len(notebook['cells'])}\n")

# Cell indices
CELL_22_IDX = 12  # Hourly filtered & scaled
CELL_24_IDX = 16  # Daily filtered & scaled

def modify_hourly_cell(cell_source_list):
    """
    Modify Cell 22 (hourly section) to implement scale-then-clip workflow.

    Args:
        cell_source_list: List of source code lines from the notebook cell

    Returns:
        Modified list of source code lines
    """
    source = ''.join(cell_source_list)

    # Step 1: Comment out the clipping application (keep configuration)
    # Find: if apply_clipping:... simulation_results_df['Power_MW'] = simulation_results_df['Power_MW'].clip(upper=clipping_threshold)

    # Pattern to find the clipping block
    clipping_pattern = r"# Apply clipping to raw power values\nif apply_clipping:\n    original_max = simulation_results_df\['Power_MW'\]\.max\(\)\n    clipped_count = \(simulation_results_df\['Power_MW'\] > clipping_threshold\)\.sum\(\)\n    simulation_results_df\['Power_MW'\] = simulation_results_df\['Power_MW'\]\.clip\(upper=clipping_threshold\)\n    print\(f\"Applied \{clipping_threshold\} MW clipping to power values: \{clipped_count\} values clipped\"\)\n    print\(f\"Maximum power reduced from \{original_max:.2f\} MW to \{simulation_results_df\['Power_MW'\]\.max\(\):.2f\} MW\"\)"

    replacement = """# SCALE-THEN-CLIP WORKFLOW: Clipping now applied DURING optimization, not before
# Keep raw hourly power values for scale-then-clip optimization
# Commenting out pre-optimization clipping:
# if apply_clipping:
#     original_max = simulation_results_df['Power_MW'].max()
#     clipped_count = (simulation_results_df['Power_MW'] > clipping_threshold).sum()
#     simulation_results_df['Power_MW'] = simulation_results_df['Power_MW'].clip(upper=clipping_threshold)
#     print(f"Applied {clipping_threshold} MW clipping to power values: {clipped_count} values clipped")
#     print(f"Maximum power reduced from {original_max:.2f} MW to {simulation_results_df['Power_MW'].max():.2f} MW")
print(f"Raw power data preserved for scale-then-clip optimization")
print(f"Maximum raw power: {simulation_results_df['Power_MW'].max():.2f} MW")"""

    # Try simple string replacement first
    if "# Apply clipping to raw power values\nif apply_clipping:" in source:
        # Find and comment out the clipping block
        lines = source.split('\n')
        modified_lines = []
        in_clipping_block = False
        skip_count = 0

        for i, line in enumerate(lines):
            if "# Apply clipping to raw power values" in line:
                # Insert new code
                modified_lines.append("# SCALE-THEN-CLIP WORKFLOW: Clipping now applied DURING optimization, not before")
                modified_lines.append("# Keep raw hourly power values for scale-then-clip optimization")
                modified_lines.append("# Commenting out pre-optimization clipping:")
                in_clipping_block = True
                continue
            elif in_clipping_block:
                if line.strip().startswith("if apply_clipping:") or \
                   line.strip().startswith("original_max =") or \
                   line.strip().startswith("clipped_count =") or \
                   line.strip().startswith("simulation_results_df['Power_MW'] = simulation_results_df['Power_MW'].clip") or \
                   "Applied {clipping_threshold} MW clipping" in line or \
                   "Maximum power reduced from" in line:
                    # Comment out this line
                    modified_lines.append("# " + line if not line.strip().startswith("#") else line)
                    if "Maximum power reduced from" in line:
                        in_clipping_block = False
                        # Add new print statements
                        modified_lines.append('print(f"Raw power data preserved for scale-then-clip optimization")')
                        modified_lines.append('print(f"Maximum raw power: {simulation_results_df[\'Power_MW\'].max():.2f} MW")')
                else:
                    modified_lines.append(line)
            else:
                modified_lines.append(line)

        source = '\n'.join(modified_lines)

    # Step 2: Add scale-then-clip helper function before the optimization section
    # Find where to insert (look for "# Function to calculate MBE" or similar)
    if "def calculate_mbe(scale_factor" in source:
        # Insert before the calculate_mbe function
        helper_function = '''
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
            pass  # Silently clip during optimization
    else:
        clipped_hourly = scaled_hourly

    return clipped_hourly

'''
        source = source.replace(
            "# Function to calculate MBE",
            helper_function + "# Function to calculate MBE (UPDATED for scale-then-clip)"
        )

    # Step 3: Modify the calculate_mbe function
    # Replace the existing calculate_mbe function with scale-then-clip version
    old_mbe_pattern = r'''def calculate_mbe\(scale_factor, apply_clip=False, clip_threshold=2\.392\):
    """Calculate Mean Bias Error for given scale factor"""
    # Scale the simulated data \(clipping already applied at power level\)
    scaled_data = metrics_df\['Simulated'\] \* scale_factor

    # Calculate MBE
    return np\.mean\(scaled_data - metrics_df\['Actual'\]\)'''

    new_mbe_function = '''def calculate_mbe_scaleThenClip(scale_factor):
    """
    Calculate MBE using scale-then-clip workflow.
    Recomputes hourly scaling and clipping for each trial factor.
    """
    # Apply scale-then-clip to hourly data
    hourly_scaled_clipped = compute_hourly_scaled_clipped(
        simulation_results_df['Power_MW'],
        scale_factor
    )

    # Match timestamps with actual data
    common_times = hourly_actual_power.index.intersection(hourly_scaled_clipped.index)
    actual_matched = hourly_actual_power.loc[common_times]
    simulated_matched = hourly_scaled_clipped.loc[common_times]

    # Calculate MBE
    mbe = np.mean(simulated_matched - actual_matched)
    return mbe'''

    source = re.sub(old_mbe_pattern, new_mbe_function, source)

    # Step 4: Update binary search to use new function
    source = source.replace(
        "mbe = calculate_mbe(mid_factor, apply_clipping, clipping_threshold)",
        "mbe = calculate_mbe_scaleThenClip(mid_factor)"
    )

    # Step 5: Update Simulated_display creation
    old_display_pattern = r'''# Apply scale factor to filtered data \(clipping already applied at power level\)
    metrics_df\['Simulated_scaled'\] = metrics_df\['Simulated'\] \* sunsolve_yield_scaling

    # No need to apply clipping again since it was done at the power level
    metrics_df\['Simulated_display'\] = metrics_df\['Simulated_scaled'\]'''

    new_display_code = '''# Apply optimal scale factor using scale-then-clip workflow
    print(f"\\nApplying scale-then-clip with optimal factor {sunsolve_yield_scaling:.10f}...")
    final_hourly_scaled_clipped = compute_hourly_scaled_clipped(
        simulation_results_df['Power_MW'],
        sunsolve_yield_scaling
    )

    # Update metrics_df with scale-then-clip results
    # Match timestamps and use the scaled-clipped hourly data
    metrics_df['Simulated_display'] = final_hourly_scaled_clipped.reindex(metrics_df.index)'''

    source = re.sub(old_display_pattern, new_display_code, source)

    # Convert back to list of lines for notebook format
    return [line + '\n' for line in source.split('\n')]


def modify_daily_cell(cell_source_list):
    """
    Modify Cell 24 (daily section) to implement scale-then-clip workflow.

    Args:
        cell_source_list: List of source code lines from the notebook cell

    Returns:
        Modified list of source code lines
    """
    source = ''.join(cell_source_list)

    # Similar modifications as hourly cell but for daily aggregation
    # Step 1: Comment out clipping
    if "# Apply clipping to raw power values" in source:
        lines = source.split('\n')
        modified_lines = []
        in_clipping_block = False

        for i, line in enumerate(lines):
            if "# Apply clipping to raw power values" in line:
                modified_lines.append("# SCALE-THEN-CLIP WORKFLOW: Clipping now applied DURING optimization, not before")
                modified_lines.append("# Keep raw hourly power values for scale-then-clip optimization")
                modified_lines.append("# Commenting out pre-optimization clipping:")
                in_clipping_block = True
                continue
            elif in_clipping_block:
                if line.strip().startswith("if apply_clipping:") or \
                   line.strip().startswith("original_max =") or \
                   line.strip().startswith("clipped_count =") or \
                   line.strip().startswith("simulation_results_df['Power_MW'] = simulation_results_df['Power_MW'].clip") or \
                   "Applied {clipping_threshold} MW clipping" in line or \
                   "Maximum power reduced from" in line:
                    modified_lines.append("# " + line if not line.strip().startswith("#") else line)
                    if "Maximum power reduced from" in line:
                        in_clipping_block = False
                        modified_lines.append('print(f"Raw power data preserved for scale-then-clip optimization")')
                        modified_lines.append('print(f"Maximum raw power: {simulation_results_df[\'Power_MW\'].max():.2f} MW")')
                else:
                    modified_lines.append(line)
            else:
                modified_lines.append(line)

        source = '\n'.join(modified_lines)

    # Step 2: Add daily energy computation function
    if "def calculate_mbe(scale_factor" in source:
        helper_function = '''
# ============================================================================
# SCALE-THEN-CLIP WORKFLOW FOR DAILY DATA
# ============================================================================

def compute_daily_energy_scaleThenClip(hourly_power_df, scale_factor):
    """
    Compute daily energy totals using scale-then-clip workflow.

    Args:
        hourly_power_df: DataFrame with hourly power (MW)
        scale_factor: Scaling factor to apply

    Returns:
        Daily energy totals (MWh/day) after scale-then-clip
    """
    # Step 1: SCALE hourly power
    scaled_hourly = hourly_power_df['Power_MW'] * scale_factor

    # Step 2: CLIP scaled hourly power (if enabled)
    if apply_clipping and clipping_threshold is not None:
        clipped_hourly = scaled_hourly.clip(upper=clipping_threshold)
    else:
        clipped_hourly = scaled_hourly

    # Step 3: Convert to energy (MW * 1 hour = MWh)
    hourly_energy = clipped_hourly * 1.0  # 1 hour interval

    # Step 4: Aggregate to daily totals
    daily_energy = hourly_energy.resample('D').sum()

    return daily_energy

'''
        source = source.replace(
            "# Function to calculate MBE",
            helper_function + "# Function to calculate MBE (UPDATED for scale-then-clip)"
        )

    # Step 3: Modify calculate_mbe for daily
    old_mbe_pattern = r'''def calculate_mbe\(scale_factor, apply_clip=False, clip_threshold=2400?\):
    """Calculate Mean Bias Error for given scale factor"""
    # Scale the simulated data \(clipping already applied at power level\)
    scaled_data = metrics_df\['Simulated'\] \* scale_factor

    # Calculate MBE
    return np\.mean\(scaled_data - metrics_df\['Actual'\]\)'''

    new_mbe_function = '''def calculate_mbe_daily_scaleThenClip(scale_factor):
    """Calculate MBE for daily data using scale-then-clip workflow"""
    # Recompute daily energy with scale-then-clip
    daily_simulated = compute_daily_energy_scaleThenClip(
        simulation_results_df,
        scale_factor
    )

    # Match dates with actual daily energy
    common_dates = daily_actual_energy.index.intersection(daily_simulated.index)
    actual_matched = daily_actual_energy.loc[common_dates]
    simulated_matched = daily_simulated.loc[common_dates]

    # Apply maintenance filter
    if 'maintenance_dates' in locals() and maintenance_dates is not None:
        mask = common_dates.normalize().isin(maintenance_dates.normalize())
        actual_matched = actual_matched[mask]
        simulated_matched = simulated_matched[mask]

    # Calculate MBE
    return np.mean(simulated_matched - actual_matched)'''

    source = re.sub(old_mbe_pattern, new_mbe_function, source)

    # Step 4: Update binary search calls
    source = source.replace(
        "mbe = calculate_mbe(mid_factor, apply_clipping, clipping_threshold)",
        "mbe = calculate_mbe_daily_scaleThenClip(mid_factor)"
    )

    # Step 5: Update Simulated_display for daily
    old_display_pattern = r'''# Apply scale factor to filtered data \(clipping already applied at power level\)
    metrics_df\['Simulated_scaled'\] = metrics_df\['Simulated'\] \* sunsolve_yield_scaling

    # No need to apply clipping again since it was done at the power level
    metrics_df\['Simulated_display'\] = metrics_df\['Simulated_scaled'\]'''

    new_display_code = '''# Recompute final daily energy using scale-then-clip workflow
    print(f"\\nApplying scale-then-clip with optimal factor {sunsolve_yield_scaling:.10f}...")
    final_daily_energy = compute_daily_energy_scaleThenClip(
        simulation_results_df,
        sunsolve_yield_scaling
    )
    metrics_df['Simulated_display'] = final_daily_energy.reindex(metrics_df.index)'''

    source = re.sub(old_display_pattern, new_display_code, source)

    # Convert back to list
    return [line + '\n' for line in source.split('\n')]


# Apply modifications
print("=" * 80)
print("MODIFYING CELL 22 (Hourly Filtered & Scaled)")
print("=" * 80)

try:
    modified_cell_22 = modify_hourly_cell(notebook['cells'][CELL_22_IDX]['source'])
    notebook['cells'][CELL_22_IDX]['source'] = modified_cell_22
    print("[OK] Cell 22 modified successfully")
except Exception as e:
    print(f"[ERROR] Error modifying Cell 22: {e}")

print("\n" + "=" * 80)
print("MODIFYING CELL 24 (Daily Filtered & Scaled)")
print("=" * 80)

try:
    modified_cell_24 = modify_daily_cell(notebook['cells'][CELL_24_IDX]['source'])
    notebook['cells'][CELL_24_IDX]['source'] = modified_cell_24
    print("[OK] Cell 24 modified successfully")
except Exception as e:
    print(f"[ERROR] Error modifying Cell 24: {e}")

# Save modified notebook
print("\n" + "=" * 80)
print("SAVING MODIFIED NOTEBOOK")
print("=" * 80)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"[OK] Notebook saved: {notebook_path}")
print("\nModifications complete!")
print("\nNext steps:")
print("1. Open notebook in Jupyter")
print("2. Run cells to verify scale-then-clip workflow")
print("3. Check that MBE ≈ 0 and results are reasonable")
