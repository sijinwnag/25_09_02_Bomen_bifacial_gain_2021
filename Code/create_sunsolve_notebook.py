#!/usr/bin/env python3
"""
Script to create SunSolve visualization notebook from PVsyst template.
Extracts cells 0-21 and modifies data loading for SunSolve Yield format.
"""

import json
import re
from datetime import datetime

def modify_data_loading_cell():
    """Create the modified data loading cell for SunSolve."""
    return '''# Define SunSolve Yield CSV file path - user can modify this directly
sunsolve_file = r"C:\\Users\\z5183876\\OneDrive - UNSW\\Documents\\GitHub\\25_09_02_Bomen_bifacial_gain_2021\\Data\\SunSolve Yield\\Per inverter\\2-1\\0_mimic_PVsyst\\25_10_07.csv"

print(f"Loading SunSolve Yield CSV file: {sunsolve_file}")

try:
    # Load SunSolve CSV with simple read_csv (no special parsing needed)
    simulation_results_df = pd.read_csv(sunsolve_file)

    print(f"Raw DataFrame loaded with shape: {simulation_results_df.shape}")
    print(f"Columns: {list(simulation_results_df.columns[:10])}... (showing first 10)")

    # Construct timestamp from 'Day of year', 'Hour', 'Minute' columns
    year = 2021
    base_date = pd.Timestamp(f"{year}-01-01")

    print("\\nConstructing timestamps from Day of year, Hour, Minute columns...")
    simulation_results_df['timestamp'] = simulation_results_df.apply(
        lambda row: base_date + pd.Timedelta(days=int(row['Day of year'])-1) +
                    pd.Timedelta(hours=int(row['Hour'])) +
                    pd.Timedelta(minutes=int(row['Minute'])),
        axis=1
    )

    # Convert power from W to MW
    print("Converting power from W to MW...")
    if 'Power [unit-system] (W)' in simulation_results_df.columns:
        simulation_results_df['Power_MW'] = simulation_results_df['Power [unit-system] (W)'] / 1e6
    else:
        raise KeyError("Column 'Power [unit-system] (W)' not found in SunSolve data")

    # Sort by timestamp to ensure chronological order
    simulation_results_df = simulation_results_df.sort_values('timestamp').reset_index(drop=True)

    # Remove duplicate timestamps if any
    duplicates = simulation_results_df.duplicated(subset=['timestamp']).sum()
    if duplicates > 0:
        print(f"Found and removing {duplicates} duplicate timestamps")
        simulation_results_df = simulation_results_df.drop_duplicates(subset=['timestamp'], keep='first')

    # Data validation and timestamp analysis
    print("\\n=== TIMESTAMP ANALYSIS ===")
    print(f"Total timestamps: {len(simulation_results_df)}")
    print(f"Date range: {simulation_results_df['timestamp'].min()} to {simulation_results_df['timestamp'].max()}")

    # Check timestamp continuity
    time_diffs = simulation_results_df['timestamp'].diff()[1:]  # Skip first NaT
    common_diff = time_diffs.value_counts().index[0]
    print(f"Most common time difference: {common_diff}")

    irregular_intervals = (time_diffs != common_diff).sum()
    if irregular_intervals > 0:
        print(f"Warning: Found {irregular_intervals} irregular time intervals")

    # Check distribution of timestamps by month
    month_counts = simulation_results_df['timestamp'].dt.month.value_counts().sort_index()
    print("\\nTimestamps per month:")
    for month, count in month_counts.items():
        print(f"Month {month}: {count} timestamps")

    # Print available columns
    print("\\nAvailable columns:")
    print(simulation_results_df.columns.tolist()[:20], "... (showing first 20)")

    # Check for Power_MW column
    print(f"\\nDoes 'Power_MW' column exist? {'Power_MW' in simulation_results_df.columns}")
    if 'Power_MW' in simulation_results_df.columns:
        print(f"Power_MW statistics:\\n{simulation_results_df['Power_MW'].describe()}")

except Exception as e:
    print(f"Error loading SunSolve Yield CSV: {e}")
    raise
'''

def replace_pvsyst_references(cell_source):
    """Replace PVsyst-specific references with SunSolve equivalents."""
    if isinstance(cell_source, list):
        source_text = ''.join(cell_source)
    else:
        source_text = cell_source

    # Replace column names
    source_text = source_text.replace("['EArray']", "['Power_MW']")
    source_text = source_text.replace('["EArray"]', '["Power_MW"]')
    source_text = source_text.replace("simulation_results_df.EArray", "simulation_results_df.Power_MW")
    source_text = source_text.replace("'EArray'", "'Power_MW'")
    source_text = source_text.replace('"EArray"', '"Power_MW"')

    # Replace plot labels and titles
    source_text = source_text.replace("PVsyst", "SunSolve Yield")
    source_text = source_text.replace("pvsyst", "sunsolve")

    # Replace comments
    source_text = source_text.replace("# Skip metadata (0-9) and units row (11)", "# SunSolve format - simple CSV")

    # Convert back to list format for notebook
    return source_text.split('\n') if '\n' in source_text else [source_text]

def create_sunsolve_notebook():
    """Main function to create the SunSolve notebook."""
    print("Reading source notebook...")
    with open('25_09_05_Data_visualiser_matching_inv.ipynb', 'r', encoding='utf-8') as f:
        source_nb = json.load(f)

    # Extract cells 0-21 (22 cells total)
    print("Extracting cells 0-21...")
    new_cells = []

    for i, cell in enumerate(source_nb['cells'][:22]):
        if i == 6:  # Replace data loading cell
            print(f"Modifying cell {i} (data loading) for SunSolve...")
            modified_cell = cell.copy()
            modified_cell['source'] = modify_data_loading_cell()
            # Clear outputs
            if 'outputs' in modified_cell:
                modified_cell['outputs'] = []
            if 'execution_count' in modified_cell:
                modified_cell['execution_count'] = None
            new_cells.append(modified_cell)
        else:
            # Copy cell and replace PVsyst references
            modified_cell = cell.copy()
            if modified_cell.get('cell_type') == 'code':
                modified_cell['source'] = replace_pvsyst_references(modified_cell['source'])
                # Clear outputs
                if 'outputs' in modified_cell:
                    modified_cell['outputs'] = []
                if 'execution_count' in modified_cell:
                    modified_cell['execution_count'] = None
            elif modified_cell.get('cell_type') == 'markdown':
                modified_cell['source'] = replace_pvsyst_references(modified_cell['source'])

            new_cells.append(modified_cell)

    # Create new notebook structure
    print("Creating new notebook structure...")
    new_notebook = {
        'cells': new_cells,
        'metadata': source_nb.get('metadata', {}),
        'nbformat': source_nb.get('nbformat', 4),
        'nbformat_minor': source_nb.get('nbformat_minor', 5)
    }

    # Add creation metadata
    if 'metadata' not in new_notebook:
        new_notebook['metadata'] = {}

    # Save new notebook
    output_file = '25_10_08_Data_visualiser_matching_inv_SunSolve.ipynb'
    print(f"Writing new notebook to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_notebook, f, indent=1, ensure_ascii=False)

    print(f"\\n✅ Successfully created {output_file}")
    print(f"   Total cells: {len(new_cells)}")
    print(f"   Modified data loading: Cell 6")
    print(f"   Replaced PVsyst references throughout")

    return output_file

if __name__ == '__main__':
    try:
        output_file = create_sunsolve_notebook()
        print(f"\\n🎉 Notebook creation complete: {output_file}")
    except Exception as e:
        print(f"\\n❌ Error creating notebook: {e}")
        raise
