#!/usr/bin/env python3
"""
Rebuild SunSolve notebook with proper structure preservation.
Key: Maintain list-of-strings format for notebook cells.
"""

import json
import re

def create_sunsolve_loading_cell():
    """Create SunSolve data loading cell as proper list of strings."""
    lines = [
        "# Define SunSolve Yield CSV file path - user can modify this directly\n",
        'sunsolve_file = r"C:\\Users\\z5183876\\OneDrive - UNSW\\Documents\\GitHub\\25_09_02_Bomen_bifacial_gain_2021\\Data\\SunSolve Yield\\Per inverter\\2-1\\0_mimic_PVsyst\\25_10_07.csv"\n',
        "\n",
        'print(f"Loading SunSolve Yield CSV file: {sunsolve_file}")\n',
        "\n",
        "try:\n",
        "    # Load SunSolve CSV with simple read_csv (no special parsing needed)\n",
        "    simulation_results_df = pd.read_csv(sunsolve_file)\n",
        "    \n",
        '    print(f"Raw DataFrame loaded with shape: {simulation_results_df.shape}")\n',
        '    print(f"Columns: {list(simulation_results_df.columns[:10])}... (showing first 10)")\n',
        "    \n",
        "    # Construct timestamp from 'Day of year', 'Hour', 'Minute' columns\n",
        "    year = 2021\n",
        '    base_date = pd.Timestamp(f"{year}-01-01")\n',
        "    \n",
        '    print("\\nConstructing timestamps from Day of year, Hour, Minute columns...")\n',
        "    simulation_results_df['timestamp'] = simulation_results_df.apply(\n",
        "        lambda row: base_date + pd.Timedelta(days=int(row['Day of year'])-1) + \n",
        "                    pd.Timedelta(hours=int(row['Hour'])) + \n",
        "                    pd.Timedelta(minutes=int(row['Minute'])),\n",
        "        axis=1\n",
        "    )\n",
        "    \n",
        "    # Convert power from W to MW\n",
        '    print("Converting power from W to MW...")\n',
        "    if 'Power [unit-system] (W)' in simulation_results_df.columns:\n",
        "        simulation_results_df['Power_MW'] = simulation_results_df['Power [unit-system] (W)'] / 1e6\n",
        "    else:\n",
        '        raise KeyError("Column \'Power [unit-system] (W)\' not found in SunSolve data")\n',
        "    \n",
        "    # Sort by timestamp to ensure chronological order\n",
        "    simulation_results_df = simulation_results_df.sort_values('timestamp').reset_index(drop=True)\n",
        "    \n",
        "    # Remove duplicate timestamps if any\n",
        "    duplicates = simulation_results_df.duplicated(subset=['timestamp']).sum()\n",
        "    if duplicates > 0:\n",
        '        print(f"Found and removing {duplicates} duplicate timestamps")\n',
        "        simulation_results_df = simulation_results_df.drop_duplicates(subset=['timestamp'], keep='first')\n",
        "    \n",
        "    # Data validation and timestamp analysis\n",
        '    print("\\n=== TIMESTAMP ANALYSIS ===")\n',
        '    print(f"Total timestamps: {len(simulation_results_df)}")\n',
        '    print(f"Date range: {simulation_results_df[\'timestamp\'].min()} to {simulation_results_df[\'timestamp\'].max()}")\n',
        "    \n",
        "    # Check timestamp continuity\n",
        "    time_diffs = simulation_results_df['timestamp'].diff()[1:]  # Skip first NaT\n",
        "    common_diff = time_diffs.value_counts().index[0]\n",
        '    print(f"Most common time difference: {common_diff}")\n',
        "    \n",
        "    irregular_intervals = (time_diffs != common_diff).sum()\n",
        "    if irregular_intervals > 0:\n",
        '        print(f"Warning: Found {irregular_intervals} irregular time intervals")\n',
        "    \n",
        "    # Check distribution of timestamps by month\n",
        "    month_counts = simulation_results_df['timestamp'].dt.month.value_counts().sort_index()\n",
        '    print("\\nTimestamps per month:")\n',
        "    for month, count in month_counts.items():\n",
        '        print(f"Month {month}: {count} timestamps")\n',
        "    \n",
        "    # Print available columns\n",
        '    print("\\nAvailable columns:")\n',
        '    print(simulation_results_df.columns.tolist()[:20], "... (showing first 20)")\n',
        "    \n",
        "    # Check for Power_MW column\n",
        '    print(f"\\nDoes \'Power_MW\' column exist? {\'Power_MW\' in simulation_results_df.columns}")\n',
        "    if 'Power_MW' in simulation_results_df.columns:\n",
        '        print(f"Power_MW statistics:\\n{simulation_results_df[\'Power_MW\'].describe()}")\n',
        "        \n",
        "except Exception as e:\n",
        '    print(f"Error loading SunSolve Yield CSV: {e}")\n',
        "    raise\n"
    ]
    return lines

def replace_earray_in_line(line):
    """Replace EArray references with Power_MW in a single line."""
    # Pattern matching for various EArray references
    line = re.sub(r"simulation_results_df\['EArray'\]", "simulation_results_df['Power_MW']", line)
    line = re.sub(r'simulation_results_df\["EArray"\]', 'simulation_results_df["Power_MW"]', line)
    line = re.sub(r"simulation_results_df\.EArray", "simulation_results_df.Power_MW", line)
    line = re.sub(r"'EArray'", "'Power_MW'", line)
    line = re.sub(r'"EArray"', '"Power_MW"', line)
    return line

def replace_pvsyst_in_markdown(lines):
    """Replace PVsyst with SunSolve Yield in markdown cells."""
    new_lines = []
    for line in lines:
        # Replace PVsyst references in markdown
        line = line.replace('PVsyst', 'SunSolve Yield')
        line = line.replace('pvsyst', 'sunsolve')
        new_lines.append(line)
    return new_lines

def rebuild_notebook():
    """Main function to rebuild notebook properly."""
    print("Reading source notebook...")
    with open('25_09_05_Data_visualiser_matching_inv.ipynb', 'r', encoding='utf-8') as f:
        source_nb = json.load(f)

    print("Extracting and modifying cells 0-21...")
    new_cells = []

    for i in range(22):  # Cells 0-21
        source_cell = source_nb['cells'][i].copy()

        if i == 6:
            # Cell 6: Complete replacement with SunSolve loading code
            print(f"  Cell {i}: Replacing with SunSolve data loading")
            source_cell['source'] = create_sunsolve_loading_cell()
            source_cell['outputs'] = []
            source_cell['execution_count'] = None

        elif source_cell.get('cell_type') == 'code':
            # Code cells: Replace EArray references line-by-line
            print(f"  Cell {i}: Applying EArray -> Power_MW replacements")
            source_lines = source_cell['source']
            new_lines = [replace_earray_in_line(line) for line in source_lines]
            source_cell['source'] = new_lines
            source_cell['outputs'] = []
            source_cell['execution_count'] = None

        elif source_cell.get('cell_type') == 'markdown':
            # Markdown cells: Replace PVsyst references
            print(f"  Cell {i}: Applying PVsyst -> SunSolve Yield replacements")
            source_cell['source'] = replace_pvsyst_in_markdown(source_cell['source'])

        new_cells.append(source_cell)

    # Create new notebook
    print("\nCreating new notebook structure...")
    new_notebook = {
        'cells': new_cells,
        'metadata': source_nb.get('metadata', {}),
        'nbformat': source_nb.get('nbformat', 4),
        'nbformat_minor': source_nb.get('nbformat_minor', 5)
    }

    # Save
    output_file = '25_10_08_Data_visualiser_matching_inv_SunSolve.ipynb'
    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_notebook, f, indent=1, ensure_ascii=False)

    print("\n" + "="*70)
    print("REBUILD COMPLETE")
    print("="*70)
    print(f"Output: {output_file}")
    print(f"Total cells: {len(new_cells)}")

    # Verify formatting
    print("\nVerifying cell structure...")
    for i in range(min(3, len(new_cells))):
        cell = new_cells[i]
        source = cell['source']
        if isinstance(source, list) and len(source) > 0:
            has_newlines = any('\n' in line for line in source)
            print(f"  Cell {i}: {len(source)} lines, newlines={'YES' if has_newlines else 'NO'}")
        else:
            print(f"  Cell {i}: Empty or malformed")

    return output_file

if __name__ == '__main__':
    try:
        output_file = rebuild_notebook()
        print(f"\nSUCCESS: Notebook rebuilt at {output_file}")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        raise
