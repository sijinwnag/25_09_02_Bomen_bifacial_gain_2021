"""
Synchronize plotting workflow between PVsyst and SunSolve notebooks.
This script ensures both notebooks use identical plotting code while
maintaining separate data sources.
"""

import json
import copy
from pathlib import Path

# File paths
pvsyst_path = Path("25_09_05_Data_visualiser_matching_inv.ipynb")
sunsolve_path = Path("25_10_08_Data_visualiser_matching_inv_SunSolve.ipynb")

print("="*80)
print("SYNCHRONIZING PLOTTING WORKFLOW")
print("="*80)

# Load both notebooks
print("\n1. Loading notebooks...")
with open(pvsyst_path, 'r', encoding='utf-8') as f:
    pvsyst_nb = json.load(f)

with open(sunsolve_path, 'r', encoding='utf-8') as f:
    sunsolve_nb = json.load(f)

print(f"   PVsyst: {len(pvsyst_nb['cells'])} cells")
print(f"   SunSolve: {len(sunsolve_nb['cells'])} cells")

# Extract SunSolve cells 19-24 (plotting section, excluding cell 18 which PVsyst already has)
print("\n2. Extracting SunSolve plotting cells (19-24)...")
sunsolve_plotting_cells = []
for idx in range(19, 25):  # Cells 19-24 (inclusive)
    cell = copy.deepcopy(sunsolve_nb['cells'][idx])
    sunsolve_plotting_cells.append(cell)
    cell_type = cell['cell_type']
    source_preview = ''.join(cell['source'])[:50]
    print(f"   Cell {idx} ({cell_type}): {source_preview}...")

# Adapt cells for PVsyst
print("\n3. Adapting cells for PVsyst...")
adapted_cells = []

for idx, cell in enumerate(sunsolve_plotting_cells):
    adapted_cell = copy.deepcopy(cell)

    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])

        # Replace column references
        source = source.replace("'Power_MW'", "'EArray'")
        source = source.replace('"Power_MW"', '"EArray"')
        source = source.replace('simulation_results_df["Power_MW"]', 'simulation_results_df["EArray"]')
        source = source.replace("simulation_results_df['Power_MW']", "simulation_results_df['EArray']")

        # Replace labels
        source = source.replace('sunsolve yield', 'PVsyst')
        source = source.replace('SunSolve Yield', 'PVsyst')
        source = source.replace('sunsolve', 'PVsyst')

        # Update source
        adapted_cell['source'] = [source]
        print(f"   Adapted cell {idx} (code cell, {len(source)} chars)")
    else:
        # Markdown cell - keep as is
        print(f"   Kept cell {idx} (markdown cell)")

    adapted_cells.append(adapted_cell)

# Add PIPELINE_VERSION to the first code cell (yearly plot - will be Cell 20)
print("\n4. Adding PIPELINE_VERSION identifier...")
for idx, cell in enumerate(adapted_cells):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        # Add version identifier at the beginning
        version_code = "PIPELINE_VERSION = 'scale→clip→resample_v1'\nprint(f'Pipeline Version: {PIPELINE_VERSION}')\n\n"
        cell['source'] = [version_code + source]
        print(f"   Added to cell {idx} (will become Cell 20 in PVsyst)")
        break

# Build new PVsyst notebook
print("\n5. Rebuilding PVsyst notebook structure...")
new_pvsyst_cells = []

# Keep cells 0-18 (all preprocessing and Cell 18 header)
for idx in range(19):
    new_pvsyst_cells.append(copy.deepcopy(pvsyst_nb['cells'][idx]))

print(f"   Kept cells 0-18 (preprocessing)")

# Insert adapted cells 19-24
new_pvsyst_cells.extend(adapted_cells)
print(f"   Inserted cells 19-24 (synchronized plotting)")

# Add remaining cells starting from old Cell 23 ("# 4. Param optimisation")
# Note: We're replacing old cells 19-22, so old cell 23 becomes new cell 25
for idx in range(23, len(pvsyst_nb['cells'])):
    new_pvsyst_cells.append(copy.deepcopy(pvsyst_nb['cells'][idx]))

print(f"   Added cells 25+ (Param optimisation onwards)")
print(f"   Total cells in new notebook: {len(new_pvsyst_cells)}")

# Update notebook cells
pvsyst_nb['cells'] = new_pvsyst_cells

# Save modified notebook
print("\n6. Saving synchronized notebook...")
with open(pvsyst_path, 'w', encoding='utf-8') as f:
    json.dump(pvsyst_nb, f, indent=1, ensure_ascii=False)

print(f"   Saved to: {pvsyst_path}")

# Validation
print("\n7. Validation...")
print(f"   Total cells: {len(new_pvsyst_cells)}")
print("\n   New structure:")
for idx in range(18, min(26, len(new_pvsyst_cells))):
    cell = new_pvsyst_cells[idx]
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source']).split('\n')[0][:80]
        print(f"   Cell {idx} (markdown): {source}")
    else:
        source_preview = ''.join(cell['source'])[:60].encode('ascii', 'replace').decode('ascii')
        print(f"   Cell {idx} (code): {source_preview}...")

print("\n" + "="*80)
print("SYNCHRONIZATION COMPLETE!")
print("="*80)
print("\nNext steps:")
print("1. Open the notebook in Jupyter")
print("2. Run cells 18-24 to verify plotting works")
print("3. Compare plots with SunSolve notebook")
print("\nBackup available at:")
print("   25_09_05_Data_visualiser_matching_inv_BACKUP_beforeSync_*.ipynb")
