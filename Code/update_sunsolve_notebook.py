"""
Script to update SunSolve notebook from clip-then-scale to scale-then-clip workflow.
"""

import json
import sys
from pathlib import Path

# File paths
notebook_path = Path(__file__).parent / "25_10_08_Data_visualiser_matching_inv_SunSolve.ipynb"

print(f"Loading notebook: {notebook_path}")

# Load the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

print(f"Notebook loaded successfully")
print(f"Total cells: {len(notebook['cells'])}")

# Find cells by searching for the characteristic code
cell_22_idx = None  # Hourly filtered & scaled
cell_24_idx = None  # Daily filtered & scaled

found_cells = []

for idx, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])

        # Look for cells with clipping configuration
        if 'apply_clipping = True' in source and 'clipping_threshold = 2.392' in source:
            # Check if it's hourly or daily section
            if 'Hourly filtered & scaled' in source or 'hourly_actual_power' in source:
                if cell_22_idx is None:  # First occurrence
                    cell_22_idx = idx
                    print(f"Found Cell 22 (Hourly section) at index {idx}")
                    found_cells.append(('Cell 22 - Hourly', idx))
            elif 'Daily filtered & scaled' in source or 'daily_actual_energy' in source:
                if cell_24_idx is None:  # First occurrence
                    cell_24_idx = idx
                    print(f"Found Cell 24 (Daily section) at index {idx}")
                    found_cells.append(('Cell 24 - Daily', idx))
            elif cell_22_idx is None and cell_24_idx is None:
                # Generic clipping cell - could be either
                print(f"Found clipping configuration at index {idx}")
                found_cells.append(('Unknown clipping cell', idx))

if cell_22_idx is None:
    print("WARNING: Could not find Cell 22")
if cell_24_idx is None:
    print("WARNING: Could not find Cell 24")

print(f"\n=== ALL CELLS WITH CLIPPING ===")
for name, idx in found_cells:
    print(f"\n{name} (index {idx}):")
    source = ''.join(notebook['cells'][idx]['source'])
    # Print first 500 chars and check for key markers
    print(source[:500])
    print("...")
    if 'hourly_actual_power' in source:
        print("  -> Contains 'hourly_actual_power' (HOURLY section)")
    if 'daily_actual_energy' in source:
        print("  -> Contains 'daily_actual_energy' (DAILY section)")
    if 'Apply clipping to raw power' in source:
        print("  -> Contains clipping application code")

print("\n=== SUMMARY ===")
print(f"Cell 22 (Hourly): {'Found' if cell_22_idx is not None else 'NOT FOUND'}")
print(f"Cell 24 (Daily): {'Found' if cell_24_idx is not None else 'NOT FOUND'}")
