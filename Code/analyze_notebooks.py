"""
Analyze notebook structure to identify plotting sections and data processing workflow
"""

import json
from pathlib import Path
import sys

# Set UTF-8 encoding for stdout
sys.stdout.reconfigure(encoding='utf-8')

# Paths to notebooks
pvsyst_nb = Path(r"C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\25_09_02_Bomen_bifacial_gain_2021\Code\25_09_05_Data_visualiser_matching_inv.ipynb")
sunsolve_nb = Path(r"C:\Users\z5183876\OneDrive - UNSW\Documents\GitHub\25_09_02_Bomen_bifacial_gain_2021\Code\25_10_08_Data_visualiser_matching_inv_SunSolve.ipynb")

def analyze_notebook(nb_path, name):
    """Analyze notebook structure and identify key sections"""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {name}")
    print(f"{'='*80}\n")

    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']
    print(f"Total cells: {len(cells)}")

    # Find section headers and cell numbers
    print("\n--- SECTION HEADERS ---")
    for idx, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])
            # Look for headers (lines starting with #)
            for line in source.split('\n'):
                if line.strip().startswith('#'):
                    print(f"Cell {idx}: {line.strip()}")

    # Find cells with plotting code
    print("\n--- CELLS WITH PLOTTING (plt.) ---")
    for idx, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'plt.' in source:
                # Get first few lines
                lines = source.split('\n')[:5]
                preview = '\n'.join(lines)
                print(f"\nCell {idx}:")
                print(preview[:200] + "...")

    # Find "4. Param optimisation" section
    print("\n--- SECTION: 4. Param optimisation ---")
    for idx, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])
            if '4. Param optimisation' in source:
                print(f"Found at Cell {idx}")
                print(f"Source: {source[:200]}")
                break

    # Find cells starting from 18
    print("\n--- CELLS 18 AND ABOVE ---")
    for idx in range(18, min(25, len(cells))):
        cell = cells[idx]
        print(f"\nCell {idx} ({cell['cell_type']}):")
        source = ''.join(cell['source'])
        print(source[:150] + "..." if len(source) > 150 else source)

# Analyze both notebooks
analyze_notebook(pvsyst_nb, "PVsyst Notebook")
analyze_notebook(sunsolve_nb, "SunSolve Notebook")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
