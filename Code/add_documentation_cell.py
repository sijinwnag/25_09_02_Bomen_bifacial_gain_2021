"""
Add workflow documentation cell to SunSolve notebook.
Inserts a markdown cell before the hourly optimization section (Cell 12).
"""

import json
from pathlib import Path

# Paths
notebook_path = Path(__file__).parent / "25_10_08_Data_visualiser_matching_inv_SunSolve.ipynb"

print(f"Loading notebook: {notebook_path}")

# Load notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

print(f"Original notebook has {len(notebook['cells'])} cells\n")

# Documentation content
documentation_markdown = """### SCALE-THEN-CLIP WORKFLOW

**⚠️ IMPORTANT METHODOLOGY UPDATE**

This notebook implements the **scale-then-clip workflow** for inverter power clipping analysis:

#### What Changed:
- **Previous approach (clip-then-scale)**: `scale × Σ(clip(hourly))` - Clipped hourly power BEFORE aggregation
- **New approach (scale-then-clip)**: `Σ(clip(scale × hourly))` - Scales hourly power FIRST, then clips, then aggregates

#### Why This Matters:
The scale-then-clip workflow is **physically more accurate** because:
1. **Realistic inverter behavior**: Inverters saturate AFTER seeing the full solar input (scaled)
2. **Non-linear clipping**: Clipping is non-commutative with scaling and aggregation
3. **Proper loss accounting**: Energy losses from inverter saturation occur at instantaneous power levels

#### Implementation Details:
- **Clipping threshold**: 2.392 MW (inverter AC capacity for INV 2-1)
- **Optimization**: Binary search finds optimal scaling factor achieving MBE ≈ 0
- **Workflow**: For each trial factor → scale hourly → clip hourly → aggregate to daily → calculate MBE
- **Sections modified**:
  - Cell 13 (Hourly filtered & scaled)
  - Cell 17 (Daily filtered & scaled)
  - Cell 21 (Monthly optimization) uses daily aggregates from above sections

#### Expected Impact:
- Slightly different optimal scaling factors compared to clip-then-scale
- More physically accurate representation of inverter clipping losses
- Better agreement with measured inverter behavior during high irradiance periods

#### References:
- See `batch_pvsyst_evaluation_scaled_clip.py` for batch processing implementation
- Compare results with clip-then-scale using `compare_results.py`
"""

# Create new markdown cell
new_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [line + '\n' for line in documentation_markdown.split('\n')]
}

# Insert before Cell 12 (which contains hourly optimization)
INSERT_POSITION = 12

print(f"Inserting documentation cell at position {INSERT_POSITION}")
print(f"(Current Cell {INSERT_POSITION} will become Cell {INSERT_POSITION + 1})")

# Insert the new cell
notebook['cells'].insert(INSERT_POSITION, new_cell)

print(f"\nNotebook now has {len(notebook['cells'])} cells")

# Save modified notebook
print(f"\nSaving modified notebook...")
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"[OK] Documentation cell added successfully")
print(f"\nNotebook saved: {notebook_path}")
print("\nNext steps:")
print("1. Open notebook in Jupyter")
print("2. Navigate to new documentation cell (before hourly optimization)")
print("3. Verify formatting and content")
print("4. Run cells to test scale-then-clip workflow")
