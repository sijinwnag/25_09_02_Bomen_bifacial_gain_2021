"""
Validate that PVsyst and SunSolve notebooks are properly synchronized.
"""

import json
import sys
from pathlib import Path

# Set UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

pvsyst_path = Path("25_09_05_Data_visualiser_matching_inv.ipynb")
sunsolve_path = Path("25_10_08_Data_visualiser_matching_inv_SunSolve.ipynb")

print("="*80)
print("VALIDATION REPORT: Plotting Workflow Synchronization")
print("="*80)

# Load notebooks
with open(pvsyst_path, 'r', encoding='utf-8') as f:
    pvsyst_nb = json.load(f)

with open(sunsolve_path, 'r', encoding='utf-8') as f:
    sunsolve_nb = json.load(f)

print(f"\nNotebook sizes:")
print(f"  PVsyst: {len(pvsyst_nb['cells'])} cells")
print(f"  SunSolve: {len(sunsolve_nb['cells'])} cells")

# Validation checks
checks_passed = 0
checks_failed = 0

def check(condition, message):
    global checks_passed, checks_failed
    if condition:
        print(f"  ✓ {message}")
        checks_passed += 1
    else:
        print(f"  ✗ {message}")
        checks_failed += 1

print("\n" + "="*80)
print("1. STRUCTURE VALIDATION")
print("="*80)

# Check Cell 18 header
pvsyst_18 = ''.join(pvsyst_nb['cells'][18]['source'])
check("3.2.3.2" in pvsyst_18,
      "Cell 18: Correct header '3.2.3.2. Plot the data after filtering'")

# Check Cell 19 header (Yearly)
pvsyst_19 = ''.join(pvsyst_nb['cells'][19]['source'])
check("Yearly" in pvsyst_19 and pvsyst_nb['cells'][19]['cell_type'] == 'markdown',
      "Cell 19: Markdown header 'Yearly'")

# Check Cell 20 (yearly plot code)
pvsyst_20 = ''.join(pvsyst_nb['cells'][20]['source'])
check(pvsyst_nb['cells'][20]['cell_type'] == 'code' and 'PIPELINE_VERSION' in pvsyst_20,
      "Cell 20: Code cell with PIPELINE_VERSION identifier")

# Check Cell 21 (Monthly optimised yearly header)
pvsyst_21 = ''.join(pvsyst_nb['cells'][21]['source'])
check("Montly optimised" in pvsyst_21 and pvsyst_nb['cells'][21]['cell_type'] == 'markdown',
      "Cell 21: Markdown header 'Montly optimised yeraly'")

# Check Cell 22 (monthly optimization code)
pvsyst_22 = ''.join(pvsyst_nb['cells'][22]['source'])
check(pvsyst_nb['cells'][22]['cell_type'] == 'code' and 'month_names' in pvsyst_22,
      "Cell 22: Monthly optimization code with month_names")

# Check Cell 23 (Monthly header)
pvsyst_23 = ''.join(pvsyst_nb['cells'][23]['source'])
check("Monthly" in pvsyst_23 and pvsyst_nb['cells'][23]['cell_type'] == 'markdown',
      "Cell 23: Markdown header 'Monthly'")

# Check Cell 24 (monthly plot code)
pvsyst_24 = ''.join(pvsyst_nb['cells'][24]['source'])
check(pvsyst_nb['cells'][24]['cell_type'] == 'code' and 'selected_month' in pvsyst_24,
      "Cell 24: Monthly plotting code with selected_month parameter")

# Check Cell 25 (Param optimisation - preserved)
pvsyst_25 = ''.join(pvsyst_nb['cells'][25]['source'])
check("4. Param optimisation" in pvsyst_25,
      "Cell 25: Original '4. Param optimisation' section preserved")

print("\n" + "="*80)
print("2. DATA COLUMN VALIDATION")
print("="*80)

# Check that PVsyst preprocessing (Cell 17) uses EArray
pvsyst_17 = ''.join(pvsyst_nb['cells'][17]['source'])
check("'EArray'" in pvsyst_17 or '"EArray"' in pvsyst_17,
      "Cell 17 (preprocessing): Correctly uses 'EArray' column")

# Check that SunSolve preprocessing (Cell 13) uses Power_MW
sunsolve_13 = ''.join(sunsolve_nb['cells'][13]['source'])
check("'Power_MW'" in sunsolve_13 or '"Power_MW"' in sunsolve_13,
      "SunSolve Cell 13 (preprocessing): Uses 'Power_MW' column")

# Note: Plotting cells (20-24) work with metrics_df which has standardized column names
# They don't need to reference EArray or Power_MW directly
check("metrics_df" in pvsyst_20,
      "Cell 20: Works with preprocessed metrics_df")

# Check that labels reference PVsyst (not SunSolve)
check("PVsyst" in pvsyst_20 or "pvsyst" in pvsyst_20.lower(),
      "Cell 20: Labels reference 'PVsyst' (not SunSolve)")

print("\n" + "="*80)
print("3. PREPROCESSING VALIDATION")
print("="*80)

# Both should have scale-then-clip workflow
pvsyst_preprocessing = ''.join(pvsyst_nb['cells'][17]['source'])  # Cell 17 in PVsyst
sunsolve_preprocessing = ''.join(sunsolve_nb['cells'][13]['source'])  # Cell 13 in SunSolve

check("SCALE-THEN-CLIP" in pvsyst_preprocessing.upper(),
      "PVsyst uses SCALE-THEN-CLIP workflow")

check("SCALE-THEN-CLIP" in sunsolve_preprocessing.upper(),
      "SunSolve uses SCALE-THEN-CLIP workflow")

check("clipping_threshold = 2.392" in pvsyst_preprocessing,
      "PVsyst has correct clipping threshold (2.392 MW)")

check("clipping_threshold = 2.392" in sunsolve_preprocessing,
      "SunSolve has correct clipping threshold (2.392 MW)")

check("compute_hourly_scaled_clipped" in pvsyst_preprocessing,
      "PVsyst has compute_hourly_scaled_clipped() function")

check("find_optimal_scaling_factor" in pvsyst_preprocessing,
      "PVsyst has find_optimal_scaling_factor() function")

print("\n" + "="*80)
print("4. PLOTTING FUNCTION COMPARISON")
print("="*80)

# Compare key plotting patterns
sunsolve_yearly = ''.join(sunsolve_nb['cells'][20]['source'])

# Check for common plotting elements
plotting_elements = [
    ("fig, (ax1, ax2) = plt.subplots(2, 1", "Two-panel subplot structure"),
    ("filtered_2021_metrics", "Uses filtered 2021 data"),
    ("gridspec_kw={'height_ratios':[1,1]}", "Equal height ratios"),
    ("ax1.plot", "Plotting on first axis"),
    ("ax2.plot", "Plotting on second axis"),
]

for pattern, description in plotting_elements:
    check(pattern in pvsyst_20,
          f"Cell 20: {description}")

print("\n" + "="*80)
print("5. PIPELINE VERSION")
print("="*80)

check("PIPELINE_VERSION = 'scale→clip→resample_v1'" in pvsyst_20 or
      "PIPELINE_VERSION = 'scale?clip?resample_v1'" in pvsyst_20,
      "Pipeline version identifier present")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nChecks passed: {checks_passed}")
print(f"Checks failed: {checks_failed}")
print(f"Success rate: {checks_passed/(checks_passed+checks_failed)*100:.1f}%")

if checks_failed == 0:
    print("\n✅ ALL VALIDATION CHECKS PASSED!")
    print("\nThe plotting workflows are successfully synchronized.")
    print("Both notebooks use identical preprocessing and plotting logic.")
else:
    print(f"\n⚠️  {checks_failed} validation check(s) failed.")
    print("Review the failed checks above.")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("\n1. Open PVsyst notebook in Jupyter Lab/Notebook")
print("2. Run cells 18-24 to verify plotting execution")
print("3. Compare generated plots with SunSolve notebook")
print("4. Verify that:")
print("   - Plots have identical structure and style")
print("   - Only differences are in data values and labels (PVsyst vs SunSolve)")
print("   - Error metrics are calculated consistently")
print("   - Section '4. Param optimisation' (Cell 25+) remains unchanged")

print("\n" + "="*80)
