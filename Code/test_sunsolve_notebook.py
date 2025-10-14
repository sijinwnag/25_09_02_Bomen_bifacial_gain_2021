#!/usr/bin/env python3
"""
Iteratively test SunSolve notebook cells.
Execute cells one by one, catch and report errors.
"""

import json
import sys
from pathlib import Path

# Global namespace to maintain state across cells
cell_namespace = {}

def execute_cell(cell_code, cell_num):
    """Execute a notebook cell and catch errors."""
    try:
        # Execute in the global namespace to maintain state
        exec(cell_code, cell_namespace)
        return True, None
    except Exception as e:
        import traceback
        error_details = {
            'type': type(e).__name__,
            'message': str(e),
            'traceback': traceback.format_exc()
        }
        return False, error_details

def test_notebook():
    """Test notebook cells iteratively."""
    print("="*70)
    print("SUNSOLVE NOTEBOOK ITERATIVE TESTING")
    print("="*70)

    # Load notebook
    notebook_path = '25_10_08_Data_visualiser_matching_inv_SunSolve.ipynb'
    print(f"\nLoading: {notebook_path}")

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    print(f"Total cells: {len(nb['cells'])}\n")

    # Priority cells to test
    test_cells = [1, 3, 6, 9, 12, 16, 18, 20]  # Skip markdown cells

    results = []
    failed_at = None

    print("Testing critical code cells...\n")
    print("-"*70)

    for i in test_cells:
        cell = nb['cells'][i]

        if cell.get('cell_type') != 'code':
            print(f"Cell {i:2d}: SKIPPED (not code)")
            continue

        # Get cell source
        cell_code = ''.join(cell['source'])

        # Preview
        preview = cell_code[:80].replace('\n', ' ').strip()
        print(f"\nCell {i:2d}: {preview}...")

        # Execute
        success, error = execute_cell(cell_code, i)

        if success:
            print(f"         [PASS]")
            results.append((i, 'PASS', None))
        else:
            print(f"         [FAIL] {error['type']}: {error['message']}")
            results.append((i, 'FAIL', error))
            failed_at = i
            break  # Stop at first failure

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for r in results if r[1] == 'PASS')
    failed = sum(1 for r in results if r[1] == 'FAIL')

    print(f"\nResults: {passed} passed, {failed} failed out of {len(test_cells)} tested")

    if failed > 0:
        print(f"\nFirst failure at cell {failed_at}")
        fail_result = [r for r in results if r[1] == 'FAIL'][0]
        print(f"\nError details:")
        print(f"  Type: {fail_result[2]['type']}")
        print(f"  Message: {fail_result[2]['message']}")
        print(f"\nFull traceback:")
        print(fail_result[2]['traceback'])

        # Suggest fixes
        print("\n" + "-"*70)
        print("SUGGESTED FIXES:")
        print("-"*70)

        error_type = fail_result[2]['type']
        error_msg = fail_result[2]['message']

        if error_type == 'ModuleNotFoundError' or error_type == 'ImportError':
            print(f"  Missing library: {error_msg}")
            print(f"  Fix: Install missing package or add to imports")

        elif error_type == 'NameError':
            print(f"  Undefined variable: {error_msg}")
            print(f"  Fix: Check if previous cells executed, or variable name changed")

        elif error_type == 'KeyError':
            print(f"  Missing column/key: {error_msg}")
            print(f"  Fix: Verify column name replacement or data structure")

        elif error_type == 'FileNotFoundError':
            print(f"  File not found: {error_msg}")
            print(f"  Fix: Check file path is correct")

        else:
            print(f"  Unexpected error: {error_type}")
            print(f"  Review traceback above for details")

    else:
        print("\nAll tested cells passed!")
        print("\nVerifying key variables...")

        # Check key variables
        checks = [
            ('inverter', 'Inverter configuration'),
            ('df', 'Electrical data DataFrame'),
            ('simulation_results_df', 'Simulation results DataFrame'),
            ('Power_MW', 'Power_MW column (in simulation_results_df)')
        ]

        for var_name, description in checks:
            if var_name == 'Power_MW':
                # Check if column exists in dataframe
                if 'simulation_results_df' in cell_namespace:
                    sim_df = cell_namespace['simulation_results_df']
                    if 'Power_MW' in sim_df.columns:
                        print(f"  * {var_name}: EXISTS in simulation_results_df")
                    else:
                        print(f"  X {var_name}: MISSING from simulation_results_df")
                else:
                    print(f"  X {var_name}: Cannot check (simulation_results_df not found)")
            elif var_name in cell_namespace:
                val = cell_namespace[var_name]
                print(f"  * {var_name}: {type(val).__name__} - {description}")
            else:
                print(f"  X {var_name}: NOT FOUND - {description}")

    return failed == 0

if __name__ == '__main__':
    try:
        success = test_notebook()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
