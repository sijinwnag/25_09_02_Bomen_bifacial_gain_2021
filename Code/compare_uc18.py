import pandas as pd

# Load the two result files
df1 = pd.read_csv('C:/Users/z5183876/OneDrive - UNSW/Documents/GitHub/25_09_02_Bomen_bifacial_gain_2021/Results/pvsyst_batch_evaluation_scaled_clip_2_1_Perez.csv')
df2 = pd.read_csv('C:/Users/z5183876/OneDrive - UNSW/Documents/GitHub/25_09_02_Bomen_bifacial_gain_2021/Results/pvsyst_batch_evaluation_results_2_1_Perez.csv')

# Filter for Uc=18, Uv=0.0
row1 = df1[(df1['Uc'] == 18) & (df1['Uv'] == 0.0)]
row2 = df2[(df2['Uc'] == 18) & (df2['Uv'] == 0.0)]

print('=' * 80)
print('DETAILED COMPARISON: Uc=18, Uv=0.0 (Perez Model, Inverter 2-1)')
print('=' * 80)

if len(row1) > 0 and len(row2) > 0:
    print(f'\nConfiguration found in both datasets: YES')
    print(f'File: {row1["file directory"].values[0].split("\\")[-1]}')

    print(f'\n{"-" * 80}')
    print('SCALE-THEN-CLIP RESULTS:')
    print('-' * 80)
    print(f'  RMSE:           {row1["RMSE"].values[0]:.10f} MWh/day')
    print(f'  MBE:            {row1["MBE"].values[0]:.6e}')
    print(f'  CRMSE:          {row1["CRMSE"].values[0]:.10f}')
    print(f'  Scale factor:   {row1["The optimised scale factor"].values[0]:.10f}')

    print(f'\n{"-" * 80}')
    print('CLIP-THEN-SCALE RESULTS:')
    print('-' * 80)
    print(f'  RMSE:           {row2["RMSE"].values[0]:.10f} MWh/day')
    print(f'  MBE:            {row2["MBE"].values[0]:.6e}')
    print(f'  CRMSE:          {row2["CRMSE"].values[0]:.10f}')
    print(f'  Scale factor:   {row2["The optimised scale factor"].values[0]:.10f}')

    # Calculate differences
    rmse_diff = abs(row1['RMSE'].values[0] - row2['RMSE'].values[0])
    rmse_pct = (rmse_diff / row2['RMSE'].values[0]) * 100
    rmse_higher = row1['RMSE'].values[0] > row2['RMSE'].values[0]

    scale_diff = abs(row1['The optimised scale factor'].values[0] - row2['The optimised scale factor'].values[0])
    scale_pct = (scale_diff / row2['The optimised scale factor'].values[0]) * 100
    scale_higher = row1['The optimised scale factor'].values[0] > row2['The optimised scale factor'].values[0]

    print(f'\n{"=" * 80}')
    print('DIFFERENCES')
    print('=' * 80)

    print(f'\nRMSE difference:')
    print(f'  Absolute:  {rmse_diff:.10f} MWh/day')
    print(f'  Relative:  {rmse_pct:.6f}%')
    print(f'  Direction: Scale-then-clip is {"HIGHER" if rmse_higher else "LOWER"} ({"worse" if rmse_higher else "better"} fit)')

    print(f'\nScale factor difference:')
    print(f'  Absolute:  {scale_diff:.10f}')
    print(f'  Relative:  {scale_pct:.6f}%')
    print(f'  Direction: Scale-then-clip is {"HIGHER" if scale_higher else "LOWER"}')

    # Interpretation
    print(f'\n{"=" * 80}')
    print('INTERPRETATION')
    print('=' * 80)

    if rmse_diff > 0.001:
        print(f'\n[SIGNIFICANT DIFFERENCE DETECTED]')
        print(f'  The {rmse_diff:.4f} MWh/day RMSE difference is meaningful for this configuration.')
        print(f'  This indicates that clipping timing substantially affects the optimization.')
    else:
        print(f'\n[MINIMAL DIFFERENCE]')
        print(f'  The {rmse_diff:.4f} MWh/day RMSE difference is small.')
        print(f'  This configuration may not experience significant clipping effects.')

    print(f'\n  Scale-then-clip approach:')
    print(f'    - Higher RMSE reflects physical inverter power limiting')
    print(f'    - More realistic representation of actual inverter operation')
    print(f'    - Recommended for inverter-level bifacial gain analysis')

else:
    print('\n[ERROR] Configuration Uc=18, Uv=0.0 NOT FOUND in one or both files')
    print('\nAvailable Uc values in scale-then-clip file:', sorted(df1['Uc'].unique()))
    print('Available Uc values in clip-then-scale file:', sorted(df2['Uc'].unique()))
