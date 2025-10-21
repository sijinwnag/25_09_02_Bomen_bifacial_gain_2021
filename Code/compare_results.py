import pandas as pd
import numpy as np

# Load the two result files
df1 = pd.read_csv('C:/Users/z5183876/OneDrive - UNSW/Documents/GitHub/25_09_02_Bomen_bifacial_gain_2021/Results/pvsyst_batch_evaluation_scaled_clip_2_1_Perez.csv')
df2 = pd.read_csv('C:/Users/z5183876/OneDrive - UNSW/Documents/GitHub/25_09_02_Bomen_bifacial_gain_2021/Results/pvsyst_batch_evaluation_results_2_1_Perez.csv')

print("=" * 80)
print("CSV FILE COMPARISON: Scale-Then-Clip vs Clip-Then-Scale")
print("=" * 80)

print(f"\nFile 1 (scale-then-clip): {df1.shape[0]} rows × {df1.shape[1]} columns")
print(f"File 2 (clip-then-scale): {df2.shape[0]} rows × {df2.shape[1]} columns")

print(f"\nColumns match: {list(df1.columns) == list(df2.columns)}")

print("\nDataFrames identical: ", df1.equals(df2))

# Check numerical differences
print("\n" + "=" * 80)
print("NUMERICAL DIFFERENCES BY COLUMN")
print("=" * 80)

numerical_cols = df1.select_dtypes(include=[np.number]).columns
differences_found = False

for col in numerical_cols:
    if col in df2.columns:
        diff = (df1[col] - df2[col]).abs()
        max_diff = diff.max()
        mean_diff = diff.mean()

        if max_diff > 1e-10:  # Only show significant differences
            differences_found = True
            print(f"\n{col}:")
            print(f"  Max difference:  {max_diff:.6e}")
            print(f"  Mean difference: {mean_diff:.6e}")
            print(f"  Non-zero diffs: {(diff > 1e-10).sum()} / {len(diff)} rows")

if not differences_found:
    print("\nNo significant numerical differences found (all < 1e-10)")

# Show specific comparison for RMSE and scaling factor
print("\n" + "=" * 80)
print("RMSE AND SCALING FACTOR COMPARISON (first 10 rows)")
print("=" * 80)

comparison_df = pd.DataFrame({
    'File': df1['file directory'].str.split('\\').str[-1].head(10),
    'RMSE_scaled_clip': df1['RMSE'].head(10),
    'RMSE_clip_scale': df2['RMSE'].head(10),
    'RMSE_diff': (df1['RMSE'] - df2['RMSE']).abs().head(10),
    'Scale_scaled_clip': df1['The optimised scale factor'].head(10),
    'Scale_clip_scale': df2['The optimised scale factor'].head(10),
    'Scale_diff': (df1['The optimised scale factor'] - df2['The optimised scale factor']).abs().head(10),
})

print(comparison_df.to_string(index=False))

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

print("\nRMSE differences:")
rmse_diff = (df1['RMSE'] - df2['RMSE']).abs()
print(f"  Min:  {rmse_diff.min():.6e}")
print(f"  Mean: {rmse_diff.mean():.6e}")
print(f"  Max:  {rmse_diff.max():.6e}")

print("\nScaling factor differences:")
scale_diff = (df1['The optimised scale factor'] - df2['The optimised scale factor']).abs()
print(f"  Min:  {scale_diff.min():.6e}")
print(f"  Mean: {scale_diff.mean():.6e}")
print(f"  Max:  {scale_diff.max():.6e}")
