# check_nan_values.py
# Quick diagnostic to check NaN values in summary statistics

import numpy as np
import h5py

summary_stats_file = '../../experimental_data/from_260703/all_summary_statistics_clean.h5'

print("="*70)
print("CHECKING NaN VALUES IN SUMMARY STATISTICS")
print("="*70)

with h5py.File(summary_stats_file, 'r') as f:
    summary_stats = f['summary_stats'][:]
    columns = list(f.attrs['columns'])

n_samples = len(summary_stats)

print(f"\nTotal samples: {n_samples:,}")
print(f"\nNaN Analysis per column:")
print("-"*70)

for i, col in enumerate(columns):
    col_data = summary_stats[:, i]
    
    n_nans = np.isnan(col_data).sum()
    n_valid = (~np.isnan(col_data)).sum()
    pct_nan = 100 * n_nans / n_samples
    
    if n_nans > 0:
        valid_min = np.nanmin(col_data)
        valid_max = np.nanmax(col_data)
        valid_mean = np.nanmean(col_data)
        
        print(f"\n{col}:")
        print(f"  NaN values:   {n_nans:,} ({pct_nan:.2f}%)")
        print(f"  Valid values: {n_valid:,} ({100-pct_nan:.2f}%)")
        print(f"  Valid range:  [{valid_min:.4f}, {valid_max:.4f}]")
        print(f"  Valid mean:   {valid_mean:.4f}")
        
        if pct_nan == 100:
            print(f"  ⚠️  ALL VALUES ARE NaN!")
        elif pct_nan > 50:
            print(f"  ⚠️  MORE THAN HALF ARE NaN!")
    else:
        col_min = col_data.min()
        col_max = col_data.max()
        col_mean = col_data.mean()
        
        print(f"\n{col}:")
        print(f"  ✓ No NaN values")
        print(f"  Range: [{col_min:.4f}, {col_max:.4f}]")
        print(f"  Mean:  {col_mean:.4f}")

print("\n" + "="*70)
print("RECOMMENDATION:")
print("="*70)

# Count columns with NaN
cols_with_nan = []
for i, col in enumerate(columns):
    if np.any(np.isnan(summary_stats[:, i])):
        cols_with_nan.append(col)

if len(cols_with_nan) == 0:
    print("✓ No NaN values found. Data is clean!")
elif len(cols_with_nan) == len(columns):
    print("❌ ALL columns have NaN values. Check your simulation code!")
else:
    print(f"⚠️  {len(cols_with_nan)} column(s) have NaN values:")
    for col in cols_with_nan:
        print(f"    - {col}")
    print(f"\nOptions:")
    print(f"  1. Exclude these columns and use only valid ones")
    print(f"  2. Fix the calculation function that produces NaN")
    print(f"  3. Replace NaN with a default value (not recommended)")

print("="*70)