# best_weights_for_observations.py
# Process summary statistics to find optimal weights for ABC

import numpy as np
import pandas as pd
import h5py
from itertools import product
from pathlib import Path
from tqdm import tqdm


def load_and_prepare_data(
        summary_stats_file='../../experimental_data/from_260703/all_summary_statistics_clean.h5',
        output_dir='../../experimental_data/from_260703'
):
    """Load, normalize, and save summary statistics (both raw and normalized CSV)."""

    print("=" * 70)
    print("LOADING AND PREPARING DATA")
    print("=" * 70)

    # Load
    print(f"\nLoading from: {summary_stats_file}")
    with h5py.File(summary_stats_file, 'r') as f:
        summary_stats = f['summary_stats'][:]
        R0 = f['R0'][:]
        sigma = f['sigma'][:]
        columns = list(f.attrs['columns'])

    n_samples = len(R0)
    print(f"✓ Loaded {n_samples:,} samples")

    # Normalize
    col_min = summary_stats.min(axis=0)
    col_max = summary_stats.max(axis=0)
    summary_stats_norm = (summary_stats - col_min) / (col_max - col_min)

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # HDF5
    norm_file = output_path / 'summary_stats_normalized.h5'
    with h5py.File(norm_file, 'w') as f:
        f.create_dataset('summary_stats_normalized', data=summary_stats_norm, compression='gzip')
        f.create_dataset('summary_stats_raw', data=summary_stats, compression='gzip')
        f.create_dataset('R0', data=R0, compression='gzip')
        f.create_dataset('sigma', data=sigma, compression='gzip')
        f.create_dataset('col_min', data=col_min)
        f.create_dataset('col_max', data=col_max)
        f.attrs['columns'] = columns
        f.attrs['n_samples'] = n_samples

    # CSV - BOTH raw and normalized
    csv_raw = output_path / 'summary_stats.csv'
    csv_norm = output_path / 'summary_stats_normalized.csv'
    
    pd.DataFrame(summary_stats).to_csv(csv_raw, index=False, header=False)
    pd.DataFrame(summary_stats_norm).to_csv(csv_norm, index=False, header=False)
    pd.DataFrame(R0).to_csv(output_path / 'R0.csv', index=False, header=False)
    pd.DataFrame(sigma).to_csv(output_path / 'sigma.csv', index=False, header=False)

    print(f"\n✓ Saved:")
    print(f"  - {norm_file}")
    print(f"  - {csv_raw} (raw)")
    print(f"  - {csv_norm} (normalized)")
    print(f"  - R0.csv, sigma.csv")

    return {
        'summary_stats': summary_stats,
        'summary_stats_norm': summary_stats_norm,
        'R0': R0,
        'sigma': sigma,
        'columns': columns,
        'col_min': col_min,
        'col_max': col_max,
        'n_samples': n_samples
    }



def normalize_standard_point(standard_point, col_min, col_max):
    """
    Normalize the standard point (observed data) using same scale as summary stats.

    Parameters:
    -----------
    standard_point : tuple or array
        (avg_prev_obs, var_prev_obs, avg_npmi_obs, div_all_isolates_obs)
    col_min : array
        Minimum values from simulation bank
    col_max : array
        Maximum values from simulation bank

    Returns:
    --------
    np.ndarray : Normalized standard point
    """
    standard = np.array(standard_point)
    standard_norm = (standard - col_min) / (col_max - col_min)

    print(f"\nStandard Point Normalization:")
    print(f"  Raw:        {standard}")
    print(f"  Normalized: {standard_norm}")

    return standard_norm


def compute_distance(summary_stats_norm,
                     standard_point_norm,
                     weights=None,
                     metric="euclidean",
                     p=3):
    """
    Compute distance between each simulation and the standard point.

    Parameters:
    -----------
    summary_stats_norm : np.ndarray
        Normalized summary statistics (n_samples, 4)
    standard_point_norm : np.ndarray
        Normalized observed data (4,)
    weights : tuple or None
        Weights for each statistic (will be normalized to sum=1)
    metric : str
        Distance metric: 'euclidean', 'manhattan', 'chebyshev', 'minkowski', 'cosine'
    p : float
        Power for Minkowski distance

    Returns:
    --------
    np.ndarray : Distance for each sample
    """

    # Resolve weights
    if weights is not None:
        w = np.array(weights, dtype=float)
        if len(w) != 4:
            raise ValueError("weights must have exactly 4 values")
        if np.any(w < 0):
            raise ValueError("All weights must be non-negative")
        w = w / w.sum()  # normalize to sum = 1
    else:
        w = np.array([0.25, 0.25, 0.25, 0.25])

    # Compute difference
    diff = summary_stats_norm - standard_point_norm

    # Compute distance based on metric
    if metric == "euclidean":
        dist = np.sqrt((w * diff ** 2).sum(axis=1))

    elif metric == "manhattan":
        dist = (w * np.abs(diff)).sum(axis=1)

    elif metric == "chebyshev":
        dist = (w * np.abs(diff)).max(axis=1)

    elif metric == "minkowski":
        dist = ((w * np.abs(diff) ** p).sum(axis=1)) ** (1 / p)

    elif metric == "cosine":
        dot = (w * summary_stats_norm * standard_point_norm).sum(axis=1)
        norm_x = np.sqrt((w * summary_stats_norm ** 2).sum(axis=1))
        norm_x0 = np.sqrt((w * standard_point_norm ** 2).sum())
        dist = 1 - dot / (norm_x * norm_x0)

    else:
        raise ValueError(f"Unknown metric '{metric}'")

    return dist


def concentration_score(R0_array, sigma_array, true_R0, true_sigma, selected_indices):
    """
    Calculate concentration score (MSE) for selected samples.

    Parameters:
    -----------
    R0_array : np.ndarray
        R0 values for all samples
    sigma_array : np.ndarray
        Sigma values for all samples
    true_R0 : float
        True R0 value
    true_sigma : float
        True sigma value
    selected_indices : np.ndarray (bool)
        Boolean mask of selected samples

    Returns:
    --------
    float : Concentration score (lower is better)
    """
    selected_R0 = R0_array[selected_indices]
    selected_sigma = sigma_array[selected_indices]

    d_R0 = ((selected_R0 - true_R0) ** 2).mean()
    d_sigma = ((selected_sigma - true_sigma) ** 2).mean()

    return d_R0 + d_sigma


def find_optimal_weights(
        summary_stats_norm,
        R0_array,
        sigma_array,
        standard_point_norm,
        true_R0,
        true_sigma,
        percentile=0.05,
        metric="euclidean",
        weight_values=None
):
    """
    Find optimal weights by grid search.

    Parameters:
    -----------
    summary_stats_norm : np.ndarray
        Normalized summary statistics
    R0_array : np.ndarray
        R0 values
    sigma_array : np.ndarray
        Sigma values
    standard_point_norm : np.ndarray
        Normalized observed data
    true_R0 : float
        True R0 value
    true_sigma : float
        True sigma value
    percentile : float
        Percentile threshold for selection (default 0.05 = 5%)
    metric : str
        Distance metric
    weight_values : list
        Values for grid search (default: [0.1, 0.2, ..., 0.9])

    Returns:
    --------
    pd.DataFrame : Results sorted by score
    """

    if weight_values is None:
        weight_values = [i / 10 for i in range(1, 10)]  # 0.1 to 0.9

    print(f"\n{'=' * 70}")
    print(f"GRID SEARCH FOR OPTIMAL WEIGHTS")
    print(f"{'=' * 70}")
    print(f"Weight values: {weight_values}")
    print(f"Total combinations: {len(weight_values) ** 4:,}")
    print(f"Percentile threshold: {percentile}")
    print(f"Distance metric: {metric}")
    print(f"True values: R0={true_R0}, sigma={true_sigma}")
    print(f"{'=' * 70}\n")

    results = []

    # Grid search over all weight combinations
    for w in tqdm(list(product(weight_values, repeat=4)), desc="Testing weights"):
        # Compute distances
        distances = compute_distance(
            summary_stats_norm,
            standard_point_norm,
            weights=w,
            metric=metric
        )

        # Select samples below threshold
        threshold = np.quantile(distances, percentile)
        selected_indices = distances <= threshold
        n_selected = selected_indices.sum()

        # Calculate concentration score
        score = concentration_score(
            R0_array,
            sigma_array,
            true_R0,
            true_sigma,
            selected_indices
        )

        results.append({
            'w1': w[0],
            'w2': w[1],
            'w3': w[2],
            'w4': w[3],
            'score': score,
            'n_selected': n_selected,
            'threshold': threshold
        })

    # Convert to DataFrame and sort
    results_df = pd.DataFrame(results).sort_values('score')

    return results_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Step 1: Load and prepare data
    data = load_and_prepare_data(
        summary_stats_file='../../experimental_data/from_260703/all_summary_statistics_clean.h5',
        output_dir='../../experimental_data/from_260703'
    )
    print("col_min: ", data['col_min'])
    print("col_max: ",  data['col_max'])

    # Step 2: Set your observed data (standard point)
    standard_point = (9.56521739, 7.06351758, -0.59527674, 18.33333333)
    standard_point_norm = normalize_standard_point(
        standard_point,
        data['col_min'],
        data['col_max']
    )

    # Step 3: Set assumed parameter values 
    true_R0 = 0.0
    true_sigma = 0.0

    print(f"\n{'=' * 70}")
    print(f"TRUE PARAMETERS:")
    print(f"  R0: {true_R0}")
    print(f"  sigma: {true_sigma}")
    print(f"{'=' * 70}")

    # Step 4: Find optimal weights
    results_df = find_optimal_weights(
        summary_stats_norm=data['summary_stats_norm'],
        R0_array=data['R0'],
        sigma_array=data['sigma'],
        standard_point_norm=standard_point_norm,
        true_R0=true_R0,
        true_sigma=true_sigma,
        percentile=0.05,
        metric="euclidean",
        weight_values=[i / 10 for i in range(1, 10)]
    )

    # Step 5: Show results
    print(f"\n{'=' * 70}")
    print(f"TOP 20 WEIGHT COMBINATIONS (Euclidean Distance)")
    print(f"{'=' * 70}\n")
    print(results_df.head(20).to_string(index=False))

    # Step 6: Save results
    output_file = '../../experimental_data/from_260703/optimal_weights_results_observations.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved full results to: {output_file}")

    # Step 7: Show best weights
    best = results_df.iloc[0]
    print(f"\n{'=' * 70}")
    print(f"BEST WEIGHTS:")
    print(f"{'=' * 70}")
    print(f"  w1 (avg_prev):        {best['w1']:.2f}")
    print(f"  w2 (var_prev):        {best['w2']:.2f}")
    print(f"  w3 (avg_npmi):        {best['w3']:.2f}")
    print(f"  w4 (div_all_isolates): {best['w4']:.2f}")
    print(f"  Concentration score:  {best['score']:.6f}")
    print(f"  Samples selected:     {int(best['n_selected']):,}")
    print(f"{'=' * 70}")