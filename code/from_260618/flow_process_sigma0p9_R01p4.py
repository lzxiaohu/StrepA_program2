# file name: flow_process_sigma0p9_R01p4.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import cdist
from pathlib import Path
import os
import time
import hashlib
from numpy.random import default_rng
import h5py
from tqdm import tqdm
from itertools import product
import csv

import functions_list_260528 as functions_list
import summary_stats_elms_260528 as ss


# Seed from theta
def seed_from_theta(theta, master_seed: int = 123):
    th = np.asarray(theta, np.float64).ravel()
    b = th.tobytes() + np.uint64(master_seed).tobytes()
    return int.from_bytes(hashlib.sha1(b).digest()[:8], 'little')


# Build parameters
def build_params(theta, fixed_params, core_params_num):
    theta = np.asarray(theta, float).ravel()
    if theta.size != core_params_num:
        raise ValueError(f"theta must be length-{core_params_num}, got {np.shape(theta)}")
    if core_params_num == 2:
        R0, sigma = float(theta[0]), float(theta[1])
        return np.array([fixed_params[0], fixed_params[1], fixed_params[2], sigma,
                         fixed_params[3], fixed_params[4], fixed_params[5], fixed_params[6],
                         fixed_params[7], fixed_params[8], R0], dtype=float)


# Simulate
def simulate_prevalence_v5_numba(theta, fixed_params, core_params_num, seed):
    seed = seed_from_theta(theta, master_seed=seed)
    rng = default_rng(seed)
    params = build_params(theta, fixed_params, core_params_num)
    AC, IMM, _ = functions_list.initialise_agents_v5(params, rng=rng)
    
    SSPrev_selected, SSPrev, AIBKS = functions_list.simulator_v6_numba(
        AC, IMM, params, 0, 1, seed=seed
    )
    # SSPrev_selected, SSPrev, AIBKS = functions_list.simulator_v6(AC, IMM, params, rng=rng)
    
    return SSPrev_selected.astype(float)


# Summary statistics
def summary_stats(series_2d):
    """Calculate summary statistics for a single simulation."""
    avg_prev_obs = ss.avg_prev_numpy(series_2d)
    var_prev_obs = np.sqrt(ss.var_prev_numpy(series_2d))
    avg_npmi_obs = ss.avg_npmi_numpy(series_2d)
    div_all_isolates_obs = ss.div_all_isolates_numpy(series_2d)
    
    return np.array([avg_prev_obs, var_prev_obs, avg_npmi_obs, div_all_isolates_obs], float)


def load_and_prepare_data(
        summary_stats_file='../../experimental_data/from_260618/all_summary_statistics_clean.h5',
        output_dir='../../experimental_data/from_260618',
        tag=''   # e.g. '_sigma0p9_R01p6'
):
    """Load, normalize, and save summary statistics (both raw and normalized CSV)."""

    print("=" * 70)
    print("LOADING AND PREPARING DATA")
    print("=" * 70)

    print(f"\nLoading from: {summary_stats_file}")
    with h5py.File(summary_stats_file, 'r') as f:
        summary_stats = f['summary_stats'][:]
        R0 = f['R0'][:]
        sigma = f['sigma'][:]
        columns = list(f.attrs['columns'])

    n_samples = len(R0)
    print(f"✓ Loaded {n_samples:,} samples")

    col_min = summary_stats.min(axis=0)
    col_max = summary_stats.max(axis=0)
    summary_stats_norm = (summary_stats - col_min) / (col_max - col_min)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    norm_file = output_path / f'summary_stats_normalized{tag}.h5'
    with h5py.File(norm_file, 'w') as f:
        f.create_dataset('summary_stats_normalized', data=summary_stats_norm, compression='gzip')
        f.create_dataset('summary_stats_raw', data=summary_stats, compression='gzip')
        f.create_dataset('R0', data=R0, compression='gzip')
        f.create_dataset('sigma', data=sigma, compression='gzip')
        f.create_dataset('col_min', data=col_min)
        f.create_dataset('col_max', data=col_max)
        f.attrs['columns'] = columns
        f.attrs['n_samples'] = n_samples

    csv_raw = output_path / f'summary_stats{tag}.csv'
    csv_norm = output_path / f'summary_stats_normalized{tag}.csv'

    pd.DataFrame(summary_stats).to_csv(csv_raw, index=False, header=False)
    pd.DataFrame(summary_stats_norm).to_csv(csv_norm, index=False, header=False)
    pd.DataFrame(R0).to_csv(output_path / f'R0{tag}.csv', index=False, header=False)
    pd.DataFrame(sigma).to_csv(output_path / f'sigma{tag}.csv', index=False, header=False)

    print(f"\n✓ Saved:")
    print(f"  - {norm_file}")
    print(f"  - {csv_raw} (raw)")
    print(f"  - {csv_norm} (normalized)")
    print(f"  - R0{tag}.csv, sigma{tag}.csv")

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


def dist_std(X):
    """
    Distance standard deviation of a set of points.
    Smaller = more concentrated.
    
    X: shape (n, 2) → [R0, sigma] pairs
    """
    
    # Pairwise distances between all points
    D = cdist(X, X, metric='euclidean')
    
    # Mean pairwise distance = spread
    return D.mean()


def calculate_posetrior_concentration_abc(
    file_dists='../../experimental_data/from_260618/dists_observations_recal.csv',
    file_R0='../../experimental_data/from_260618/R0.csv',
    file_sigma='../../experimental_data/from_260618/sigma.csv',
    percentile=1.0
):
    """
    Calculate posterior concentration (all samples).
    
    Small value = ABC learned a lot from the data (posterior tightly concentrated)
    Large value = ABC learned little from the observation(posterior spread out)
    
    Parameters:
    -----------
    file_dists : str
        CSV file with distances
    file_R0 : str
        CSV file with R0 values
    file_sigma : str
        CSV file with sigma values
    percentile : float
        Percentile threshold for posterior
    
    Returns:
    --------
    float : posterior concentration
    """
    
    print("="*70)
    print("CALCULATING DIST STD: POSTERIOR CONCENTRATION")
    print("="*70)
    
    # Load data
    distances = pd.read_csv(file_dists, header=None).values.ravel()
    R0_array = pd.read_csv(file_R0, header=None).values.ravel()
    sigma_array = pd.read_csv(file_sigma, header=None).values.ravel()
    
    n_total = len(distances)
    print(f"Total samples: {n_total:,}")
    
    # -------------------------------------------------------------------------
    # Cloud 1: POSTERIOR = accepted samples at percentile
    # -------------------------------------------------------------------------
    threshold = np.percentile(distances, percentile)
    selected = distances <= threshold
    
    post_R0 = R0_array[selected]
    post_sigma = sigma_array[selected]
    n_posterior = selected.sum()
    
    print(f"Posterior ({percentile}%): {n_posterior:,} samples")
    print(f"  R0:    mean={post_R0.mean():.3f}, std={post_R0.std():.3f}")
    print(f"  sigma: mean={post_sigma.mean():.3f}, std={post_sigma.std():.3f}") 
    
    
    # Stack into (n, 2) arrays: [R0, sigma]
    X = np.column_stack([post_R0, post_sigma])    # posterior
        
    print(f"Posterior shape: {X.shape}")
    
    # -------------------------------------------------------------------------
    # Calculate posterior concentration
    # -------------------------------------------------------------------------
    print(f"\nCalculating posterior concentration...")
    ed = dist_std(X)
    
    print(f"\n{'='*70}")
    print(f"posterior concentration RESULT")
    print(f"{'='*70}")
    print(f"Distance std (posterior concentration): {ed:.6f}")
    print(f"\nInterpretation:")
    print(f"  Small value → posterior tightly concentrated")
    print(f"               → ABC learned a lot from the observation")
    print(f"  Large value → posterior spread out")
    print(f"               → ABC learned little from the observation")
    
    return ed


def plot_posetrior_concentration(
    file_dists='../../experimental_data/from_260618/dists_sigma0p9_R01p4_seed1690_recal.csv',
    file_R0='../../experimental_data/from_260618/R0.csv',
    file_sigma='../../experimental_data/from_260618/sigma.csv',
    percentile=1.0,
    true_R0=None,        # ← True R0 value to mark
    true_sigma=None,     # ← True sigma value to mark
    xlim=(1, 4),
    ylim=(0.2, 1.0),
    bw_method='silverman',
    title="ABC Posterior",
    save_path='../../figures/from_260618/ppc/sigma0p9/R01p4_seed1690/',
    save_filename='posterior_concentration_plot.png'
):
    """
    Plot scatter and contour side by side for the posterior samples.
    Marks true R0 and sigma values if provided.
    """
    
    print("="*70)
    print("PLOTTING SCATTER + CONTOUR")
    print("="*70)
    
    # Load data
    distances = pd.read_csv(file_dists, header=None).values.ravel()
    R0_array = pd.read_csv(file_R0, header=None).values.ravel()
    sigma_array = pd.read_csv(file_sigma, header=None).values.ravel()
    
    # Select samples at percentile
    threshold = np.percentile(distances, percentile)
    selected = distances <= threshold
    
    selected_R0 = R0_array[selected]
    selected_sigma = sigma_array[selected]
    n_selected = selected.sum()
    
    print(f"Percentile {percentile}%: {n_selected:,} samples")
    
    if true_R0 is not None and true_sigma is not None:
        print(f"True values: R0={true_R0}, sigma={true_sigma}")
    
    # Find peak from KDE
    xy = np.vstack([selected_R0, selected_sigma])
    kde = stats.gaussian_kde(xy, bw_method=bw_method)
    
    x_grid = np.linspace(xlim[0], xlim[1], 200)
    y_grid = np.linspace(ylim[0], ylim[1], 200)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    
    peak_idx = np.unravel_index(np.argmax(Z), Z.shape)
    peak_R0 = X[peak_idx]
    peak_sigma = Y[peak_idx]
    
    print(f"Peak: R0={peak_R0:.4f}, sigma={peak_sigma:.4f}")
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # ========================================================================
    # Left: Scatter plot
    # ========================================================================
    sc = ax1.scatter(selected_R0, selected_sigma,
                    c=distances[selected],
                    cmap='YlOrRd_r',
                    alpha=0.4, s=10, zorder=2)
    
    cbar1 = plt.colorbar(sc, ax=ax1, pad=0.02)
    cbar1.set_label('Distance', fontsize=12, fontweight='bold')
    
    # Mark peak
    ax1.scatter(peak_R0, peak_sigma,
               s=300, c='blue', marker='+',
               linewidths=3, zorder=10,
               label=f'Peak (R0={peak_R0:.3f}, σ={peak_sigma:.3f})')
    ax1.annotate(f'R0={peak_R0:.3f}\nσ={peak_sigma:.3f}',
                xy=(peak_R0, peak_sigma),
                xytext=(15, 15), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='blue', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    
    # Mark true values
    if true_R0 is not None and true_sigma is not None:
        ax1.scatter(true_R0, true_sigma,
                   s=300, c='red', marker='*',
                   edgecolors='black', linewidths=1.5,
                   zorder=11, label=f'True (R0={true_R0}, σ={true_sigma})')
        ax1.annotate(f'True\nR0={true_R0}\nσ={true_sigma}',
                    xy=(true_R0, true_sigma),
                    xytext=(-60, 15), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    ax1.set_xlabel('R0', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Sigma', fontsize=13, fontweight='bold')
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_title(f'Scatter\n({n_selected:,} samples)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10, framealpha=0.9)
    
    # ========================================================================
    # Right: Contour plot
    # ========================================================================
    contourf = ax2.contourf(X, Y, Z, levels=10,
                           cmap='YlOrRd', alpha=0.8, zorder=2)
    ax2.contour(X, Y, Z, levels=10,
               colors='black', linewidths=0.5, alpha=0.5, zorder=3)
    
    cbar2 = plt.colorbar(contourf, ax=ax2, pad=0.02)
    cbar2.set_label('Density', fontsize=12, fontweight='bold')
    
    # Mark peak
    ax2.scatter(peak_R0, peak_sigma,
               s=300, c='blue', marker='+',
               linewidths=3, zorder=10,
               label=f'Peak (R0={peak_R0:.3f}, σ={peak_sigma:.3f})')
    ax2.annotate(f'R0={peak_R0:.3f}\nσ={peak_sigma:.3f}',
                xy=(peak_R0, peak_sigma),
                xytext=(15, 15), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='blue', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    
    # Mark true values
    if true_R0 is not None and true_sigma is not None:
        ax2.scatter(true_R0, true_sigma,
                   s=300, c='red', marker='*',
                   edgecolors='black', linewidths=1.5,
                   zorder=11, label=f'True (R0={true_R0}, σ={true_sigma})')
        ax2.annotate(f'True\nR0={true_R0}\nσ={true_sigma}',
                    xy=(true_R0, true_sigma),
                    xytext=(-60, 15), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    ax2.set_xlabel('R0', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Sigma', fontsize=13, fontweight='bold')
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_title(f'Contour (KDE bw={bw_method})\n({n_selected:,} samples)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10, framealpha=0.9)
    
    # Overall title
    fig.suptitle(f'{title} - {percentile}% percentile',
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save
    Path(save_path).mkdir(parents=True, exist_ok=True)
    save_file = f"{save_path}{save_filename}"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved to: {save_file}")
    plt.close()
    
    return peak_R0, peak_sigma



def analyze_distances(
    filepath: str,
    standard_point: tuple,
    weights: tuple = None,
    metric: str = "euclidean",
    p: float = 3,
    output_path: str = "output.csv"
) -> pd.DataFrame:
    """
    Load a headerless CSV with 4 features, apply Min-Max normalization,
    and compute weighted distance of each row to a standard point.

    Parameters
    ----------
    filepath       : path to the input CSV file (no header, 4 columns)
    standard_point : tuple of (A0, B0, C0, D0)
    weights        : optional tuple of (wA, wB, wC, wD)
    metric         : distance metric (euclidean, manhattan, chebyshev, minkowski, cosine)
    p              : power for minkowski distance
    output_path    : path to save the result CSV

    Returns
    -------
    pd.DataFrame with normalized columns A, B, C, D and a 'distance' column
    """
    # 1. Load
    df = pd.read_csv(filepath, header=None, names=["A", "B", "C", "D"])

    # 2. Min-Max normalization
    col_min = df.min()
    col_max = df.max()
    df_norm = (df - col_min) / (col_max - col_min)

    # 3. Normalize the standard point on the same scale
    standard = np.array(standard_point)
    standard_norm = (standard - col_min.values) / (col_max.values - col_min.values)

    # 4. Resolve weights (normalize so they sum to 1)
    if weights is not None:
        w = np.array(weights, dtype=float)
        if len(w) != 4:
            raise ValueError("weights must have exactly 4 values (wA, wB, wC, wD).")
        if np.any(w < 0):
            raise ValueError("All weights must be non-negative.")
        w = w / w.sum()
    else:
        w = np.array([0.25, 0.25, 0.25, 0.25])

    # 5. Weighted distance
    diff = (df_norm[["A", "B", "C", "D"]] - standard_norm).values
    
    if metric == "euclidean":
        dist = np.sqrt((w * diff ** 2).sum(axis=1))
    elif metric == "manhattan":
        dist = (w * np.abs(diff)).sum(axis=1)
    elif metric == "chebyshev":
        dist = (w * np.abs(diff)).max(axis=1)
    elif metric == "minkowski":
        dist = ((w * np.abs(diff) ** p).sum(axis=1)) ** (1 / p)
    elif metric == "cosine":
        dot = (w * df_norm[["A", "B", "C", "D"]].values * standard_norm).sum(axis=1)
        norm_x = np.sqrt((w * df_norm[["A", "B", "C", "D"]].values ** 2).sum(axis=1))
        norm_x0 = np.sqrt((w * standard_norm ** 2).sum())
        dist = 1 - dot / (norm_x * norm_x0)
    else:
        raise ValueError(f"Unknown metric '{metric}'")

    df_norm["distance"] = dist

    # 6. Save & return
    df_norm[["distance"]].to_csv(output_path, index=False, header=False)
    print(f"Done. Results saved to '{output_path}'.")
    return df_norm









def plot_r0_sigma(true_R0, true_sigma, peak_R0s, peak_sigmas,
                   xlim=(1, 4), ylim=(0.2, 1.0),
                   title="True value vs Peak value", 
                   save_path='../../figures/from_260618/ppc/sigma0p9/R01p4/', 
                   save_filename='peak_values_plot.png'):
    fig, ax = plt.subplots(figsize=(7, 6))

    # 20 peak estimates as blue crosses
    ax.scatter(peak_R0s, peak_sigmas, marker='+', color='blue', s=300,
               label='Peak estimates', zorder=2)

    # True value as a red star
    ax.scatter(true_R0, true_sigma, marker='*', color='red', s=300,
               label='True value', zorder=3)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r'$R_0$', fontsize=13)
    ax.set_ylabel(r'$\sigma$', fontsize=13)
    if title:
        ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, save_filename)
        plt.savefig(full_path, dpi=300)
        print(f"Saved to {full_path}")

    plt.show()
    return fig, ax


def save_results_to_csv(TRUE_R0, TRUE_SIGMA, peak_R0s, peak_sigmas,
                         save_path='', save_filename='results.csv'):
    full_path = os.path.join(save_path, save_filename) if save_path else save_filename

    if save_path:
        os.makedirs(save_path, exist_ok=True)

    with open(full_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['TRUE_R0', 'TRUE_SIGMA', 'peak_R0', 'peak_sigma'])
        for r0, sigma in zip(peak_R0s, peak_sigmas):
            writer.writerow([TRUE_R0, TRUE_SIGMA, r0, sigma])

    print(f"Saved to {full_path}")

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":

    start = time.perf_counter()

    # True values from filename
    # to be tuned
    true_R0 = 1.4
    true_sigma = 0.9

    seed_number = [1591, 266, 9117, 9749, 9900, 
                    3992, 1338, 6070, 2489, 4915, 
                    2222, 9734, 8660, 6342, 3746, 
                    1018, 9765, 9474, 5175, 7357]
    # seed_number = [1212, 1234, 1267, 1314, 1690]

    peak_R0s = []
    peak_sigmas = []

    save_path='../../figures/from_260618/ppc/sigma0p9/R01p4'
    # save_path='../../figures/from_260618/ppc/sigma0p9/R01p4_seed1690/'

    file_dists='../../experimental_data/from_260618/dists_sigma0p9_R01p4'
    # file_dists='../../experimental_data/from_260618/dists_sigma0p9_R01p4_seed1690_recal'
    file_R0='../../experimental_data/from_260618/R0'
    file_sigma='../../experimental_data/from_260618/sigma'

    core_params_num = 2  # core params: R0 and sigma

    # Fixed parameters
    DurationSimulation = 20.0
    Nstrains = 42
    omega = 0.2
    x = 10.0
    Cperweek = 34.53
    Nagents = 2500
    alpha = 7.0
    AgeDeath = 71.0

    if core_params_num == 2:
        Dimmunity = 10.0 * 52.14
        fixed_params = np.array([DurationSimulation, Nstrains, Dimmunity, omega,
                            x, Cperweek, Nagents, alpha, AgeDeath], dtype=float)
    else:
        raise ValueError('Invalid core params num')

    tag = '_sigma0p9_R01p4'

    data = load_and_prepare_data(
            summary_stats_file='../../experimental_data/from_260618/all_summary_statistics_clean.h5',
            output_dir='../../experimental_data/from_260618',
            tag=tag
        )


    for seed_sn in seed_number: 

        theta_true = np.array([true_R0, true_sigma], float)

        # Generate synthetic observed data
        Tdry = simulate_prevalence_v5_numba(theta_true, fixed_params, core_params_num, seed=seed_sn)
        # print(Tdry)
        
        # Calculate summary statistics (this is your standard point)
        s_obs = summary_stats(Tdry)
        
        print(f"  Standard point (s_obs): {s_obs}")
        
        # Test reproducibility
        Tdry_check = simulate_prevalence_v5_numba(theta_true, fixed_params, core_params_num, seed=seed_sn)
        is_reproducible = np.allclose(Tdry, Tdry_check)
        print(f"  Reproducible: {is_reproducible} ✓" if is_reproducible else f"  Reproducible: {is_reproducible} ✗")

        

        # Set your observed data (standard point)
        standard_point = s_obs
        standard_point_norm = normalize_standard_point(
            standard_point,
            data['col_min'],
            data['col_max']
        )

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

        # Step 6: Save results
        output_file = f'../../experimental_data/from_260618/optimal_weights_results_sigma0p9_R01p4_seed{seed_sn}.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\n✓ Saved full results to: {output_file}")

        weights = results_df.iloc[0][['w1', 'w2', 'w3', 'w4']].values

        # Calculate distances with Euclidean metric
        result = analyze_distances(
            filepath="../../experimental_data/from_260618/summary_stats_normalized.csv",
            standard_point=standard_point_norm,  # Fixed: added commas
            weights=weights,  # Fixed: added commas
            metric="euclidean",
            p=3,
            output_path=f"../../experimental_data/from_260618/dists_sigma0p9_R01p4_seed{seed_sn}_recal.csv"
        )


        for percentile, filename in [
            (0.01, 'posterior_scatter_contour_0p01percentile'),
        ]:
            print(f"\n{'='*70}")
            print(f"PERCENTILE: {percentile}%")
            print(f"{'='*70}")
            
            # posterior concentration
            ed = calculate_posetrior_concentration_abc(
                file_dists=f'{file_dists}_seed{seed_sn}_recal.csv',
                file_R0=f'{file_R0}.csv',
                file_sigma=f'{file_sigma}.csv',
                percentile=percentile
            )
            print(f"\n✅ posterior concentration: {ed:.6f}")
            
            for bw in ['silverman', 'scott', 0.5]:

                # Plot
                peak_R0, peak_sigma = plot_posetrior_concentration(
                    file_dists=f'{file_dists}_seed{seed_sn}_recal.csv',
                    file_R0=f'{file_R0}.csv',
                    file_sigma=f'{file_sigma}.csv',
                    percentile=percentile,
                    true_R0=true_R0,        # ← True R0 from filename
                    true_sigma=true_sigma,  # ← True sigma from filename
                    xlim=(1, 4),
                    ylim=(0.2, 1.0),
                    bw_method=bw,
                    title="ABC Posterior",
                    save_path=f'{save_path}_seed{seed_sn}/',
                    save_filename=f'{filename}_{bw}.png'
                )
                print(f"✅ Peak: R0={peak_R0:.4f}, sigma={peak_sigma:.4f}")

            peak_R0s.append(peak_R0)
            peak_sigmas.append(peak_sigma)
    
    plot_r0_sigma(true_R0, true_sigma, peak_R0s, peak_sigmas, 
                    xlim=(1, 4), 
                    ylim=(0.2, 1.0),
                    title="True value vs Peak value", 
                    save_path=f'{save_path}/', 
                    save_filename='peak_values_plot.png')

save_results_to_csv(true_R0, true_sigma, peak_R0s, peak_sigmas,
                         save_path='../../experimental_data/from_260618/', save_filename='peak_estimates_sigma0p9_R01p4.csv')

end = time.perf_counter()
print(f"\n⏱️  Elapsed: {end - start:.2f} seconds")
print("="*70)