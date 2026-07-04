# calculate_posterior_concentration_sigma0p94_R01p2.py
# Calculate posterior concentration

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import cdist
from pathlib import Path


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
    file_dists='../../experimental_data/from_260703/dists_observations_recal.csv',
    file_R0='../../experimental_data/from_260703/R0.csv',
    file_sigma='../../experimental_data/from_260703/sigma.csv',
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
    file_dists='../../experimental_data/from_260703/dists_sigma0p94_R01p2_recal.csv',
    file_R0='../../experimental_data/from_260703/R0.csv',
    file_sigma='../../experimental_data/from_260703/sigma.csv',
    percentile=1.0,
    true_R0=None,        # ← True R0 value to mark
    true_sigma=None,     # ← True sigma value to mark
    xlim=(1, 4),
    ylim=(0.2, 1.0),
    bw_method='silverman',
    title="ABC Posterior",
    save_path='../../figures/from_260703/ppc/sigma0p94/R01p2/',
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


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":

    # True values from filename
    TRUE_R0 = 1.1
    TRUE_SIGMA = 0.9

    for percentile, filename in [
        (0.01, 'posterior_scatter_contour_0p01percentile'),
    ]:
        print(f"\n{'='*70}")
        print(f"PERCENTILE: {percentile}%")
        print(f"{'='*70}")
        
        # posterior concentration
        ed = calculate_posetrior_concentration_abc(
            file_dists='../../experimental_data/from_260703/dists_sigma0p94_R01p2_recal.csv',
            file_R0='../../experimental_data/from_260703/R0.csv',
            file_sigma='../../experimental_data/from_260703/sigma.csv',
            percentile=percentile
        )
        print(f"\n✅ posterior concentration: {ed:.6f}")
        
        for bw in ['silverman', 'scott', 0.5, 0.3]:

            # Plot
            peak_R0, peak_sigma = plot_posetrior_concentration(
                file_dists='../../experimental_data/from_260703/dists_sigma0p94_R01p2_recal.csv',
                file_R0='../../experimental_data/from_260703/R0.csv',
                file_sigma='../../experimental_data/from_260703/sigma.csv',
                percentile=percentile,
                true_R0=TRUE_R0,        # ← True R0 from filename
                true_sigma=TRUE_SIGMA,  # ← True sigma from filename
                xlim=(1, 4),
                ylim=(0.2, 1.0),
                bw_method=bw,
                title="ABC Posterior",
                save_path='../../figures/from_260703/ppc/sigma0p94/R01p2/',
                save_filename=f'{filename}_{bw}.png'
            )
            print(f"✅ Peak: R0={peak_R0:.4f}, sigma={peak_sigma:.4f}")