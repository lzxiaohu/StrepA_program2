# file name: scatter_plots_observations.py 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def plot_histograms_with_kde(
    file_dists='dists.csv',
    file_R0='R0.csv',
    file_sigma='sigma.csv',
    percentile=5,
    true_R0=None,
    true_sigma=None,
    title="ABC Posterior Distribution",
    save_path='../../experimental_data/from_260703/',
    R0_range=(1, 8),
    sigma_range=(0.2, 1.0),
    bins=50
):
    """
    Plot histograms with KDE for R0 and sigma at a given percentile.
    
    Parameters:
    -----------
    file_dists : str
        CSV file with distances
    file_R0 : str
        CSV file with R0 samples
    file_sigma : str
        CSV file with sigma samples
    percentile : float
        Percentile threshold (e.g., 5 means keep closest 5%)
    true_R0 : float, optional
        True R0 value to mark
    true_sigma : float, optional
        True sigma value to mark
    title : str
        Figure title
    save_path : str
        Directory to save figure
    R0_range : tuple
        X-axis limits for R0
    sigma_range : tuple
        X-axis limits for sigma
    bins : int
        Number of histogram bins
    """
    
    # Load data
    distances = pd.read_csv(file_dists, header=None).values.ravel()
    data_R0 = pd.read_csv(file_R0, header=None).values.ravel()
    data_sigma = pd.read_csv(file_sigma, header=None).values.ravel()
    
    # Select samples
    threshold = np.percentile(distances, percentile)
    selected_indices = distances <= threshold
    
    selected_R0 = data_R0[selected_indices]
    selected_sigma = data_sigma[selected_indices]
    n_selected = len(selected_R0)
    
    print(f"Percentile {percentile}%: {n_selected:,} samples selected")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ========================================================================
    # R0 Histogram + KDE
    # ========================================================================
    
    # Histogram
    ax1.hist(selected_R0, bins=bins, density=True, alpha=0.6, 
             color='skyblue', edgecolor='black', linewidth=0.5, label='Histogram')
    
    # KDE
    kde_R0 = stats.gaussian_kde(selected_R0, bw_method='silverman')
    x_R0 = np.linspace(R0_range[0], R0_range[1], 500)
    kde_vals_R0 = kde_R0(x_R0)
    ax1.plot(x_R0, kde_vals_R0, 'b-', linewidth=2.5, label='KDE')
    
    # Find mode (peak of KDE)
    mode_idx_R0 = np.argmax(kde_vals_R0)
    mode_R0 = x_R0[mode_idx_R0]
    ax1.axvline(mode_R0, color='purple', linestyle='-.', linewidth=2,
               label=f'Mode = {mode_R0:.2f}')
    
    # Mark true value
    if true_R0 is not None:
        ax1.axvline(true_R0, color='red', linestyle='--', linewidth=2.5, 
                   label=f'True R0 = {true_R0}')
        # Add probability density at true value
        density_at_true = kde_R0(true_R0)[0]
        ax1.plot(true_R0, density_at_true, 'ro', markersize=10, zorder=5)
    
    # Statistics
    mean_R0 = selected_R0.mean()
    median_R0 = np.median(selected_R0)
    std_R0 = selected_R0.std()
    
    ax1.axvline(mean_R0, color='green', linestyle=':', linewidth=2, 
               label=f'Mean = {mean_R0:.2f}')
    ax1.axvline(median_R0, color='orange', linestyle=':', linewidth=2,
               label=f'Median = {median_R0:.2f}')
    
    # Labels
    ax1.set_xlabel('R0', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=14, fontweight='bold')
    ax1.set_title(f'R0 Distribution\n(mean={mean_R0:.2f}, std={std_R0:.2f})',
                 fontsize=15, fontweight='bold')
    ax1.set_xlim(R0_range)
    ax1.legend(loc='best', fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # ========================================================================
    # Sigma Histogram + KDE
    # ========================================================================
    
    # Histogram
    ax2.hist(selected_sigma, bins=bins, density=True, alpha=0.6,
             color='lightcoral', edgecolor='black', linewidth=0.5, label='Histogram')
    
    # KDE
    kde_sigma = stats.gaussian_kde(selected_sigma, bw_method='silverman')
    x_sigma = np.linspace(sigma_range[0], sigma_range[1], 500)
    kde_vals_sigma = kde_sigma(x_sigma)
    ax2.plot(x_sigma, kde_vals_sigma, 'r-', linewidth=2.5, label='KDE')
    
    # Find mode (peak of KDE)
    mode_idx_sigma = np.argmax(kde_vals_sigma)
    mode_sigma = x_sigma[mode_idx_sigma]
    ax2.axvline(mode_sigma, color='purple', linestyle='-.', linewidth=2,
               label=f'Mode = {mode_sigma:.2f}')
    
    # Mark true value
    if true_sigma is not None:
        ax2.axvline(true_sigma, color='red', linestyle='--', linewidth=2.5,
                   label=f'True σ = {true_sigma}')
        # Add probability density at true value
        density_at_true = kde_sigma(true_sigma)[0]
        ax2.plot(true_sigma, density_at_true, 'ro', markersize=10, zorder=5)
    
    # Statistics
    mean_sigma = selected_sigma.mean()
    median_sigma = np.median(selected_sigma)
    std_sigma = selected_sigma.std()
    
    ax2.axvline(mean_sigma, color='green', linestyle=':', linewidth=2,
               label=f'Mean = {mean_sigma:.2f}')
    ax2.axvline(median_sigma, color='orange', linestyle=':', linewidth=2,
               label=f'Median = {median_sigma:.2f}')
    
    # Labels
    ax2.set_xlabel('sigma', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=14, fontweight='bold')
    ax2.set_title(f'Sigma Distribution\n(mean={mean_sigma:.2f}, std={std_sigma:.2f})',
                 fontsize=15, fontweight='bold')
    ax2.set_xlim(sigma_range)
    ax2.legend(loc='best', fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Overall title
    fig.suptitle(f'{title}\n{percentile}% closest samples ({n_selected:,} total)',
                fontsize=17, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save
    import os
    os.makedirs(save_path, exist_ok=True)
    save_file = f"{save_path}{title.replace(' ', '_').replace('-', '')}_kde_p{percentile}.png"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to: {save_file}")
    plt.close()
 
 
def plot_histograms_multiple_percentiles(
    file_dists='dists.csv',
    file_R0='R0.csv',
    file_sigma='sigma.csv',
    percentiles=[10, 5, 2, 1, 0.5, 0.1],
    true_R0=None,
    true_sigma=None,
    title="ABC Posterior - Multiple Percentiles",
    save_path='../../experimental_data/from_260703/',
    R0_range=(1, 8),
    sigma_range=(0.2, 1.0),
    bins=30
):
    """
    Create multi-panel plot showing histograms with KDE for different percentiles.
    
    Each row shows one percentile:
    - Left column: R0 distribution
    - Right column: sigma distribution
    """
    
    # Load data
    distances = pd.read_csv(file_dists, header=None).values.ravel()
    data_R0 = pd.read_csv(file_R0, header=None).values.ravel()
    data_sigma = pd.read_csv(file_sigma, header=None).values.ravel()
    
    print(f"Loaded {len(distances):,} samples")
    
    # Create figure with correct number of rows
    n_percentiles = len(percentiles)
    fig, axes = plt.subplots(n_percentiles, 2, figsize=(16, 4 * n_percentiles))
    
    # Handle case where there's only one percentile
    if n_percentiles == 1:
        axes = axes.reshape(1, -1)
    
    for idx, p in enumerate(percentiles):
        row = idx
        
        # Select samples
        threshold = np.percentile(distances, p)
        selected_indices = distances <= threshold
        
        selected_R0 = data_R0[selected_indices]
        selected_sigma = data_sigma[selected_indices]
        n_selected = len(selected_R0)
        
        print(f"Percentile {p}%: {n_selected:,} samples")
        
        # R0 plot (left column)
        ax_R0 = axes[row, 0]
        
        # Histogram
        ax_R0.hist(selected_R0, bins=bins, density=True, alpha=0.6,
                  color='skyblue', edgecolor='black', linewidth=0.5)
        
        # KDE
        if len(selected_R0) > 1:
            kde_R0 = stats.gaussian_kde(selected_R0, bw_method='silverman')
            x_R0 = np.linspace(R0_range[0], R0_range[1], 300)
            kde_vals_R0 = kde_R0(x_R0)
            ax_R0.plot(x_R0, kde_vals_R0, 'b-', linewidth=2)
            
            # Find mode (peak of KDE)
            mode_idx_R0 = np.argmax(kde_vals_R0)
            mode_R0 = x_R0[mode_idx_R0]
            ax_R0.axvline(mode_R0, color='purple', linestyle='-.', linewidth=1.5,
                         label=f'Mode={mode_R0:.2f}')
        else:
            mode_R0 = selected_R0.mean()
        
        # True value
        if true_R0 is not None:
            ax_R0.axvline(true_R0, color='red', linestyle='--', linewidth=2,
                         label=f'True={true_R0}')
        
        # Mean
        mean_R0 = selected_R0.mean()
        ax_R0.axvline(mean_R0, color='green', linestyle=':', linewidth=1.5,
                     label=f'Mean={mean_R0:.2f}')
        
        ax_R0.set_xlabel('R0', fontsize=12)
        ax_R0.set_ylabel('Density', fontsize=12)
        ax_R0.set_title(f'{p}% - R0 (n={n_selected:,})',
                       fontsize=12, fontweight='bold')
        ax_R0.set_xlim(R0_range)
        ax_R0.legend(loc='best', fontsize=9)
        ax_R0.grid(True, alpha=0.3)
        
        # Sigma plot (right column)
        ax_sigma = axes[row, 1]
        
        # Histogram
        ax_sigma.hist(selected_sigma, bins=bins, density=True, alpha=0.6,
                     color='lightcoral', edgecolor='black', linewidth=0.5)
        
        # KDE
        if len(selected_sigma) > 1:
            kde_sigma = stats.gaussian_kde(selected_sigma, bw_method='silverman')
            x_sigma = np.linspace(sigma_range[0], sigma_range[1], 300)
            kde_vals_sigma = kde_sigma(x_sigma)
            ax_sigma.plot(x_sigma, kde_vals_sigma, 'r-', linewidth=2)
            
            # Find mode (peak of KDE)
            mode_idx_sigma = np.argmax(kde_vals_sigma)
            mode_sigma = x_sigma[mode_idx_sigma]
            ax_sigma.axvline(mode_sigma, color='purple', linestyle='-.', linewidth=1.5,
                            label=f'Mode={mode_sigma:.2f}')
        else:
            mode_sigma = selected_sigma.mean()
        
        # True value
        if true_sigma is not None:
            ax_sigma.axvline(true_sigma, color='red', linestyle='--', linewidth=2,
                            label=f'True={true_sigma}')
        
        # Mean
        mean_sigma = selected_sigma.mean()
        ax_sigma.axvline(mean_sigma, color='green', linestyle=':', linewidth=1.5,
                        label=f'Mean={mean_sigma:.2f}')
        
        ax_sigma.set_xlabel('sigma', fontsize=12)
        ax_sigma.set_ylabel('Density', fontsize=12)
        ax_sigma.set_title(f'{p}% - Sigma (n={n_selected:,})',
                          fontsize=12, fontweight='bold')
        ax_sigma.set_xlim(sigma_range)
        ax_sigma.legend(loc='best', fontsize=9)
        ax_sigma.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save
    import os
    os.makedirs(save_path, exist_ok=True)
    save_file = f"{save_path}{title.replace(' ', '_').replace('-', '')}_kde_multi.png"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved to: {save_file}")
    plt.close()
 

def plot_dots_percentile_colors(file_dists='dists.csv', 
                                file_R0='R0.csv', 
                                file_sigma='sigma.csv',
                                percentile=1,
                                true_R0=None,
                                true_sigma=None,
                                title="R0 vs sigma - Color by Distance",
                                save_path='../../experimental_data/from_260312/',
                                xlim=(1, 8),
                                ylim=(0.2, 1.0),
                                cmap='viridis_r'):  # 'viridis_r' = purple (close) to yellow (far)
    """
    Plot R0 vs sigma with colors representing distance from standard point.
    
    Parameters:
    -----------
    file_dists : str
        CSV file with distances (single column)
    file_R0 : str
        CSV file with R0 samples
    file_sigma : str
        CSV file with sigma samples
    percentile : float
        Percentile threshold (e.g., 1 means keep closest 1%)
    true_R0 : float, optional
        True R0 value to mark on plot
    true_sigma : float, optional
        True sigma value to mark on plot
    title : str
        Figure title
    save_path : str
        Directory to save figure
    xlim : tuple
        X-axis limits (R0 range)
    ylim : tuple
        Y-axis limits (sigma range)
    cmap : str
        Colormap name. Options:
        - 'viridis_r': purple (close) → yellow (far)
        - 'RdYlGn_r': red (far) → green (close)
        - 'coolwarm': blue (close) → red (far)
        - 'plasma_r': purple (close) → yellow (far)
    """
    
    # Load data
    print("Loading data...")
    distances = pd.read_csv(file_dists, header=None).values.ravel()
    data_R0 = pd.read_csv(file_R0, header=None).values.ravel()
    data_sigma = pd.read_csv(file_sigma, header=None).values.ravel()
    
    print(f"✓ Loaded {len(distances):,} samples")
    print(f"  Distance range: [{distances.min():.6f}, {distances.max():.6f}]")
    
    # Select samples based on percentile
    threshold = np.percentile(distances, percentile)
    selected_indices = np.where(distances <= threshold)[0]
    n_selected = len(selected_indices)
    
    print(f"\nPercentile {percentile}%:")
    print(f"  Threshold: {threshold:.6f}")
    print(f"  Selected: {n_selected:,} samples ({100*n_selected/len(distances):.2f}%)")
    
    # Get selected data
    selected_R0 = data_R0[selected_indices]
    selected_sigma = data_sigma[selected_indices]
    selected_dist = distances[selected_indices]
    
    print(f"  Distance range of selected: [{selected_dist.min():.6f}, {selected_dist.max():.6f}]")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create scatter plot with color mapping
    scatter = ax.scatter(
        selected_R0, 
        selected_sigma, 
        c=selected_dist,  # Color by distance
        cmap=cmap,
        s=50,  # Point size
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5,
        vmin=selected_dist.min(),
        vmax=selected_dist.max()
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Distance from Standard Point', pad=0.02)
    cbar.ax.tick_params(labelsize=11)
    
    # Mark true values if provided
    if true_R0 is not None and true_sigma is not None:
        ax.scatter(true_R0, true_sigma, s=400, c='red', marker='*',
                  edgecolors='black', linewidths=3, label='True value', zorder=10)
        ax.legend(loc='best', fontsize=12, framealpha=0.9)
    
    # Set fixed ranges
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Labels and title
    ax.set_xlabel('R0', fontsize=14, fontweight='bold')
    ax.set_ylabel('sigma', fontsize=14, fontweight='bold')
    ax.set_title(
        f'{title}\n'
        f'{percentile}% closest to standard point '
        f'({n_selected:,} samples, dist ≤ {threshold:.4f})',
        fontsize=16, fontweight='bold', pad=20
    )
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics text box
    textstr = f'Distance Range:\n  Min: {selected_dist.min():.6f}\n  Max: {selected_dist.max():.6f}\n  Mean: {selected_dist.mean():.6f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save figure
    import os
    os.makedirs(save_path, exist_ok=True)
    save_file = f"{save_path}{title.replace(' ', '_').replace('-', '')}_p{percentile}.png"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved to: {save_file}")
    plt.close()
 

def plot_dots_multiple_percentiles_colors(
    file_dists='dists.csv',
    file_R0='R0.csv',
    file_sigma='sigma.csv',
    percentiles=[10, 5, 2, 1, 0.5, 0.1],
    true_R0=None,
    true_sigma=None,
    title="ABC Posterior - Distance-Colored",
    save_path='../../experimental_data/from_260312/',
    xlim=(1, 8),
    ylim=(0.2, 1.0),
    cmap='viridis_r',
    vmin=None,  # Manual color scale min (optional)
    vmax=None   # Manual color scale max (optional)
):
    """
    Create a 6-panel plot with different percentiles, all color-coded by distance.
    
    Parameters:
    -----------
    vmin : float, optional
        Manual minimum for color scale. If None, uses tightest percentile's min.
        Use this to ensure consistent colors across multiple plots.
    vmax : float, optional
        Manual maximum for color scale. If None, uses tightest percentile's max.
        Use this to ensure consistent colors across multiple plots.
    """
    
    # Load data
    print("Loading data...")
    distances = pd.read_csv(file_dists, header=None).values.ravel()
    data_R0 = pd.read_csv(file_R0, header=None).values.ravel()
    data_sigma = pd.read_csv(file_sigma, header=None).values.ravel()
    
    print(f"✓ Loaded {len(distances):,} samples")
    
    # Create figure with 6 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    # First pass: determine the appropriate color range
    if vmin is not None and vmax is not None:
        # Use manual color range
        color_vmin = vmin
        color_vmax = vmax
        print(f"\nUsing MANUAL color scale: [{color_vmin:.6f}, {color_vmax:.6f}]")
    else:
        # Auto-detect from the tightest percentile (smallest p)
        min_p = min(percentiles)
        threshold_min = np.percentile(distances, min_p)
        indices_min = np.where(distances <= threshold_min)[0]
        dist_min_selection = distances[indices_min]
        
        color_vmin = dist_min_selection.min()
        color_vmax = dist_min_selection.max()
        
        print(f"\nAuto-detected color scale: [{color_vmin:.6f}, {color_vmax:.6f}]")
        print(f"(Based on {min_p}% percentile selection)")
        print(f"💡 Tip: To ensure consistent colors across multiple plots, use:")
        print(f"   vmin={color_vmin:.6f}, vmax={color_vmax:.6f}")
    
    print()
    
    for idx, p in enumerate(percentiles):
        ax = axes[idx]
        
        # Select samples
        threshold = np.percentile(distances, p)
        selected_indices = np.where(distances <= threshold)[0]
        n_selected = len(selected_indices)
        
        selected_R0 = data_R0[selected_indices]
        selected_sigma = data_sigma[selected_indices]
        selected_dist = distances[selected_indices]
        
        print(f"Percentile {p}%: {n_selected:,} samples")
        print(f"  Threshold: {threshold:.6f}")
        print(f"  Distance range: [{selected_dist.min():.6f}, {selected_dist.max():.6f}]")
        
        # Scatter plot with SHARED color scale
        scatter = ax.scatter(
            selected_R0,
            selected_sigma,
            c=selected_dist,
            cmap=cmap,
            s=30,
            alpha=0.6,
            edgecolors='none',
            vmin=color_vmin,  # Shared min
            vmax=color_vmax   # Shared max
        )
        
        # Mark true values
        if true_R0 is not None and true_sigma is not None:
            ax.scatter(true_R0, true_sigma, s=300, c='red', marker='*',
                      edgecolors='black', linewidths=2, zorder=10)
        
        # Set fixed ranges
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        # Labels
        ax.set_xlabel('R0', fontsize=12)
        ax.set_ylabel('sigma', fontsize=12)
        ax.set_title(
            f'{p}% closest\n({n_selected:,} samples, dist ≤ {threshold:.4f})',
            fontsize=13, fontweight='bold'
        )
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add shared colorbar on the right
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    norm = Normalize(vmin=color_vmin, vmax=color_vmax)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f'Distance from Standard Point\n[{color_vmin:.4f}, {color_vmax:.4f}]', 
                  fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)
    
    # Overall title
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    
    # Save
    import os
    os.makedirs(save_path, exist_ok=True)
    save_file = f"{save_path}{title.replace(' ', '_').replace('-', '')}.png"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved to: {save_file}")
    plt.close()


def plot_dots_multiple_percentiles(file_dists='dists.csv', 
                                   file_R0='R0.csv', 
                                   file_sigma='sigma.csv',
                                   percentiles=[100, 80, 60, 40, 20, 10],
                                   true_R0=None,
                                   true_sigma=None,
                                   title="R0 vs sigma - Multiple Percentiles",
                                   save_path='../../experimental_data/from_260312/',
                                   xlim=(1, 8),        # Fixed x-axis range
                                   ylim=(0.2, 1.0)):   # Fixed y-axis range
    """
    Create 6 subplots showing R0 vs sigma for different distance percentiles.
    
    Parameters:
    -----------
    file_dists : str
        CSV file with distances (single column)
    file_R0 : str
        CSV file with R0 samples
    file_sigma : str
        CSV file with sigma samples
    percentiles : list
        List of percentiles to plot [100, 80, 60, 40, 20, 10]
    true_R0 : float, optional
        True R0 value to mark on plots
    true_sigma : float, optional
        True sigma value to mark on plots
    title : str
        Overall figure title
    save_path : str
        Directory to save figure
    xlim : tuple
        X-axis limits (R0 range), default (1, 8)
    ylim : tuple
        Y-axis limits (sigma range), default (0.2, 1.0)
    """
    # Load data
    distances = pd.read_csv(file_dists, header=None).values.ravel()
    data_R0 = pd.read_csv(file_R0, header=None).values.ravel()
    data_sigma = pd.read_csv(file_sigma, header=None).values.ravel()
    
    print(f"Loaded {len(distances)} samples")
    
    # Create figure with 6 subplots (2 rows x 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()  # Flatten to 1D array for easy indexing
    
    # Plot for each percentile
    for idx, percentile in enumerate(percentiles):
        ax = axes[idx]
        
        # Calculate threshold
        if percentile == 100:
            # Keep all samples
            selected_indices = np.arange(len(distances))
        else:
            threshold = np.percentile(distances, percentile)
            selected_indices = np.where(distances <= threshold)[0]
        
        n_selected = len(selected_indices)
        print(f"Percentile {percentile}: {n_selected} samples selected")
        
        # Select corresponding samples
        selected_R0 = data_R0[selected_indices]
        selected_sigma = data_sigma[selected_indices]
        
        # Scatter plot
        ax.scatter(selected_R0, selected_sigma, alpha=0.5, s=20, c='blue')
        
        # Mark true values if provided
        if true_R0 is not None and true_sigma is not None:
            ax.scatter(true_R0, true_sigma, s=200, c='red', marker='*',
                      edgecolors='black', linewidths=2, label='True value', zorder=5)
            ax.legend(loc='best')
        
        # Set fixed axis ranges
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        # Labels and title
        ax.set_xlabel('R0', fontsize=11)
        ax.set_ylabel('sigma', fontsize=11)
        ax.set_title(f'{percentile}th percentile\n({n_selected} samples, {n_selected/len(distances)*100:.1f}%)',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save figure
    import os
    os.makedirs(save_path, exist_ok=True)
    save_file = f"{save_path}{title.replace(' ', '_').replace('-', '')}.png"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"Saved to {save_file}")
    plt.close()


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


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    # 1. Calculate distances with Euclidean metric
    result = analyze_distances(
        filepath="../../experimental_data/from_260703/summary_stats_normalized.csv",
        standard_point=(0.08330164, 0.1184396, 0.0732722, 0.33333333),  # Fixed: added commas 
        weights=(0.9, 0.9, 0.1, 0.1),  # Fixed: added commas
        metric="euclidean",
        p=3,
        output_path="../../experimental_data/from_260703/dists_observations_recal.csv"
    )
    
    # 2. Plot with higher percentiles (10%, 5%, 4%, 3%, 2%, 1%)
    # plot_dots_multiple_percentiles(
    #     file_dists='../../experimental_data/from_260703/dists_observations_recal.csv',
    #     file_R0='../../experimental_data/from_260703/R0.csv',
    #     file_sigma='../../experimental_data/from_260703/sigma.csv',
    #     percentiles=[10, 5, 4, 3, 2, 1],
    #     true_R0=0.0,
    #     true_sigma=0.0,
    #     title="R0 vs sigma - Multiple Percentiles-euclidean1",
    #     save_path="../../figures/from_260703/ppc/observations/",
    #     xlim=(1, 8),      # R0 range
    #     ylim=(0.2, 1.0)   # sigma range
    # )

    plot_dots_multiple_percentiles_colors(
        file_dists='../../experimental_data/from_260703/dists_observations_recal.csv',
        file_R0='../../experimental_data/from_260703/R0.csv',
        file_sigma='../../experimental_data/from_260703/sigma.csv',
        percentiles=[10, 5, 4, 3, 2, 1],
        true_R0=None,
        true_sigma=None,
        title="ABC Posterior - Distance Colored1",
        save_path="../../figures/from_260703/ppc/observations/",
        xlim=(1, 4),
        ylim=(0.2, 1.0),
        cmap='viridis_r', 
        vmin=0.021266, 
        vmax=0.047695
    )
    
    # 3. Plot with lower percentiles (2%, 1%, 0.5%, 0.4%, 0.2%, 0.1%)
    # plot_dots_multiple_percentiles(
    #     file_dists='../../experimental_data/from_260703/dists_observations_recal.csv',
    #     file_R0='../../experimental_data/from_260703/R0.csv',
    #     file_sigma='../../experimental_data/from_260703/sigma.csv',
    #     percentiles=[2, 1, 0.5, 0.4, 0.2, 0.1],
    #     true_R0=0.0,
    #     true_sigma=0.0,
    #     title="R0 vs sigma - Multiple Percentiles-euclidean",
    #     save_path="../../figures/from_260703/ppc/observations/",
    #     xlim=(1, 8),      # R0 range
    #     ylim=(0.2, 1.0)   # sigma range
    # )
    

    plot_dots_multiple_percentiles_colors(
        file_dists='../../experimental_data/from_260703/dists_observations_recal.csv',
        file_R0='../../experimental_data/from_260703/R0.csv',
        file_sigma='../../experimental_data/from_260703/sigma.csv',
        percentiles=[2, 1, 0.5, 0.1, 0.05, 0.01],
        true_R0=None,
        true_sigma=None,
        title="ABC Posterior - Distance Colored",
        save_path="../../figures/from_260703/ppc/observations/",
        xlim=(1, 4),
        ylim=(0.2, 1.0),
        cmap='viridis_r', 
        vmin=0.021266, 
        vmax=0.047695
    )

    plot_dots_percentile_colors(
            file_dists='../../experimental_data/from_260703/dists_observations_recal.csv',
            file_R0='../../experimental_data/from_260703/R0.csv',
            file_sigma='../../experimental_data/from_260703/sigma.csv',
            percentile=1,  # Show closest 1%
            true_R0=None,
            true_sigma=None,
            title="ABC Posterior - Top 1%",
            save_path="../../figures/from_260703/ppc/observations/",
            xlim=(1, 4),
            ylim=(0.2, 1.0),
            cmap='viridis_r'  # Purple (close) to yellow (far)
        )

    plot_dots_percentile_colors(
            file_dists='../../experimental_data/from_260703/dists_observations_recal.csv',
            file_R0='../../experimental_data/from_260703/R0.csv',
            file_sigma='../../experimental_data/from_260703/sigma.csv',
            percentile=0.1,  # Show closest 0.1%
            true_R0=None,
            true_sigma=None,
            title="ABC Posterior - Top 0.1%",
            save_path="../../figures/from_260703/ppc/observations/",
            xlim=(1, 4),
            ylim=(0.2, 1.0),
            cmap='viridis_r'  # Purple (close) to yellow (far)
        )

    plot_dots_percentile_colors(
            file_dists='../../experimental_data/from_260703/dists_observations_recal.csv',
            file_R0='../../experimental_data/from_260703/R0.csv',
            file_sigma='../../experimental_data/from_260703/sigma.csv',
            percentile=0.05,  # Show closest 0.05%
            true_R0=None,
            true_sigma=None,
            title="ABC Posterior - Top 0.05%",
            save_path="../../figures/from_260703/ppc/observations/",
            xlim=(1, 4),
            ylim=(0.2, 1.0),
            cmap='viridis_r'  # Purple (close) to yellow (far)
        )

    plot_dots_percentile_colors(
            file_dists='../../experimental_data/from_260703/dists_observations_recal.csv',
            file_R0='../../experimental_data/from_260703/R0.csv',
            file_sigma='../../experimental_data/from_260703/sigma.csv',
            percentile=0.01,  # Show closest 0.01%
            true_R0=None,
            true_sigma=None,
            title="ABC Posterior - Top 0.01%",
            save_path="../../figures/from_260703/ppc/observations/",
            xlim=(1, 4),
            ylim=(0.2, 1.0),
            cmap='viridis_r'  # Purple (close) to yellow (far)
        )

        # Example 1: Single percentile with detailed KDE
    plot_histograms_with_kde(
            file_dists='../../experimental_data/from_260703/dists_observations_recal.csv',
            file_R0='../../experimental_data/from_260703/R0.csv',
            file_sigma='../../experimental_data/from_260703/sigma.csv',
            percentile=0.01,
            true_R0=None,
            true_sigma=None,
            title="ABC Posterior - R0=None, sigma=None",
            save_path="../../figures/from_260703/ppc/observations/",
            R0_range=(1, 4),
            sigma_range=(0.2, 1.0),
            bins=50
        )

        # Example 2: Multiple percentiles comparison
    plot_histograms_multiple_percentiles(
            file_dists='../../experimental_data/from_260703/dists_observations_recal.csv',
            file_R0='../../experimental_data/from_260703/R0.csv',
            file_sigma='../../experimental_data/from_260703/sigma.csv',
            percentiles=[2, 1, 0.5, 0.1, 0.05, 0.01],
            true_R0=None,
            true_sigma=None,
            title="ABC Posterior Evolution - R0=None, sigma=None",
            save_path="../../figures/from_260703/ppc/observations/",
            R0_range=(1, 4),
            sigma_range=(0.2, 1.0),
            bins=30
        )

    print("\n✅ All plots generated successfully!")