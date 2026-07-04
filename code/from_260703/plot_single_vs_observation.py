"""
plot_single_vs_observation.py

Plots one selected simulation (clean_id = 908653) alongside the observed data,
using the same heatmap and scatter style as plot_observation_both().

Outputs (saved to save_path):
  - comparison_heatmap_id908653.png   (side-by-side heatmaps)
  - comparison_scatter_id908653.png   (side-by-side scatter plots)
"""

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIG — adjust paths as needed
# ============================================================
CLEAN_ID   = 908653
VMIN, VMAX = 0, 20

file_R0      = '../../experimental_data/from_260703/R0.csv'
file_sigma   = '../../experimental_data/from_260703/sigma.csv'
file_dists   = '../../experimental_data/from_260703/dists_observations_recal.csv'
valid_idx_f  = '../../experimental_data/from_260703/valid_indices.csv'
sim_banks_dir= '../../experimental_data/from_260703/simulation_banks'
save_path    = '../../figures/from_260703/ppc/observations/'

# ============================================================
# OBSERVED DATA  (42 x 23)
# ============================================================
observations = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,2,0,3,1,5,2,4,2,0,0,3,0],
    [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
    [0,0,0,2,0,0,5,3,1,0,0,0,0,2,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,1,0,2,1,0,2,0,1,2,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,13,7,3],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,2,1,2,0,0,0,0,0,1,0,0,0,2,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,5,3,1,3,0,2,1,2,0,1],
    [0,0,0,0,0,0,2,3,2,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,2,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,4,1,1,1,0,0,1,2,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,3,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,2,0,3,2,3,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,1,1,4,1,0,1,4,0,2,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
    [0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,2,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,3,3,1,2,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,2],
    [0,0,0,0,0,0,0,0,2,1,0,0,0,0,2,2,5,4,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,1,1,0],
])

# ============================================================
# LOAD SIMULATION FOR CLEAN_ID
# ============================================================
print(f"Loading simulation for clean_id = {CLEAN_ID} ...")


def build_file_index(sim_files):
    """Build cumulative sample-count boundaries from actual file sizes."""
    counts = []
    for f in sim_files:
        with h5py.File(f, 'r') as h5f:
            counts.append(len(h5f['R0']))
    cum = np.cumsum(counts)  # cum[i] = total samples in files[0..i]
    return counts, cum

def locate_sample(original_id, sim_files, cum):
    """Find which file and local index a global original_id belongs to."""
    file_idx = int(np.searchsorted(cum, original_id, side='right'))
    if file_idx >= len(sim_files):
        raise IndexError(f"original_id {original_id} out of range")
    prev_total = cum[file_idx - 1] if file_idx > 0 else 0
    local_idx = original_id - prev_total
    return file_idx, local_idx


valid_indices = np.loadtxt(valid_idx_f, dtype=int)
original_id   = valid_indices[CLEAN_ID]

R0_val    = pd.read_csv(file_R0,    header=None).values.ravel()[CLEAN_ID]
sigma_val = pd.read_csv(file_sigma, header=None).values.ravel()[CLEAN_ID]
dist_val  = pd.read_csv(file_dists, header=None).values.ravel()[CLEAN_ID]

sim_files = sorted(Path(sim_banks_dir).glob('simulation_bank_part_*.h5'))
with h5py.File(sim_files[0], 'r') as f:
    samples_per_file = len(f['R0'])

counts, cum = build_file_index(sim_files)   # compute once, reuse
file_idx, local_idx = locate_sample(original_id, sim_files, cum)

with h5py.File(sim_files[file_idx], 'r') as f:
    simulation = f['simulations'][local_idx].astype(float)  # (42, 23)

sim_label = (f"Simulation case\n"
             f"$R_0={R0_val:.3f}$, $\\sigma={sigma_val:.3f}$, dist={dist_val:.5f}")
obs_label = "Observed data"

print(f"  Original ID : {original_id}")
print(f"  R0={R0_val:.3f}, sigma={sigma_val:.3f}, dist={dist_val:.5f}")
print(f"  Simulation shape: {simulation.shape}")

Path(save_path).mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER — heatmap panel
# ============================================================
def heatmap_panel(ax, matrix, title, vmin, vmax):
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd',
                   interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.set_xlabel('Time Point', fontsize=11)
    ax.set_ylabel('Strain',     fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(np.arange(0, 23, 5))
    ax.set_yticks(np.arange(0, 42, 5))
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    return im


# ============================================================
# HELPER — scatter panel
# ============================================================
def scatter_panel(ax, matrix, title, vmin, vmax):
    strains, times = np.where(matrix > 0)
    prevalences    = matrix[strains, times]
    sc = ax.scatter(times, strains,
                    c=prevalences, s=prevalences * 10,
                    cmap='YlOrRd', alpha=0.7,
                    edgecolors='black', linewidths=0.3,
                    vmin=vmin, vmax=vmax)
    ax.set_xlabel('Time Point', fontsize=11)
    ax.set_ylabel('Strain',     fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(-0.5, 22.5)
    ax.set_ylim(-0.5, 41.5)
    ax.set_xticks(np.arange(0, 23, 5))
    ax.set_yticks(np.arange(0, 42, 5))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    return sc


# ============================================================
# FIGURE 1 — HEATMAP COMPARISON
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Posterior Predictive Check — Heatmap',
             fontsize=15, fontweight='bold')

im0 = heatmap_panel(axes[0], observations, obs_label, VMIN, VMAX)
im1 = heatmap_panel(axes[1], simulation,   sim_label, VMIN, VMAX)

fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.91, 0.12, 0.02, 0.75])
fig.colorbar(im1, cax=cbar_ax, label='Isolate count')

plt.tight_layout(rect=[0, 0, 0.88, 1])
out = f"{save_path}comparison_heatmap_id{CLEAN_ID}.png"
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")


# ============================================================
# FIGURE 2 — SCATTER COMPARISON
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Posterior Predictive Check — Scatter',
             fontsize=15, fontweight='bold')

sc0 = scatter_panel(axes[0], observations, obs_label, VMIN, VMAX)
sc1 = scatter_panel(axes[1], simulation,   sim_label, VMIN, VMAX)

fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.91, 0.12, 0.02, 0.75])
fig.colorbar(sc1, cax=cbar_ax, label='Isolate count')

plt.tight_layout(rect=[0, 0, 0.88, 1])
out = f"{save_path}comparison_scatter_id{CLEAN_ID}.png"
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

print("\nDone. Both figures saved.")