"""
plot_single_synthetic_vs_observation.py

Plots one selected simulation (clean_id = 000000) alongside the observed data,
using the same heatmap and scatter style as plot_observation_both().

Outputs (saved to save_path):
  - comparison_heatmap_id000000.png   (side-by-side heatmaps)
  - comparison_scatter_id000000.png   (side-by-side scatter plots)
"""

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import hashlib
from numpy.random import default_rng

# Import your existing functions
import functions_list_260528 as functions_list

# ============================================================
# CONFIG — adjust paths as needed
# ============================================================
CLEAN_ID   = 000000
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

core_params_num = 2  # core params: R0 and sigma
# core_params_num = 3  # core params: R0, sigma, and Dimmunity

# fixed parameters
DurationSimulation = 20.0     # years: 20.0
Nstrains = 42       # number of strains: 42
omega = 0.2     # immunity cross strains: 0.1
x = 10.0        #
Cperweek = 34.53    #
Nagents = 2500      # number of agents
alpha = 0.007 * Nagents        # migration rate: 3.0
AgeDeath = 71.0     # life expectancy

if core_params_num == 2:
    Dimmunity = 10.0 * 52.14
    fixed_params = np.array([DurationSimulation, Nstrains, Dimmunity, omega,
                         x, Cperweek, Nagents, alpha, AgeDeath], dtype=float)


# function: seed_from_theta
def seed_from_theta(theta, master_seed: int = 123):
    th = np.asarray(theta, np.float64).ravel()
    b = th.tobytes() + np.uint64(master_seed).tobytes()
    return int.from_bytes(hashlib.sha1(b).digest()[:8], 'little')


# function: build parameters
def build_params(theta, fixed_params, core_params_num):
    theta = np.asarray(theta, float).ravel()
    if theta.size != core_params_num:
        raise ValueError(f"theta must be length-{core_params_num}, got {np.shape(theta)}")
    if core_params_num == 2:
        R0, sigma = float(theta[0]), float(theta[1])
        return np.array([fixed_params[0], fixed_params[1], fixed_params[2], sigma,
                         fixed_params[3], fixed_params[4], fixed_params[5], fixed_params[6],
                         fixed_params[7], fixed_params[8], R0
                         ], dtype=float)
    elif core_params_num == 3:
        R0, sigma, Dimmunity = float(theta[0]), float(theta[1]), float(theta[2])
        return np.array([fixed_params[0], fixed_params[1], Dimmunity, sigma,
                         fixed_params[2], fixed_params[3], fixed_params[4], fixed_params[5],
                         fixed_params[6], fixed_params[7], R0
                         ], dtype=float)
    else:
        raise ValueError('Invalid core params num')

def simulate_prevalence_v5_numba(theta, fixed_params, core_params_num, seed):
    seed = seed_from_theta(theta, master_seed=seed)
    rng = default_rng(seed)
    params = build_params(theta, fixed_params, core_params_num)
    AC, IMM, _ = functions_list.initialise_agents_v5(params, rng=rng)
    SSPrev_selected, SSPrev, AIBKS = functions_list.simulator_v6_numba(
        AC, IMM, params, 0, 1, seed=seed
    )
    return SSPrev_selected.astype(float)

R0_val = 1.5276
sigma_val = 0.9598
y_sim = simulate_prevalence_v5_numba(
        [R0_val, sigma_val],
        fixed_params,
        core_params_num,
        seed=123
    )

# sim_label = (f"Simulation case\n"
#              f"$R_0={R0_val:.3f}$, $\\sigma={sigma_val:.3f}$, dist={dist_val:.5f}")
sim_label = (f"Simulation case\n"
             f"$R_0={R0_val:.3f}$, $\\sigma={sigma_val:.3f}$")

obs_label = "Observed data"

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
im1 = heatmap_panel(axes[1], y_sim,   sim_label, VMIN, VMAX)

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
sc1 = scatter_panel(axes[1], y_sim,   sim_label, VMIN, VMAX)

fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.91, 0.12, 0.02, 0.75])
fig.colorbar(sc1, cax=cbar_ax, label='Isolate count')

plt.tight_layout(rect=[0, 0, 0.88, 1])
out = f"{save_path}comparison_scatter_id{CLEAN_ID}.png"
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

print("\nDone. Both figures saved.")