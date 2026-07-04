# file name: generate_standard_point_fixed.py
# Generate standard point (observed data) from true parameter values

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
import functions_list_260528 as functions_list
import summary_stats_elms_260528 as ss
import hashlib
import time

start = time.perf_counter()

core_params_num = 2  # core params: R0 and sigma

# Fixed parameters
DurationSimulation = 20.0
Nstrains = 42
omega = 0.2
x = 10.0
Cperweek = 34.53
Nagents = 2500
alpha = 0.007 * Nagents        # migration rate: 3.0
AgeDeath = 71.0

if core_params_num == 2:
    Dimmunity = 10.0 * 52.14
    fixed_params = np.array([DurationSimulation, Nstrains, Dimmunity, omega,
                         x, Cperweek, Nagents, alpha, AgeDeath], dtype=float)
else:
    raise ValueError('Invalid core params num')


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


# Seed from theta
def seed_from_theta(theta, master_seed: int = 123):
    th = np.asarray(theta, np.float64).ravel()
    b = th.tobytes() + np.uint64(master_seed).tobytes()
    return int.from_bytes(hashlib.sha1(b).digest()[:8], 'little')


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


# ============================================================================
# GENERATE STANDARD POINTS FOR DIFFERENT TRUE VALUES
# ============================================================================

# Define true parameter values to test
R0_values = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
sigma_values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.94, 0.975]

observations = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 3, 1, 5, 2, 4, 2, 0, 0, 3, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 0, 0, 5, 3, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 2, 1, 0, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 13, 7, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 2, 1, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 3, 1, 3, 0, 2, 1, 2, 0, 1],
    [0, 0, 0, 0, 0, 0, 2, 3, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 4, 1, 1, 1, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 3, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 3, 2, 3, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 1, 4, 1, 0, 1, 4, 0, 2, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 3, 1, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 2, 2, 5, 4, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0]
]

# Convert to numpy array
observations = np.array(observations)

print(f"Shape: {observations.shape}")  # Should be (42, 23)

print("="*70)
print("GENERATING STANDARD POINTS")
print("="*70)

# Store results
results = []

# Calculate summary statistics (this is observations)
s_obs = summary_stats(observations)
print("These summary stats from the actual observations.")
print(f"  Standard point (raw): {s_obs}")

results.append({
            'R0': 0.0,
            'sigma': 0.0,
            'avg_prev_obs': s_obs[0],
            'var_prev_obs': s_obs[1],
            'avg_npmi_obs': s_obs[2],
            'div_all_isolates_obs': s_obs[3]
        })

for sigma_val in sigma_values:
    for R0_val in R0_values:
        print(f"\nGenerating standard point for R0={R0_val}, sigma={sigma_val}")
        
        # Run simulation with true values
        theta_true = np.array([R0_val, sigma_val], float)
        
        # Generate synthetic observed data
        Tdry = simulate_prevalence_v5_numba(theta_true, fixed_params, core_params_num, seed=123)
        # print(Tdry)
        
        # Calculate summary statistics (this is your standard point)
        s_obs = summary_stats(Tdry)
        
        print(f"  Standard point (raw): {s_obs}")
        
        # Test reproducibility
        Tdry_check = simulate_prevalence_v5_numba(theta_true, fixed_params, core_params_num, seed=123)
        is_reproducible = np.allclose(Tdry, Tdry_check)
        print(f"  Reproducible: {is_reproducible} ✓" if is_reproducible else f"  Reproducible: {is_reproducible} ✗")
        
        results.append({
            'R0': R0_val,
            'sigma': sigma_val,
            'avg_prev_obs': s_obs[0],
            'var_prev_obs': s_obs[1],
            'avg_npmi_obs': s_obs[2],
            'div_all_isolates_obs': s_obs[3]
        })

# Convert to DataFrame and save
import pandas as pd
results_df = pd.DataFrame(results)

print("\n" + "="*70)
print("ALL STANDARD POINTS")
print("="*70)
print(results_df.to_string(index=False))

# Save to CSV
output_file = '../../experimental_data/from_260703/standard_points.csv'
results_df.to_csv(output_file, index=False)
print(f"\n✓ Saved to: {output_file}")

# Show specific example (R0=2.5, sigma=0.8)
print("\n" + "="*70)
print("EXAMPLE: R0=2.5, sigma=0.8")
print("="*70)
example = results_df[(results_df['R0'] == 2.5) & (results_df['sigma'] == 0.8)]
if not example.empty:
    row = example.iloc[0]
    print(f"Standard point (raw values):")
    print(f"  avg_prev_obs:         {row['avg_prev_obs']:.8f}")
    print(f"  var_prev_obs:         {row['var_prev_obs']:.8f}")
    print(f"  avg_npmi_obs:         {row['avg_npmi_obs']:.8f}")
    print(f"  div_all_isolates_obs: {row['div_all_isolates_obs']:.8f}")
    
    print(f"\nAs tuple for copy-paste:")
    print(f"standard_point = ({row['avg_prev_obs']:.8f}, {row['var_prev_obs']:.8f}, "
          f"{row['avg_npmi_obs']:.8f}, {row['div_all_isolates_obs']:.8f})")

end = time.perf_counter()
print(f"\n⏱️  Elapsed: {end - start:.2f} seconds")
print("="*70)