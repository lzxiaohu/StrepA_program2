# file name: simulations_bank.py


# Packages:
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
import time
import logging
import sys
import hashlib
from numpy.random import default_rng

# Import your existing functions
import functions_list_260528 as functions_list


start = time.perf_counter()


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
# R0: updated parameter (Basic reproductive number)
# Sigma: updated parameter ()

if core_params_num == 2:
    Dimmunity = 10.0 * 52.14
    fixed_params = np.array([DurationSimulation, Nstrains, Dimmunity, omega,
                         x, Cperweek, Nagents, alpha, AgeDeath], dtype=float)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)


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


# function: seed_from_theta
def seed_from_theta(theta, master_seed: int = 123):
    th = np.asarray(theta, np.float64).ravel()
    b = th.tobytes() + np.uint64(master_seed).tobytes()
    return int.from_bytes(hashlib.sha1(b).digest()[:8], 'little')


# function: prior_value_2params
def prior_value_2params(R0_range, sigma_range, rng):
    # intercept a: around distance range
    R0_sel = rng.uniform(R0_range[0], R0_range[1])
    # slope b: often negative
    sigma_sel = rng.uniform(sigma_range[0], sigma_range[1])

    return R0_sel, sigma_sel

# function: simulate_prevalence_v5_numba
def simulate_prevalence_v5_numba(theta, fixed_params, core_params_num, seed):
    #
    #
    seed = seed_from_theta(theta, master_seed=seed)
    rng = default_rng(seed)
    params = build_params(theta, fixed_params, core_params_num)
    AC, IMM, _ = functions_list.initialise_agents_v5(params, rng=rng)

    # call the reproducible simulator that uses only this seed
    SSPrev_selected, SSPrev, AIBKS = functions_list.simulator_v6_numba(
        AC, IMM, params, 0, 1, seed=seed
    )

    # Option A: return the Nstrain * 23 matrix (strain × selected times)
    return SSPrev_selected.astype(float)


def simulate_prevalence_v5_numba(theta, fixed_params, core_params_num, seed):
    seed = seed_from_theta(theta, master_seed=seed)
    rng = default_rng(seed)
    params = build_params(theta, fixed_params, core_params_num)
    AC, IMM, _ = functions_list.initialise_agents_v5(params, rng=rng)
    SSPrev_selected, SSPrev, AIBKS = functions_list.simulator_v6_numba(
        AC, IMM, params, 0, 1, seed=seed
    )
    return SSPrev_selected.astype(float)


def run_single_simulation(args):
    """Wrapper for parallel processing."""
    sample_id, R0_range, sigma_range, fixed_params, core_params_num, master_seed = args

    rng = np.random.default_rng(master_seed + sample_id)
    R0_sel, sigma_sel = prior_value_2params(R0_range, sigma_range, rng)
    y_sim = simulate_prevalence_v5_numba(
        [R0_sel, sigma_sel],
        fixed_params,
        core_params_num,
        seed=master_seed + sample_id
    )

    # Return in order for batch assembly
    return y_sim, R0_sel, sigma_sel


# ============================================================================
# OPTIMIZED: Batched array storage for best HDF5 performance
# ============================================================================

def abc_simulation_batched_arrays(
        R0_range,
        sigma_range,
        fixed_params,
        core_params_num,
        n_simulations=500_000,
        output_dir='../../experimental_data/from_260618/simulation_banks',
        max_file_size_mb=95,
        n_jobs=30,
        master_seed=123
):
    """
    Run ABC simulations with OPTIMIZED HDF5 storage using batched arrays.

    Instead of creating 500k small datasets, we create 3 large arrays:
    - simulations: (n_samples, 42, 23) - all simulation results
    - R0_values: (n_samples,) - all R0 values
    - sigma_values: (n_samples,) - all sigma values

    This is MUCH faster for HDF5 I/O!

    Parameters:
    -----------
    R0_range : tuple
        Range for R0 parameter
    sigma_range : tuple
        Range for sigma parameter
    fixed_params : np.ndarray
        Fixed parameters
    core_params_num : int
        Number of core parameters
    n_simulations : int
        Total number of simulations to run
    output_dir : str
        Directory to save results
    max_file_size_mb : int
        Maximum file size (for GitHub compatibility)
    n_jobs : int
        Number of parallel workers
    master_seed : int
        Master random seed
    """

    logging.info("=" * 70)
    logging.info("ABC SIMULATION - OPTIMIZED BATCHED STORAGE")
    logging.info("=" * 70)
    logging.info(f"Platform: 32 vCPUs, 64GB RAM")
    logging.info(f"Configuration:")
    logging.info(f"  - Total simulations: {n_simulations:,}")
    logging.info(f"  - Parallel workers: {n_jobs}")
    logging.info(f"  - R0 range: {R0_range}")
    logging.info(f"  - Sigma range: {sigma_range}")
    logging.info(f"  - Storage: Batched arrays (optimized HDF5)")
    logging.info("=" * 70)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Calculate samples per file
    # Each sample: 42*23*8 (simulation) + 8 (R0) + 8 (sigma) = 7,376 bytes
    # With compression (~65%): ~2,600 bytes per sample
    # For batched arrays, we can fit more due to better compression
    bytes_per_sample = 2500  # Slightly better with batched storage
    max_bytes = max_file_size_mb * 1024 * 1024
    samples_per_file = int(max_bytes / bytes_per_sample)
    n_files = (n_simulations + samples_per_file - 1) // samples_per_file

    logging.info(f"File splitting: {n_files} files (~{samples_per_file:,} samples each)")
    logging.info("")

    start_time = time.perf_counter()
    start_sample_id = 500_000 * 1
    # filenum = 14 * 3

    # Process each output file
    for file_idx in range(14*1, 14*1 + n_files):
        file_start = (file_idx - 14*1) * samples_per_file + start_sample_id
        file_end = min(file_start + samples_per_file, n_simulations + start_sample_id)
        n_samples_this_file = file_end - file_start

        output_file = output_path / f'simulation_bank_part_{file_idx:04d}.h5'

        logging.info("=" * 70)
        logging.info(f"FILE {file_idx + 1}/{n_files}: {output_file.name}")
        logging.info(f"Samples {file_start:,} to {file_end - 1:,} ({n_samples_this_file:,} total)")
        logging.info("=" * 70)

        # Pre-allocate arrays for this file
        simulations_batch = np.zeros((n_samples_this_file, 42, 23), dtype=np.float32)
        R0_batch = np.zeros(n_samples_this_file, dtype=np.float32)
        sigma_batch = np.zeros(n_samples_this_file, dtype=np.float32)

        # Prepare arguments for parallel processing
        args_list = [
            (i, R0_range, sigma_range, fixed_params, core_params_num, master_seed)
            for i in range(file_start, file_end)
        ]

        # Run parallel simulations
        logging.info(f"Running {n_samples_this_file:,} simulations with {n_jobs} workers...")

        with mp.Pool(n_jobs) as pool:
            results = list(tqdm(
                pool.imap(run_single_simulation, args_list),
                total=len(args_list),
                desc="Simulating",
                unit="sample"
            ))

        # Fill batched arrays
        logging.info("Assembling batch arrays...")
        for idx, (y_sim, R0_sel, sigma_sel) in enumerate(tqdm(results, desc="Assembling")):
            simulations_batch[idx] = y_sim
            R0_batch[idx] = R0_sel
            sigma_batch[idx] = sigma_sel

        # Save to HDF5 as 3 large arrays (MUCH faster!)
        logging.info("Writing batched arrays to HDF5 (compressed)...")
        write_start = time.perf_counter()

        with h5py.File(output_file, 'w') as f:
            # Store metadata
            f.attrs['file_index'] = file_idx
            f.attrs['total_files'] = n_files
            f.attrs['start_sample'] = file_start
            f.attrs['end_sample'] = file_end
            f.attrs['n_samples'] = n_samples_this_file
            f.attrs['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
            f.attrs['R0_range'] = R0_range
            f.attrs['sigma_range'] = sigma_range

            # Create 3 large datasets (optimized for HDF5!)
            f.create_dataset(
                'simulations',
                data=simulations_batch,
                compression='gzip',
                compression_opts=6,
                chunks=(min(1000, n_samples_this_file), 42, 23)  # Optimal chunk size
            )

            f.create_dataset(
                'R0',
                data=R0_batch,
                compression='gzip',
                compression_opts=6
            )

            f.create_dataset(
                'sigma',
                data=sigma_batch,
                compression='gzip',
                compression_opts=6
            )

        write_time = time.perf_counter() - write_start

        # Check file size
        file_size_mb = output_file.stat().st_size / (1024 ** 2)
        status = "✓" if file_size_mb < 100 else "⚠️ EXCEEDS 100MB"

        logging.info(f"File size: {file_size_mb:.2f} MB {status}")
        logging.info(f"Write time: {write_time:.1f} seconds")

        # Show progress
        elapsed = time.perf_counter() - start_time
        samples_done = file_end
        samples_remaining = n_simulations - samples_done
        rate = samples_done / elapsed
        eta_seconds = samples_remaining / rate if rate > 0 else 0

        logging.info(f"Progress: {samples_done:,}/{n_simulations:,} ({100 * samples_done / n_simulations:.1f}%)")
        logging.info(f"Rate: {rate:.1f} samples/sec")
        logging.info(f"ETA: {eta_seconds / 60:.1f} minutes")
        logging.info("")

    # Final summary
    total_time = time.perf_counter() - start_time

    logging.info("=" * 70)
    logging.info("SIMULATION COMPLETE!")
    logging.info("=" * 70)
    logging.info(f"Total samples: {n_simulations:,}")
    logging.info(f"Total time: {total_time / 60:.1f} minutes ({total_time / 3600:.2f} hours)")
    logging.info(f"Average rate: {n_simulations / total_time:.1f} samples/sec")
    logging.info(f"Output directory: {output_dir}/")

    # Storage summary
    total_size_mb = sum(
        f.stat().st_size for f in output_path.glob('*.h5')
    ) / (1024 ** 2)

    logging.info(f"Storage used: {total_size_mb:.2f} MB ({total_size_mb / 1024:.2f} GB)")
    logging.info("=" * 70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':

    # Parameter ranges
    R0_range = [1.0, 4.0]
    sigma_range = [0.2, 1.0]

    if core_params_num == 2:
        # Run optimized simulation with batched array storage
        abc_simulation_batched_arrays(
            R0_range=R0_range,
            sigma_range=sigma_range,
            fixed_params=fixed_params,
            core_params_num=core_params_num,
            n_simulations=500_000,
            output_dir='../../experimental_data/from_260618/simulation_banks',
            max_file_size_mb=90,
            n_jobs=60,
            master_seed=123
        )

        end = time.perf_counter()
        logging.info(f"Total elapsed: {end - start:.2f} seconds ({(end - start) / 60:.2f} minutes)")
    else:
        raise NotImplementedError("Only core_params_num=2 is implemented")