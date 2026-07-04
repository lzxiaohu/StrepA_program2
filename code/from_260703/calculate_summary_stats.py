# file name: calculate_summary_stats.py


import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
import summary_stats_elms_260528 as ss
import time
 
 
def summary_stats(series_2d):
    """Calculate summary statistics for a single simulation."""
    avg_prev_obs = ss.avg_prev_numpy(series_2d)
    var_prev_obs = np.sqrt(ss.var_prev_numpy(series_2d))
    avg_npmi_obs = ss.avg_npmi_numpy(series_2d)
    div_all_isolates_obs = ss.div_all_isolates_numpy(series_2d)
    
    return np.array([avg_prev_obs, var_prev_obs, avg_npmi_obs, div_all_isolates_obs], float)
 
 
def process_single_simulation(args):
    """Worker function for parallel processing."""
    simulation, sample_id = args
    stats = summary_stats(simulation)
    return sample_id, stats
 
 
def calculate_summary_stats_clean(
    input_dir='../../experimental_data/from_260703/simulation_banks',
    output_file='../../experimental_data/from_260703/all_summary_statistics_clean.h5',
    mapping_file='../../experimental_data/from_260703/valid_indices.csv',
    n_jobs=40
):
    """
    Calculate summary statistics, remove NaN, and save index mapping.
    
    The mapping file contains: clean_index → original_index
    This allows you to find the original simulation matrix for any clean sample.
    """
    
    input_path = Path(input_dir)
    files = sorted(input_path.glob('simulation_bank_part_*.h5'))
    
    # Count total samples
    total_samples = 0
    for filepath in files:
        with h5py.File(filepath, 'r') as f:
            total_samples += len(f['R0'])
    
    print("="*70)
    print("CALCULATING SUMMARY STATISTICS → CLEAN DATA (NaN REMOVAL)")
    print("="*70)
    print(f"Input directory: {input_dir}")
    print(f"Input files: {len(files)}")
    print(f"Total samples: {total_samples:,}")
    print(f"Output file: {output_file}")
    print(f"Mapping file: {mapping_file}")
    print(f"Parallel workers: {n_jobs}")
    print("="*70)
    
    # Pre-allocate arrays
    print(f"\nAllocating arrays for {total_samples:,} samples...")
    all_summary_stats = np.zeros((total_samples, 4), dtype=np.float32)
    all_R0 = np.zeros(total_samples, dtype=np.float32)
    all_sigma = np.zeros(total_samples, dtype=np.float32)
    
    start_time = time.time()
    current_idx = 0
    
    # Process each file
    for file_num, filepath in enumerate(files, 1):
        print(f"\n{'='*70}")
        print(f"FILE {file_num}/{len(files)}: {filepath.name}")
        print(f"{'='*70}")
        
        with h5py.File(filepath, 'r') as f:
            n_samples = len(f['R0'])
            print(f"Samples in this file: {n_samples:,}")
            print(f"Global index: {current_idx:,} to {current_idx + n_samples - 1:,}")
            
            # Load data
            simulations = f['simulations'][:]
            R0_array = f['R0'][:]
            sigma_array = f['sigma'][:]
            
            # Store parameters
            all_R0[current_idx:current_idx + n_samples] = R0_array
            all_sigma[current_idx:current_idx + n_samples] = sigma_array
            
            # Prepare arguments
            args_list = [(simulations[i], current_idx + i) for i in range(n_samples)]
            
            # Calculate statistics in parallel
            print(f"Calculating summary statistics with {n_jobs} workers...")
            with mp.Pool(n_jobs) as pool:
                results = list(tqdm(
                    pool.imap(process_single_simulation, args_list),
                    total=len(args_list),
                    desc="  Processing",
                    ncols=80
                ))
            
            # Store results
            for sample_id, stats in results:
                all_summary_stats[sample_id] = stats
            
            current_idx += n_samples
            
            # Progress update
            elapsed = time.time() - start_time
            rate = current_idx / elapsed
            remaining = total_samples - current_idx
            eta = remaining / rate if rate > 0 else 0
            
            print(f"Progress: {current_idx:,}/{total_samples:,} ({100*current_idx/total_samples:.1f}%)")
            print(f"Rate: {rate:.1f} samples/sec")
            print(f"ETA: {eta/60:.1f} minutes")
    
    # REMOVE ROWS WITH NaN AND SAVE MAPPING
    print(f"\n{'='*70}")
    print("CHECKING FOR NaN VALUES...")
    print(f"{'='*70}")
    
    # Find rows with any NaN
    has_nan = np.any(np.isnan(all_summary_stats), axis=1)
    n_nan_rows = has_nan.sum()
    n_valid_rows = (~has_nan).sum()
    
    print(f"Original samples: {total_samples:,}")
    print(f"Rows with NaN:    {n_nan_rows:,} ({100*n_nan_rows/total_samples:.3f}%)")
    print(f"Valid rows:       {n_valid_rows:,} ({100*n_valid_rows/total_samples:.3f}%)")
    
    if n_nan_rows > 0:
        print(f"\n⚠️  Removing {n_nan_rows:,} rows with NaN values...")
        print(f"   Keeping {n_valid_rows:,} clean samples")
        
        # *** SAVE MAPPING: clean_index → original_index ***
        valid_indices = np.where(~has_nan)[0]
        
        print(f"\n📋 Creating index mapping:")
        print(f"   valid_indices[0] = {valid_indices[0]} (1st clean sample → original sample {valid_indices[0]})")
        print(f"   valid_indices[1] = {valid_indices[1]} (2nd clean sample → original sample {valid_indices[1]})")
        print(f"   ...")
        print(f"   valid_indices[{n_valid_rows-1}] = {valid_indices[-1]} (last clean sample → original sample {valid_indices[-1]})")
        
        # Keep only valid rows
        all_summary_stats_clean = all_summary_stats[~has_nan]
        all_R0_clean = all_R0[~has_nan]
        all_sigma_clean = all_sigma[~has_nan]
        final_n_samples = n_valid_rows
    else:
        print(f"\n✓ No NaN values found! All data is clean.")
        
        # Identity mapping (no samples removed)
        valid_indices = np.arange(total_samples)
        
        all_summary_stats_clean = all_summary_stats
        all_R0_clean = all_R0
        all_sigma_clean = all_sigma
        final_n_samples = total_samples
    
    # Save the mapping file
    print(f"\n{'='*70}")
    print("SAVING INDEX MAPPING...")
    print(f"{'='*70}")
    
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.savetxt(mapping_file, valid_indices, fmt='%d', 
              header=f'Mapping: clean_index → original_index\n'
                     f'Total clean samples: {final_n_samples}\n'
                     f'Original total: {total_samples}\n'
                     f'Removed: {n_nan_rows}')
    
    print(f"✓ Saved mapping to: {mapping_file}")
    print(f"  Format: valid_indices[clean_idx] = original_idx")
    print(f"  Length: {len(valid_indices):,} rows")
    
    # Save clean data to HDF5
    print(f"\n{'='*70}")
    print("SAVING CLEAN DATA TO HDF5...")
    print(f"{'='*70}")
    
    with h5py.File(output_file, 'w') as f:
        # Save clean summary statistics
        f.create_dataset(
            'summary_stats',
            data=all_summary_stats_clean,
            compression='gzip',
            compression_opts=6,
            chunks=(min(10000, final_n_samples), 4)
        )
        
        # Save clean parameters
        f.create_dataset(
            'R0',
            data=all_R0_clean,
            compression='gzip',
            compression_opts=6
        )
        
        f.create_dataset(
            'sigma',
            data=all_sigma_clean,
            compression='gzip',
            compression_opts=6
        )
        
        # ALSO save the mapping inside HDF5
        f.create_dataset(
            'valid_indices',
            data=valid_indices,
            compression='gzip',
            compression_opts=6
        )
        
        # Add metadata
        f.attrs['n_samples'] = final_n_samples
        f.attrs['n_samples_original'] = total_samples
        f.attrs['n_samples_removed'] = n_nan_rows
        f.attrs['removal_rate'] = float(n_nan_rows) / total_samples
        f.attrs['n_statistics'] = 4
        f.attrs['columns'] = ['avg_prev_obs', 'var_prev_obs', 'avg_npmi_obs', 'div_all_isolates_obs']
        f.attrs['R0_range'] = [1.0, 8.0]
        f.attrs['sigma_range'] = [0.2, 1.0]
        f.attrs['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        f.attrs['data_quality'] = 'clean (NaN rows removed)'
        f.attrs['mapping_info'] = 'valid_indices dataset maps clean → original indices'
    
    # Final summary
    total_time = time.time() - start_time
    file_size_mb = Path(output_file).stat().st_size / (1024**2)
    
    print(f"\n{'='*70}")
    print("✅ SUMMARY STATISTICS COMPLETE (CLEAN DATA)!")
    print(f"{'='*70}")
    print(f"Original samples:  {total_samples:,}")
    print(f"Removed (NaN):     {n_nan_rows:,} ({100*n_nan_rows/total_samples:.3f}%)")
    print(f"Final samples:     {final_n_samples:,}")
    print(f"Total time:        {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Average rate:      {total_samples/total_time:.1f} samples/sec")
    print(f"Output HDF5:       {output_file} ({file_size_mb:.2f} MB)")
    print(f"Mapping CSV:       {mapping_file}")
    print(f"{'='*70}")
    
    print("\n" + "="*70)
    print("HOW TO USE THE MAPPING:")
    print("="*70)
    print("""
# Load the mapping
import numpy as np
valid_indices = np.loadtxt('valid_indices.csv', dtype=int)
 
# You have a clean index from distance file (0 to 499,664)
clean_idx = 12345
 
# Map to original simulation bank index (0 to 499,999)
original_idx = valid_indices[clean_idx]
 
# Now load the correct simulation matrix
file_idx = original_idx // samples_per_file
local_idx = original_idx % samples_per_file
 
with h5py.File(f'simulation_bank_part_{file_idx:04d}.h5', 'r') as f:
    simulation = f['simulations'][local_idx]
    """)
 
 
# ============================================================================
# MAIN EXECUTION
# ============================================================================
 
if __name__ == "__main__":
    
    calculate_summary_stats_clean(
        input_dir='../../experimental_data/from_260703/simulation_banks',
        output_file='../../experimental_data/from_260703/all_summary_statistics_clean.h5',
        mapping_file='../../experimental_data/from_260703/valid_indices.csv',
        n_jobs=60
    )