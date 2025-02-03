#!/usr/bin/env python3
"""
Convert EDM4HEP tracker hits to HDF5 format.
"""

import argparse
from pathlib import Path
from tqdm import tqdm

from utils import get_run_paths, ensure_output_dir, get_chunk_info
from edm4hep_processor import load_edm4hep_event, process_hits_data
from hdf5_manager import build_hdf5_dataset

def convert_hits(
    base_dir: str,
    output_base_dir: str,
    dataset_name: str,
    chunk_size: int = 1000,
    run_size: int = 10,
) -> None:
    """
    Convert EDM4HEP tracker hits to HDF5 format.
    
    Args:
        base_dir: Base directory containing EDM4HEP files
        output_base_dir: Base directory for output files
        dataset_name: Name of the dataset
        chunk_size: Number of events per output file
        run_size: Number of events per run
    """
    # Get run directories
    run_dirs = get_run_paths(base_dir)
    num_runs = len(run_dirs)
    
    # Calculate chunk information
    num_events, runs_per_chunk, num_chunks = get_chunk_info(num_runs, run_size, chunk_size)
    
    print(f"Processing {num_runs} runs with {num_events} total events")
    print(f"Processing {runs_per_chunk} runs per chunk to get ~{chunk_size} events per file")
    
    # Create output directory
    output_dir = ensure_output_dir(output_base_dir, dataset_name)
    dataset_name = dataset_name.replace("/", ".")
    
    # Process chunks of runs
    for start_run in tqdm(range(0, num_runs, runs_per_chunk), desc="Processing chunks"):
        all_events_data = []
        
        # Process each run in the chunk
        for run_idx in range(start_run, min(start_run + runs_per_chunk, len(run_dirs))):
            run_dir = run_dirs[run_idx]
            print(f"Processing run {run_idx} at {run_dir}")
            
            # Load EDM4HEP file
            edm4hep_file = f"{run_dir}/edm4hep.root"
            event = load_edm4hep_event(edm4hep_file, event_num=0)
            
            if "tracker_df" not in event:
                print(f"No tracker hits found in run {run_idx}")
                continue
            
            # Process event
            event_df = process_hits_data(
                run_idx * run_size,
                event["tracker_df"]
            )
            all_events_data.append(event_df)
            
        if all_events_data:
            # Calculate event range for filename
            start_event = start_run * run_size
            end_event = min((start_run + runs_per_chunk) * run_size - 1, 
                           len(run_dirs) * run_size - 1)
            
            output_file = f"{output_dir}/{dataset_name}.events{start_event}-{end_event}.h5"
            build_hdf5_dataset(pd.concat(all_events_data, ignore_index=True), output_file)
            print(f"Saved {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert EDM4HEP tracker hits to HDF5")
    parser.add_argument("base_dir", help="Base directory containing EDM4HEP files")
    parser.add_argument("output_dir", help="Output directory for HDF5 files")
    parser.add_argument("dataset_name", help="Name of the dataset")
    parser.add_argument("--chunk-size", type=int, default=1000,
                      help="Number of events per output file")
    parser.add_argument("--run-size", type=int, default=10,
                      help="Number of events per run")
    
    args = parser.parse_args()
    
    convert_hits(
        args.base_dir,
        args.output_dir,
        args.dataset_name,
        args.chunk_size,
        args.run_size
    )

if __name__ == "__main__":
    import pandas as pd
    main() 