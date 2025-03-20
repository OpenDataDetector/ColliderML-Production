#!/usr/bin/env python3
"""
Convert EDM4HEP track data to HDF5 format.
"""

import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import h5py
import uproot
import math
from typing import Dict, List, Any

from utils.utils import get_run_paths, ensure_output_dir, get_chunk_info
from utils.edm4hep_utils import pixel_readouts, strip_readouts

from utils.track_utils import (
    convert_hit_ids, load_track_summary,
    create_particle_barcode_map, get_majority_particle_id, load_root_file,
    get_particle_ids_from_events
)

import awkward as ak

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Convert EDM4HEP track data to HDF5")
    
    # Required arguments
    parser.add_argument(
        "--config",
        help="Path to YAML config file",
        type=str,
        required=True
    )
    
    # Optional chunk index - if not provided, process all chunks
    parser.add_argument(
        "--chunk-index", 
        help="Index of chunk to process (for parallel processing). If not provided, process all chunks.",
        type=int,
        default=None
    )
    
    return parser.parse_args()

def process_event_for_tracks(
    run_dir: Path,
    local_event_num: int,
    global_event_num: int,
    tracksummary_arrays: Any,
    tracks_csv_pattern: str,
    simhits_df: pd.DataFrame,
    edm4hep_hits_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Process a single event.
    
    Args:
        run_dir: Path to run directory
        local_event_num: Event number within the run
        global_event_num: Global event number across all runs
        tracksummary_arrays: Arrays from tracksummary ROOT file
        tracks_csv_pattern: Pattern for tracks CSV filenames
        simhits_df: DataFrame containing simhits data
        edm4hep_hits_df: DataFrame containing edm4hep hits data
        
    Returns:
        DataFrame containing track data for this event
    """
    # Load tracks CSV
    tracks_csv = pd.read_csv(run_dir / tracks_csv_pattern.format(local_event_num))
    
    # Get track summary data for this event
    arrays = tracksummary_arrays[local_event_num]
    track_data = {}
    for field in arrays.fields:
        if field == 'event_nr':
            continue
        try:
            array = ak.to_numpy(arrays[field])
            assert len(array.shape) == 1
            track_data[field] = array
        except Exception:
            continue
            
    track_fitting_df = pd.DataFrame(track_data).rename(columns={"track_nr": "track_id"})

    # Get this local event hits
    local_event_edm4hep_hits = edm4hep_hits_df[edm4hep_hits_df.event_id == local_event_num]
    local_event_simhits = simhits_df[simhits_df.event_id == local_event_num]

    # Build particle ID - particle barcode mapping
    particle_barcode_map = create_particle_barcode_map(local_event_edm4hep_hits, local_event_simhits)

    # Get majority particle ID for each track
    majority_particle_ids = tracks_csv.Hits_ID.apply(get_majority_particle_id, args=(local_event_simhits, particle_barcode_map))    
    
    # Combine data
    track_finding_data = {
        "event_id": global_event_num,
        "track_id": tracks_csv.track_id.values,
        "num_hits": tracks_csv.nMeasurements.values,
        "num_outliers": tracks_csv.nOutliers.values,
        "num_holes": tracks_csv.nHoles.values,
        "num_shared_hits": tracks_csv.nSharedHits.values,
        "chi2": tracks_csv.chi2.values,
        "hit_ids": tracks_csv.Hits_ID.apply(convert_hit_ids).values,
        "majority_particle_id": majority_particle_ids.values,
    }
    
    track_fitting_data = {
        "event_id": global_event_num,
        "track_id": track_fitting_df.track_id.values,
        "d0": track_fitting_df.eLOC0_fit.values,
        "z0": track_fitting_df.eLOC1_fit.values,
        "phi": track_fitting_df.ePHI_fit.values,
        "theta": track_fitting_df.eTHETA_fit.values,
        "qop": track_fitting_df.eQOP_fit.values,
        "time": track_fitting_df.eT_fit.values,
        "d0_truth": track_fitting_df.t_d0.values,
        "z0_truth": track_fitting_df.t_z0.values,
        "phi_truth": track_fitting_df.t_phi.values,
        "theta_truth": track_fitting_df.t_theta.values,
        "charge_truth": track_fitting_df.t_charge.values,
        "p_truth": track_fitting_df.t_p.values,
        "pT_truth": track_fitting_df.t_pT.values,
        "time_truth": track_fitting_df.t_time.values,
    }
    
    full_track_df = pd.DataFrame(track_finding_data)
    event_df = full_track_df.merge(pd.DataFrame(track_fitting_data), 
                                 on=["event_id", "track_id"])
    return event_df

def build_hdf5_tracks(df: pd.DataFrame, output_file: str) -> None:
    """
    Build HDF5 file with event/track/hit hierarchy.
    
    Args:
        df: DataFrame containing track data
        output_file: Path to output HDF5 file
    """
    with h5py.File(output_file, 'a') as f:
        if 'events' not in f:
            events_group = f.create_group('events')
        else:
            events_group = f['events']
            
        for event_id, event_df in df.groupby('event_id'):
            event_group = events_group.create_group(f'event_{event_id}')
            
            # Store track data
            track_data = event_df.drop(columns=['hit_ids', 'event_id'])
            event_group.create_dataset('tracks', data=track_data.to_records(index=False))
            
            # Store hit arrays
            hit_arrays = event_df['hit_ids'].values
            event_group.create_dataset(
                'hit_ids',
                data=hit_arrays,
                dtype=h5py.vlen_dtype(np.dtype('int32')),
                compression="gzip",
                compression_opts=9
            )

def process_run_for_tracks(
    run_dir: str | Path,
    run_number: int,
    run_size: int,
    file_patterns: dict
) -> List[pd.DataFrame]:
    """
    Process all events in a single run.
    
    Args:
        run_dir: Path to run directory
        run_number: Run number (for global event numbering)
        run_size: Number of events in each run
        file_patterns: Dictionary of file patterns and names
        
    Returns:
        List of DataFrames, one for each event in the run
    """
    run_dir = Path(run_dir)
    
    # Load track summary data once for the whole run
    tracksummary_arrays = load_track_summary(run_dir / file_patterns["tracksummary_file"])

    # Load simulated hits and EDM4hep data
    simhits_df = load_root_file(run_dir / file_patterns["simhits_file"])
    
    edm4hep_events = uproot.open(run_dir / file_patterns["edm4hep_file"])["events"]
    edm4hep_hits_df = get_particle_ids_from_events(edm4hep_events, pixel_readouts + strip_readouts)
    
    run_events = []
    for local_event_num in range(run_size):
        try:
            # Calculate global event number
            global_event_num = run_number * run_size + local_event_num
            
            event_df = process_event_for_tracks(
                run_dir,
                local_event_num,
                global_event_num,
                tracksummary_arrays,
                file_patterns["tracks_csv_pattern"],
                simhits_df,
                edm4hep_hits_df
            )
            run_events.append(event_df)
            
        except FileNotFoundError as e:
            print(f"Skipping missing event {local_event_num} in {run_dir}: {str(e)}")
            continue
            
    return run_events

def process_chunk_for_tracks(
    run_dirs: List[Path],
    start_run: int,
    runs_per_chunk: int,
    output_dir: Path,
    dataset_name: str,
    run_size: int,
    file_patterns: dict
) -> None:
    """
    Process a chunk of runs.
    
    Args:
        run_dirs: List of run directory paths
        start_run: Index of first run in chunk
        runs_per_chunk: Number of runs to process
        output_dir: Output directory for HDF5 files
        dataset_name: Name of dataset
        run_size: Number of events per run
        file_patterns: Dictionary of file patterns and names
    """
    # Calculate event range for this chunk
    start_event = start_run * run_size
    end_run = min(start_run + runs_per_chunk, len(run_dirs))
    end_event = (end_run * run_size) - 1
    
    # Build output filename with event range
    output_file = output_dir / f"{dataset_name}.events{start_event}-{end_event}.h5"
    
    # Skip if file already exists
    if output_file.exists():
        print(f"\nSkipping events {start_event}-{end_event} - output file already exists: {output_file}")
        return
        
    chunk_run_dirs = run_dirs[start_run:end_run]
    
    # Process each run in chunk
    all_track_data = []
    for run_idx, run_dir in enumerate(tqdm(chunk_run_dirs, desc="Processing runs", leave=False)):
        run_events = process_run_for_tracks(
            run_dir,
            start_run + run_idx,
            run_size,
            file_patterns
        )
        all_track_data.extend(run_events)
            
    # Save chunk to HDF5
    if all_track_data:
        all_events_df = pd.concat(all_track_data, ignore_index=True)
        build_hdf5_tracks(all_events_df, str(output_file))  # Convert Path to string for h5py
        print(f"\nSaved events {start_event}-{end_event} to {output_file}")
    else:
        print(f"\nNo data to save for events {start_event}-{end_event}")

def main():
    # Parse arguments
    args = parse_args()
    
    # Load config file
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Extract parameters from config
    dataset = config["dataset"]
    version = config["version"]
    
    # Build paths from config
    input_base_dir = Path(config["common"]["output_base_dir"]) / dataset / version
    output_base_dir = Path(config["common"]["staging_dir"])
    output_path = config.get("output_path", f"{dataset}/{version}/reco/tracks")
    
    # Processing parameters
    chunk_size = config.get("chunk_size", 1000)
    run_size = config.get("run_size", 10)
    
    # File patterns
    file_patterns = {
        "tracks_csv_pattern": config.get("tracks_csv_pattern", "event{:09d}-tracks_ambi.csv"),
        "tracksummary_file": config.get("tracksummary_file", "tracksummary_ambi.root"),
        "simhits_file": config.get("simhits_file", "simhits.root"),
        "edm4hep_file": config.get("edm4hep_file", "edm4hep.root")
    }
    
    print("\nStarting track conversion with configuration:")
    print(f"Dataset: {dataset}, Version: {version}")
    print(f"Input directory: {input_base_dir}")
    print(f"Output directory: {output_base_dir}/{output_path}")
    print(f"Chunk size: {chunk_size}, Run size: {run_size}")
    
    # Get run directories
    run_dirs = get_run_paths(input_base_dir)
    num_runs = len(run_dirs)
    
    # Calculate chunk information
    runs_per_chunk = chunk_size // run_size
    num_chunks = math.ceil(num_runs / runs_per_chunk)
    
    print(f"Found {num_runs} run directories")
    print(f"Each chunk will process {runs_per_chunk} runs")
    print(f"Total number of chunks: {num_chunks}")
    
    # Create output directory and ensure it's a Path object
    output_dir = ensure_output_dir(output_base_dir, output_path)
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
        
    dataset_name = output_path.replace("/", ".")
    
    # Process chunks based on argument
    if args.chunk_index is not None:
        # Process only the specified chunk
        chunk_idx = args.chunk_index
        if chunk_idx < num_chunks:
            start_run = chunk_idx * runs_per_chunk
            print(f"Processing chunk {chunk_idx}: runs {start_run} to {min(start_run + runs_per_chunk, num_runs) - 1}")
            
            process_chunk_for_tracks(
                run_dirs, 
                start_run, 
                runs_per_chunk, 
                output_dir, 
                dataset_name,
                run_size,
                file_patterns
            )
        else:
            print(f"Chunk index {chunk_idx} is out of range (0-{num_chunks-1})")
    else:
        # Process all chunks sequentially
        print(f"No chunk index provided - processing all {num_chunks} chunks sequentially")
        for chunk_idx in range(num_chunks):
            start_run = chunk_idx * runs_per_chunk
            print(f"\nProcessing chunk {chunk_idx}/{num_chunks-1}: runs {start_run} to {min(start_run + runs_per_chunk, num_runs) - 1}")
            
            process_chunk_for_tracks(
                run_dirs, 
                start_run, 
                runs_per_chunk, 
                output_dir, 
                dataset_name,
                run_size,
                file_patterns
            )

if __name__ == "__main__":
    main()