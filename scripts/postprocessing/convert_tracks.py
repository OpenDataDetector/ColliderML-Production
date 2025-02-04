#!/usr/bin/env python3
"""
Convert EDM4HEP track data to HDF5 format.
"""

import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import h5py
import uproot
from typing import Dict, List, Any

from utils.utils import get_run_paths, ensure_output_dir, get_chunk_info
from utils.edm4hep_utils import pixel_readouts, strip_readouts

from utils.track_utils import (
    convert_hit_ids, load_track_summary,
    create_particle_barcode_map, get_majority_particle_id, load_root_file,
    get_particle_ids_from_events
)

import awkward as ak
from utils.config import create_base_parser, load_config

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
    print(f"\nBuilding HDF5 file: {output_file}")
    print(f"Input DataFrame shape: {df.shape}")
    
    with h5py.File(output_file, 'a') as f:
        if 'events' not in f:
            print("Creating events group")
            events_group = f.create_group('events')
        else:
            print("Using existing events group")
            events_group = f['events']
            
        for event_id, event_df in df.groupby('event_id'):
            print(f"\nProcessing event {event_id}")
            event_group = events_group.create_group(f'event_{event_id}')
            
            # Store track data
            track_data = event_df.drop(columns=['hit_ids', 'event_id'])
            print(f"Track data shape: {track_data.shape}")
            event_group.create_dataset('tracks', data=track_data.to_records(index=False))
            
            # Store hit arrays
            hit_arrays = event_df['hit_ids'].values
            print(f"Number of hit arrays: {len(hit_arrays)}")
            event_group.create_dataset(
                'hit_ids',
                data=hit_arrays,
                dtype=h5py.vlen_dtype(np.dtype('int32')),
                compression="gzip",
                compression_opts=9
            )
            print(f"Stored hit arrays for event {event_id}")

def process_run_for_tracks(
    run_dir: str | Path,
    run_number: int,
    run_size: int = 10,
    tracks_csv_pattern: str = "event{:09d}-tracks_ambi.csv",
    tracksummary_file: str = "tracksummary_ambi.root",
    simhits_file: str = "simhits.root",
    edm4hep_file: str = "edm4hep.root",
) -> List[pd.DataFrame]:
    """
    Process all events in a single run.
    
    Args:
        run_dir: Path to run directory
        run_number: Run number (for global event numbering)
        run_size: Number of events in each run
        tracks_csv_pattern: Pattern for tracks CSV filenames
        tracksummary_file: Name of track summary ROOT file
        simhits_file: Name of simulated hits ROOT file
        edm4hep_file: Name of EDM4hep ROOT file
        
    Returns:
        List of DataFrames, one for each event in the run
    """
    run_dir = Path(run_dir)
    print(f"\nProcessing run {run_number} at {run_dir}")
    
    # Load track summary data once for the whole run
    print("\nLoading track summary data")
    tracksummary_arrays = load_track_summary(run_dir / tracksummary_file)

    # Load simulated hits and EDM4hep data
    print(f"\nLoading ROOT file: {run_dir / simhits_file}")
    simhits_df = load_root_file(run_dir / simhits_file)
    
    print(f"\nLoading EDM4hep file: {run_dir / edm4hep_file}")
    edm4hep_events = uproot.open(run_dir / edm4hep_file)["events"]
    edm4hep_hits_df = get_particle_ids_from_events(edm4hep_events, pixel_readouts + strip_readouts)
    
    run_events = []
    for local_event_num in range(run_size):
        try:
            # Calculate global event number
            global_event_num = run_number * run_size + local_event_num
            print(f"\nProcessing event {local_event_num} (global event {global_event_num})")
            
            event_df = process_event_for_tracks(
                run_dir,
                local_event_num,
                global_event_num,
                tracksummary_arrays,
                tracks_csv_pattern,
                simhits_df,
                edm4hep_hits_df
            )
            run_events.append(event_df)
            print(f"Added event DataFrame with shape: {event_df.shape}")
            
        except FileNotFoundError as e:
            print(f"Skipping missing event {local_event_num} in {run_dir}: {str(e)}")
            continue
            
    print(f"Processed {len(run_events)} events in run {run_number}")
    return run_events

def process_chunk_for_tracks(
    run_dirs: List[Path],
    start_run: int,
    runs_per_chunk: int,
    output_dir: Path,
    dataset_name: str,
    run_size: int
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
    """
    end_run = min(start_run + runs_per_chunk, len(run_dirs))
    chunk_run_dirs = run_dirs[start_run:end_run]
    
    # Process each run in chunk
    all_track_data = []
    for run_idx, run_dir in enumerate(tqdm(chunk_run_dirs, desc="Processing runs", leave=False)):
        run_events = process_run_for_tracks(
            run_dir,
            start_run + run_idx,
            run_size
        )
        all_track_data.extend(run_events)
            
    # Save chunk to HDF5
    chunk_num = start_run // runs_per_chunk
    output_file = output_dir / f"{dataset_name}.chunk{chunk_num:03d}.h5"
    
    if all_track_data:
        all_events_df = pd.concat(all_track_data, ignore_index=True)
        build_hdf5_tracks(all_events_df, output_file)
        print(f"\nSaved chunk {chunk_num} to {output_file}")
    else:
        print(f"\nNo data to save for chunk {chunk_num}")

def convert_tracks(
    base_dir: Path | str,
    output_base_dir: Path | str,
    dataset_name: str,
    chunk_size: int = 1000,
    run_size: int = 10,
) -> None:
    """
    Convert track data to HDF5 format.
    
    Args:
        base_dir: Base directory containing EDM4HEP files
        output_base_dir: Base directory for output files
        dataset_name: Name of the dataset as a subdirectory of the base directory
        chunk_size: Number of events per output file
        run_size: Number of events per run
    """
    base_dir = Path(base_dir)
    output_base_dir = Path(output_base_dir)
    
    print(f"\nStarting track conversion")
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_base_dir}")
    print(f"Dataset name: {dataset_name}")
    
    # Get run directories
    run_dirs = get_run_paths(base_dir)
    num_runs = len(run_dirs)
    print(f"Found {num_runs} runs")
    
    # Calculate chunk information
    num_events, runs_per_chunk, num_chunks = get_chunk_info(num_runs, run_size, chunk_size)
    print(f"Processing {num_runs} runs with {num_events} total events")
    print(f"Processing {runs_per_chunk} runs per chunk to get ~{chunk_size} events per file")
    
    # Create output directory
    output_dir = ensure_output_dir(output_base_dir, dataset_name)
    dataset_name = dataset_name.replace("/", ".")
    print(f"Using output directory: {output_dir}")
    
    # Process chunks of runs
    for start_run in tqdm(range(0, num_runs, runs_per_chunk), desc="Processing chunks"):
        process_chunk_for_tracks(
            run_dirs, 
            start_run, 
            runs_per_chunk, 
            Path(output_dir), 
            dataset_name,
            run_size
        )

def main():
    # Create parser with common arguments
    parser = create_base_parser("Convert EDM4HEP track data to HDF5")
    
    # Add script-specific arguments
    parser.add_argument(
        "--tracks-csv-pattern",
        help="Pattern for tracks CSV filenames",
        type=str,
        default="event{:09d}-tracks_ambi.csv"
    )
    parser.add_argument(
        "--tracksummary-file",
        help="Name of track summary ROOT file",
        type=str,
        default="tracksummary_ambi.root"
    )
    parser.add_argument(
        "--simhits-file",
        help="Name of simulated hits ROOT file",
        type=str,
        default="simhits.root"
    )
    parser.add_argument(
        "--edm4hep-file",
        help="Name of EDM4hep ROOT file",
        type=str,
        default="edm4hep.root"
    )
    
    # Parse args and load config
    args = parser.parse_args()
    config = load_config(args)
    
    print("\nStarting track conversion with configuration:")
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    
    convert_tracks(
        config.base_dir,
        config.output_dir,
        config.dataset_name,
        config.chunk_size,
        config.run_size
    )

if __name__ == "__main__":
    main() 