#!/usr/bin/env python3
"""
Convert EDM4HEP tracker hits to HDF5 format.
"""

import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import h5py
import uproot
from typing import Dict, List, Any
import logging

from utils.utils import get_run_paths, ensure_output_dir, get_chunk_info
from utils.edm4hep_utils import pixel_readouts, strip_readouts, load_edm4hep_file
from utils.config import create_base_parser, load_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('hit_conversion.log')
    ]
)

def process_event_for_hits(
    event_id: int,
    tracker_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Process hit data for a single event using EDM4HEP tracker hits.
    
    Args:
        event_id: Event number
        tracker_df: DataFrame containing EDM4HEP tracker hits
        
    Returns:
        DataFrame containing hit data for this event
    """
    logging.debug(f"Processing event {event_id} with {len(tracker_df)} hits")
    
    # Select relevant columns
    hit_columns = [
        "cellID",
        "EDep",
        "time",
        "x",
        "y",
        "z",
        "px",
        "py",
        "pz",
        "particle_id",
        "detector",
    ]
    
    event_hits = tracker_df[hit_columns].copy().rename(
        columns={
            "EDep": "energy",
            "cellID": "cell_id"
            }
            )
    
    # Add event_id
    event_hits["event_id"] = event_id
    
    # Log detector statistics
    detector_stats = event_hits.detector.value_counts()
    logging.debug(f"Detector hit counts for event {event_id}:")
    for detector, count in detector_stats.items():
        logging.debug(f"  {detector}: {count} hits")
    
    return event_hits

def build_hdf5_hits(
    df: pd.DataFrame,
    output_file: str
) -> None:
    """
    Build HDF5 file with event/hit hierarchy.
    
    Structure:
    /events/
        /event_0/
            /hits    # Dataset containing hit properties
        /event_1/
            ...
    """
    logging.info(f"Building HDF5 file {output_file}")
    logging.info(f"Total events to write: {df.event_id.nunique()}")
    logging.info(f"Total hits to write: {len(df)}")
    logging.info("DataFrame column types:")
    for col in df.columns:
        logging.info(f"  {col}: {df[col].dtype}")
    
    # Create a compound dtype for the hits dataset
    dt = np.dtype([
        ('cell_id', np.int64),
        ('energy', np.float64),
        ('time', np.float64),
        ('x', np.float64),
        ('y', np.float64),
        ('z', np.float64),
        ('px', np.float64),
        ('py', np.float64),
        ('pz', np.float64),
        ('particle_id', np.int64),
        ('detector', h5py.string_dtype(encoding='utf8')),
    ])
    
    with h5py.File(output_file, 'a') as f:
        # Create events group if it doesn't exist
        if 'events' not in f:
            events_group = f.create_group('events')
        else:
            events_group = f['events']
            
        # Group DataFrame by event_id
        for event_id, event_df in df.groupby('event_id'):
            # Create event group
            event_group = events_group.create_group(f'event_{event_id}')
            
            # Drop event_id as it's stored in the group name
            event_df = event_df.drop(columns=['event_id'])
            
            # Create structured array with the correct dtype
            hits_data = np.empty(len(event_df), dtype=dt)
            
            # Copy data field by field, ensuring correct types
            for name in dt.names:
                hits_data[name] = event_df[name].values
            
            # Store hit data
            event_group.create_dataset(
                'hits',
                data=hits_data,
                compression="gzip",
                compression_opts=9
            )
            logging.debug(f"Wrote event {event_id} with {len(hits_data)} hits")

def process_run_for_hits(
    run_dir: Path,
    run_number: int,
    run_size: int = 10,
) -> List[pd.DataFrame]:
    """
    Process all events in a single run.
    
    Args:
        run_dir: Path to run directory
        run_number: Run number (for global event numbering)
        run_size: Number of events in each run
        
    Returns:
        List of DataFrames, one for each event in the run
    """
    run_dir = Path(run_dir)
    edm4hep_file = run_dir / "edm4hep.root"
    
    logging.info(f"Processing run {run_number} from {run_dir}")
    logging.info(f"Using EDM4HEP file: {edm4hep_file}")
    
    run_events = []
    successful_events = 0
    for local_event_num in range(run_size):
        try:
            # Calculate global event number
            global_event_num = run_number * run_size + local_event_num
            
            # Load EDM4HEP event
            event = load_edm4hep_file(edm4hep_file, event_num=local_event_num, collections=["tracker"])
            
            if "tracker_df" not in event:
                logging.warning(f"No tracker hits found in event {local_event_num}")
                continue
                
            # Process event
            event_df = process_event_for_hits(
                global_event_num,
                event["tracker_df"]
            )
            run_events.append(event_df)
            successful_events += 1
            
        except Exception as e:
            logging.error(f"Error processing event {local_event_num} in {run_dir}: {str(e)}")
            continue
    
    logging.info(f"Successfully processed {successful_events}/{run_size} events in run {run_number}")
    return run_events

def process_chunk_for_hits(
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
    # Calculate event range for this chunk
    start_event = start_run * run_size
    end_run = min(start_run + runs_per_chunk, len(run_dirs))
    print(f"end_run: {end_run}")
    end_event = (end_run * run_size) - 1
    
    logging.info(f"\nProcessing chunk: events {start_event}-{end_event}")
    
    # Build output filename with event range
    output_file = output_dir / f"{dataset_name}.events{start_event}-{end_event}.h5"
    
    # Skip if file already exists
    if output_file.exists():
        logging.info(f"Skipping events {start_event}-{end_event} - output file already exists: {output_file}")
        return
        
    chunk_run_dirs = run_dirs[start_run:end_run]
    
    # Process each run in chunk
    all_hit_data = []
    total_hits = 0
    for run_idx, run_dir in enumerate(tqdm(chunk_run_dirs, desc="Processing runs", leave=False)):
        run_events = process_run_for_hits(
            run_dir,
            start_run + run_idx,
            run_size
        )
        all_hit_data.extend(run_events)
        total_hits += sum(len(df) for df in run_events)
            
    # Save chunk to HDF5
    if all_hit_data:
        all_events_df = pd.concat(all_hit_data, ignore_index=True)
        logging.info(f"Chunk statistics:")
        logging.info(f"  Total events: {all_events_df.event_id.nunique()}")
        logging.info(f"  Total hits: {total_hits}")
        logging.info(f"  Average hits per event: {total_hits/all_events_df.event_id.nunique():.1f}")
        
        build_hdf5_hits(all_events_df, output_file)
        logging.info(f"Saved events {start_event}-{end_event} to {output_file}")
    else:
        logging.warning(f"No data to save for events {start_event}-{end_event}")

def convert_hits(
    base_dir: Path | str,
    output_base_dir: Path | str,
    dataset_name: str,
    chunk_size: int = 1000,
    run_size: int = 10,
    *,
    chunk_index: int | None = None,
    max_chunks: int | None = None,
    max_runs: int | None = None,
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
    base_dir = Path(base_dir)
    output_base_dir = Path(output_base_dir)
    
    logging.info("Starting hit conversion")
    logging.info(f"Base directory: {base_dir}")
    logging.info(f"Output directory: {output_base_dir}")
    logging.info(f"Dataset name: {dataset_name}")
    
    # Get run directories
    run_dirs = get_run_paths(base_dir)
    if max_runs is not None:
        run_dirs = run_dirs[:max(0, int(max_runs))]
    num_runs = len(run_dirs)
    
    # Calculate chunk information
    num_events, runs_per_chunk, num_chunks = get_chunk_info(num_runs, run_size, chunk_size)
    
    logging.info(f"Processing {num_runs} runs with {num_events} total events")
    logging.info(f"Processing {runs_per_chunk} runs per chunk to get ~{chunk_size} events per file")
    logging.info(f"Total chunks to process: {num_chunks}")
    
    # Create output directory
    output_dir = ensure_output_dir(output_base_dir, dataset_name)
    dataset_name = dataset_name.replace("/", ".")
    
    # Process chunks of runs
    if chunk_index is not None:
        # Single specified chunk
        start_run = chunk_index * runs_per_chunk
        if start_run < num_runs:
            process_chunk_for_hits(
                run_dirs,
                start_run,
                runs_per_chunk,
                Path(output_dir),
                dataset_name,
                run_size,
            )
        else:
            logging.warning(f"Requested chunk_index {chunk_index} is out of range; nothing to do.")
        return

    # Otherwise iterate, possibly capped by max_chunks
    processed = 0
    for start_run in tqdm(range(0, num_runs, runs_per_chunk), desc="Processing chunks"):
        if max_chunks is not None and processed >= int(max_chunks):
            break
        process_chunk_for_hits(
            run_dirs,
            start_run,
            runs_per_chunk,
            Path(output_dir),
            dataset_name,
            run_size,
        )
        processed += 1

def main():
    # Create parser with common arguments
    parser = create_base_parser("Convert EDM4HEP tracker hits to HDF5")
    
    # Add script-specific arguments
    parser.add_argument(
        "--edm4hep-file",
        help="Name of EDM4hep ROOT file",
        type=str,
        default="edm4hep.root"
    )
    
    # Parse args and load config
    args = parser.parse_args()
    config = load_config(args)
    
    logging.info("\nStarting hit conversion with configuration:")
    for key, value in vars(config).items():
        if key != 'config':  # Skip config file path
            logging.info(f"{key}: {value}")
    
    convert_hits(
        config.base_dir,
        config.output_dir,
        config.dataset_name,
        config.chunk_size,
        config.run_size,
        chunk_index=getattr(config, "chunk_index", None),
        max_chunks=getattr(config, "max_chunks", None),
        max_runs=getattr(config, "max_runs", None),
    )

if __name__ == "__main__":
    main() 