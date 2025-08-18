#!/usr/bin/env python3
"""
Convert digitized tracker measurements (measurements.root) to HDF5 format.

This merges measurements with EDM4hep tracker hits to attach detector labels
and truth particle links, using the true_x/true_y/true_z coordinates present
in the measurements file to match to EDM4hep hit x/y/z.
"""

import argparse
import yaml
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import h5py
import uproot
from tqdm import tqdm
import logging
import sys

# Use relative imports to avoid conflicts with other utils modules
from utils.path_utils import get_run_paths, make_dir, get_chunk_info
from utils.track_utils import load_root_file

sys.path.append("/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software/OtherLibraries/pyedm4hep")
from pyedm4hep import EDM4hepEvent


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def _convert_detector_to_int(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert detector column from strings to integers to avoid HDF5 object dtype issues.
    """
    if 'detector' not in df.columns:
        return df
    
    detector_mapping = {
        'PixelBarrelReadout': 0,
        'PixelEndcapReadout': 1, 
        'ShortStripBarrelReadout': 2,
        'ShortStripEndcapReadout': 3,
        'LongStripBarrelReadout': 4,
        'LongStripEndcapReadout': 5
    }
    
    df = df.copy()
    df['detector'] = df['detector'].map(detector_mapping)
    return df


def _merge_measurements_with_tracker(meas_df: pd.DataFrame, tracker_df: pd.DataFrame, include_meas_cols: List[str] = [], include_simhits_cols: List[str] = []) -> pd.DataFrame:
    """
    Merge measurements and EDM4hep tracker hits by coordinate matching.

    Strategy: cast both coordinate sets to float32 and merge on equality.
    This mirrors the notebook exploration where float32 alignment produced
    exact matches. If columns are missing, the merge safely degrades.

    Also brings through selected geometry identifiers from the measurements
    file (e.g. volume_id, layer_id, surface_id) when present.
    """
    # Guard on required columns
    coord_meas = [c for c in ["true_x", "true_y", "true_z"] if c in meas_df.columns]
    coord_trk = [c for c in ["x", "y", "z"] if c in tracker_df.columns]
    if len(coord_meas) != 3 or len(coord_trk) != 3:
        return meas_df

    # Cast to float32 for stable equality
    meas_df = meas_df.copy()
    tracker_df = tracker_df.copy()
    meas_df.loc[:, ["true_x", "true_y", "true_z"]] = meas_df[["true_x", "true_y", "true_z"]].astype(np.float32)
    tracker_df.loc[:, ["x", "y", "z"]] = tracker_df[["x", "y", "z"]].astype(np.float32)

    # Select minimal simulation hits columns to append
    if not include_simhits_cols:
        include_simhits_cols = [
            "x", "y", "z", "time", "px", "py", "pz", "particle_id", "cellID", "detector", "EDep", "pathLength"
        ]
    rhs = tracker_df[include_simhits_cols].copy()

    # Select minimal measurement columns to append
    if not include_meas_cols:
        include_meas_cols = [
            "true_x", "true_y", "true_z", "rec_x", "rec_y", "rec_z",
            "volume_id", "layer_id", "surface_id"
        ]
    lhs = meas_df[include_meas_cols].copy()

    merged = pd.merge(
        lhs,
        rhs,
        left_on=["true_x", "true_y", "true_z"],
        right_on=["x", "y", "z"],
        how="left"
    )

    # Drop the original true_x, true_y, true_z from measurements (now duplicated)
    merged = merged.drop(columns=["true_x", "true_y", "true_z"])

    # Rename columns to match final naming convention
    # Simhits x,y,z become true_x, true_y, true_z
    # Measurements rec_x, rec_y, rec_z become x, y, z
    merged = merged.rename(columns={
        "x": "true_x",
        "y": "true_y", 
        "z": "true_z",
        "rec_x": "x",
        "rec_y": "y",
        "rec_z": "z",
        "cellID": "cell_id",
        "EDep": "e_dep",
        "pathLength": "path_length"
    })

    # Convert detector strings to integers
    merged = _convert_detector_to_int(merged)

    return merged


def process_event_for_digihits(event_id: int, local_event_num: int, measurements_df: pd.DataFrame, tracker_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Build per-event digitized measurements dataframe, merged with tracker.
    """
    # Filter event slice
    if "event_nr" in measurements_df.columns:
        ev_meas = measurements_df[measurements_df.event_nr == local_event_num].copy()
    else:
        # Assume the file is already a single-event view
        ev_meas = measurements_df.copy()

    measurements_length = len(ev_meas)
    event_measurements = _merge_measurements_with_tracker(ev_meas, tracker_df)
    merged_length = len(event_measurements)
    logging.debug(
        f"Event {event_id}: merged measurements {measurements_length} -> {merged_length}"
    )

    # Event id is the global id passed in
    event_measurements["event_id"] = event_id
    return event_measurements


def build_hdf5_digihits(df: pd.DataFrame, output_file: str) -> None:
    """
    Write digitized measurements to HDF5 under /events/event_#/measurements.
    """
    with h5py.File(output_file, 'a') as f:
        events_group = f.create_group('events') if 'events' not in f else f['events']

        for event_id, event_df in df.groupby('event_id'):
            event_group_name = f'event_{event_id}'
            if event_group_name in events_group:
                # Remove existing group to avoid conflicts
                del events_group[event_group_name]
            event_group = events_group.create_group(event_group_name)

            # Drop event_id for storage
            data_df = event_df.drop(columns=['event_id'], errors='ignore')

            event_group.create_dataset(
                'measurements',
                data=data_df.to_records(index=False),
                # compression='gzip', # TODO: Investigate compression (reduces size by 2x)
                # compression_opts=9
            )


def process_run_for_digihits(run_dir: Path, run_number: int, run_size: int) -> List[pd.DataFrame]:
    """
    Process all events in a run directory into a list of dataframes.
    """
    run_dir = Path(run_dir)
    measurements_path = run_dir / "measurements.root"
    edm4hep_path = run_dir / "edm4hep.root"

    if not measurements_path.exists():
        logging.warning(f"Missing measurements file: {measurements_path}")
        return []

    try:
        meas_df = load_root_file(str(measurements_path))
    except Exception as e:
        logging.error(f"Failed to load measurements: {e}")
        return []

    # If EDM4hep exists, we will merge per-event with tracker
    edm_available = edm4hep_path.exists()

    run_events: List[pd.DataFrame] = []
    for local_event_num in tqdm(range(run_size), desc="Processing events"):
        global_event_num = run_number * run_size + local_event_num

        tracker_df = None
        if edm_available:
            try:
                event = EDM4hepEvent(str(edm4hep_path), event_index=local_event_num)
                tracker_df = event.get_tracker_hits_df()
            except Exception as e:
                logging.warning(f"Failed to load tracker for event {local_event_num} in {run_dir}: {e}")

        ev_df = process_event_for_digihits(global_event_num, local_event_num, meas_df, tracker_df)
        if not ev_df.empty:
            run_events.append(ev_df)

    return run_events


def process_chunk_for_digihits(
    run_dirs: List[Path],
    start_run: int,
    runs_per_chunk: int,
    output_dir: Path,
    dataset_name: str,
    run_size: int,
    force_overwrite: bool = False,
) -> None:
    """
    Process a chunk of runs and write one HDF5 file for the chunk.

    Args:
        run_dirs: List of run directories to process
        start_run: Index of the first run to process
        runs_per_chunk: Number of runs to process in each chunk
        output_dir: Directory to write the output HDF5 file
        dataset_name: Name of the dataset
        run_size: Number of events per run
        force_overwrite: Whether to overwrite existing output file

    Returns:
        None
    """
    start_event = start_run * run_size
    end_run = min(start_run + runs_per_chunk, len(run_dirs))
    end_event = (end_run * run_size) - 1

    output_file = Path(output_dir) / f"{dataset_name}.reco.tracker_hits.events{start_event}-{end_event}.h5"
    if output_file.exists() and not force_overwrite:
        logging.info(f"Skipping events {start_event}-{end_event} - exists: {output_file}")
        return

    all_event_dfs: List[pd.DataFrame] = []
    total_rows = 0
    for run_idx, run_dir in enumerate(tqdm(run_dirs[start_run:end_run], desc="Processing runs", leave=False)):
        try:
            evs = process_run_for_digihits(run_dir, start_run + run_idx, run_size)
            all_event_dfs.extend(evs)
            total_rows += sum(len(df) for df in evs)
        except Exception as e:
            logging.error(f"Error processing run {start_run + run_idx}: {e}")

    if all_event_dfs:
        all_df = pd.concat(all_event_dfs, ignore_index=True)
        logging.info(f"Writing {len(all_df)} measurements across {all_df.event_id.nunique()} events -> {output_file}")
        build_hdf5_digihits(all_df, str(output_file))
    else:
        logging.warning(f"No data to save for events {start_event}-{end_event}")


def convert_digihits(
    base_dir: Path | str,
    output_base_dir: Path | str,
    dataset_name: str,
    chunk_size: int = 1000,
    run_size: int = 10,
    chunk_index: int | None = None,
) -> None:
    """
    Convert digitized measurements to HDF5 files grouped by event.
    """
    base_dir = Path(base_dir)
    output_base_dir = Path(output_base_dir)

    run_dirs = get_run_paths(base_dir)
    num_runs = len(run_dirs)

    num_events, runs_per_chunk, num_chunks = get_chunk_info(num_runs, run_size, chunk_size)
    logging.info(f"Processing {num_runs} runs ({num_events} events), {runs_per_chunk} runs/chunk, {num_chunks} chunks")

    output_dir = make_dir(output_base_dir, f"{dataset_name}/reco/tracker_hits")
    dataset_name = dataset_name.replace("/", ".")
    
    if chunk_index is None:
        for start_run in tqdm(range(0, num_runs, runs_per_chunk), desc="Processing chunks"):
            process_chunk_for_digihits(
                run_dirs,
                start_run,
                runs_per_chunk,
                output_dir,
                dataset_name,
                run_size,
            )
    else:
        start_run = chunk_index * runs_per_chunk
        if start_run >= num_runs:
            logging.warning(f"Chunk index {chunk_index} out of range. num_runs={num_runs}, runs_per_chunk={runs_per_chunk}")
            return
        process_chunk_for_digihits(
            run_dirs,
            start_run,
            runs_per_chunk,
            output_dir,
            dataset_name,
            run_size,
        )


def main():
    # Align CLI/config handling and file naming with convert_tracks.py
    parser = argparse.ArgumentParser(description="Convert EDM4HEP digitized tracker measurements to HDF5")
    parser.add_argument(
        "--config",
        help="Path to YAML config file",
        type=str,
        required=True
    )
    parser.add_argument(
        "--chunk-index",
        help="Optional chunk index to process (for distributed runs)",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    campaign = config["campaign"]
    dataset = config["dataset"]
    version = config["version"]

    input_base_dir = Path(config["common"]["output_base_dir"]) / campaign / dataset / version
    output_base_dir = Path(config["common"]["staging_dir"])
    output_path = config.get("output_path", f"{campaign}/{dataset}/{version}/reco/digihits")

    chunk_size = config.get("chunk_size", 1000)
    run_size = config.get("run_size", 10)

    logging.info("\nStarting digitized hit conversion with configuration:")
    logging.info(f"Campaign: {campaign}, Dataset: {dataset}, Version: {version}")
    logging.info(f"Input directory: {input_base_dir}")
    logging.info(f"Output directory: {output_base_dir}/{output_path}")
    logging.info(f"Chunk size: {chunk_size}, Run size: {run_size}")

    convert_digihits(
        input_base_dir,
        output_base_dir,
        f"{campaign}/{dataset}/{version}",
        chunk_size,
        run_size,
        args.chunk_index,
    )


if __name__ == "__main__":
    main()


