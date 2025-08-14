#!/usr/bin/env python3
"""
Convert digitized tracker measurements (measurements.root) to HDF5 format.

This merges measurements with EDM4hep tracker hits to attach detector labels
and truth particle links, using the true_x/true_y/true_z coordinates present
in the measurements file to match to EDM4hep hit x/y/z.
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import h5py
import uproot
from tqdm import tqdm
import logging

from utils.utils import get_run_paths, ensure_output_dir, get_chunk_info
from utils.config import create_base_parser, load_config
from utils.track_utils import load_root_file
from utils.edm4hep_utils import load_edm4hep_file


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def _merge_measurements_with_tracker(meas_df: pd.DataFrame, tracker_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge measurements and EDM4hep tracker hits by coordinate matching.

    Strategy: cast both coordinate sets to float32 and merge on equality.
    This mirrors the notebook exploration where float32 alignment produced
    exact matches. If columns are missing, the merge safely degrades.
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

    # Select minimal tracker columns to append
    take_cols = [
        c for c in [
            "x", "y", "z", "px", "py", "pz", "particle_id", "detector",
            "r", "R", "phi", "theta", "eta", "pt"
        ] if c in tracker_df.columns
    ]
    rhs = tracker_df[take_cols].copy()

    merged = pd.merge(
        meas_df,
        rhs,
        left_on=["true_x", "true_y", "true_z"],
        right_on=["x", "y", "z"],
        how="left"
    )

    # Drop the duplicated x,y,z from tracker if they exist
    for c in ["x", "y", "z"]:
        if c in merged.columns:
            merged = merged.drop(columns=[c])

    return merged


def process_event_for_digihits(event_id: int, local_event_num: int, measurements_df: pd.DataFrame, tracker_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Build per-event digitized measurements dataframe, optionally merged with tracker.
    """
    # Filter event slice
    if "event_nr" in measurements_df.columns:
        ev_meas = measurements_df[measurements_df.event_nr == local_event_num].copy()
    elif "event_id" in measurements_df.columns:
        ev_meas = measurements_df[measurements_df.event_id == local_event_num].copy()
    else:
        # Assume the file is already a single-event view
        ev_meas = measurements_df.copy()

    # Keep native dtypes; avoid casting here to preserve original info

    # Optional merge with tracker to add labels/kinematics
    if tracker_df is not None and not tracker_df.empty:
        before = len(ev_meas)
        ev_meas = _merge_measurements_with_tracker(ev_meas, tracker_df)
        after = len(ev_meas)
        # Basic coverage logging if detector/particle_id exist after merge
        missing_det = ev_meas["detector"].isna().sum() if "detector" in ev_meas.columns else None
        missing_pid = ev_meas["particle_id"].isna().sum() if "particle_id" in ev_meas.columns else None
        logging.debug(
            f"Event {event_id}: merged measurements {before} -> {after}, missing detector={missing_det}, missing particle_id={missing_pid}"
        )

    # Event id is the global id passed in
    ev_meas["event_id"] = event_id
    return ev_meas


def build_hdf5_digihits(df: pd.DataFrame, output_file: str) -> None:
    """
    Write digitized measurements to HDF5 under /events/event_#/measurements.
    Uses a structured dtype inferred from dataframe dtypes, with utf8 strings.
    """
    # Prepare string dtype helper
    str_dtype = h5py.string_dtype(encoding='utf8')

    with h5py.File(output_file, 'a') as f:
        events_group = f.create_group('events') if 'events' not in f else f['events']

        for event_id, event_df in df.groupby('event_id'):
            event_group = events_group.create_group(f'event_{event_id}')

            # Drop event_id for storage
            data_df = event_df.drop(columns=['event_id'], errors='ignore')

            # Build a safe dtype mapping
            name_to_dtype = {}
            for name, series in data_df.items():
                if pd.api.types.is_integer_dtype(series):
                    name_to_dtype[name] = np.int64
                elif pd.api.types.is_float_dtype(series):
                    name_to_dtype[name] = np.float64
                elif pd.api.types.is_bool_dtype(series):
                    name_to_dtype[name] = np.int8
                else:
                    name_to_dtype[name] = str_dtype

            dt = np.dtype([(n, t) for n, t in name_to_dtype.items()])
            out = np.empty(len(data_df), dtype=dt)
            for name in data_df.columns:
                target_t = name_to_dtype[name]
                if isinstance(target_t, h5py.Datatype) or target_t == str_dtype:
                    out[name] = data_df[name].astype(str).values
                else:
                    out[name] = np.asarray(data_df[name].values, dtype=target_t)

            event_group.create_dataset(
                'measurements',
                data=out,
                compression='gzip',
                compression_opts=9
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
    for local_event_num in range(run_size):
        global_event_num = run_number * run_size + local_event_num

        tracker_df = None
        if edm_available:
            try:
                event = load_edm4hep_file(str(edm4hep_path), event_num=local_event_num, collections=["tracker"])
                tracker_df = event.get("tracker_df")
            except Exception as e:
                logging.warning(f"Failed to load tracker for event {local_event_num} in {run_dir}: {e}
")

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
) -> None:
    """
    Process a chunk of runs and write one HDF5 file for the chunk.
    """
    start_event = start_run * run_size
    end_run = min(start_run + runs_per_chunk, len(run_dirs))
    end_event = (end_run * run_size) - 1

    output_file = Path(output_dir) / f"{dataset_name}.events{start_event}-{end_event}.h5"
    if output_file.exists():
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

    output_dir = ensure_output_dir(str(output_base_dir), dataset_name)
    dataset_name = dataset_name.replace("/", ".")

    for start_run in tqdm(range(0, num_runs, runs_per_chunk), desc="Processing chunks"):
        process_chunk_for_digihits(
            run_dirs,
            start_run,
            runs_per_chunk,
            Path(output_dir),
            dataset_name,
            run_size,
        )


def main():
    parser = create_base_parser("Convert EDM4HEP digitized tracker measurements to HDF5")
    args = parser.parse_args()
    config = load_config(args)

    logging.info("\nStarting digitized hit conversion with configuration:")
    for key, value in vars(config).items():
        if key != 'config':
            logging.info(f"{key}: {value}")

    convert_digihits(
        config.base_dir,
        config.output_dir,
        config.dataset_name,
        config.chunk_size,
        config.run_size,
    )


if __name__ == "__main__":
    main()


