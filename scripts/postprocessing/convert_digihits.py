#!/usr/bin/env python3
"""
Convert digitized tracker measurements (measurements.root) to HDF5 or Parquet format.

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
import time

# Use relative imports to avoid conflicts with other utils modules
from utils.path_utils import get_run_paths, make_dir
from utils.driver import iterate_and_process_chunks, local_events_for_run
from utils.track_utils import load_root_file
from utils.parquet_utils import build_parquet_from_flat_df

sys.path.append("/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software/OtherLibraries/pyedm4hep")
from pyedm4hep import EDM4hepEvent, EDM4hepEventBatch


logger = logging.getLogger(__name__)


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


def _merge_measurements_with_tracker(meas_df: pd.DataFrame, tracker_df: pd.DataFrame,
                                     include_meas_cols: List[str] = [],
                                     include_simhits_cols: List[str] = []) -> pd.DataFrame:
    """
    Merge measurements and EDM4hep tracker hits by coordinate matching.

    Requirements handled:
    - Preserve the original order and length of the measurements rows.
    - Allow duplicate coordinates on both sides without producing a cartesian product.
      Achieved by stable 1:1 pairing within each duplicate (x,y,z) group using
      a per-group sequence number (cumcount) on both sides.

    If required coordinate columns are missing, return the measurements unchanged.
    """
    # Guard on required columns
    coord_meas = [c for c in ["true_x", "true_y", "true_z"] if c in meas_df.columns]
    coord_trk = [c for c in ["x", "y", "z"] if c in tracker_df.columns]
    if len(coord_meas) != 3 or len(coord_trk) != 3:
        return meas_df

    # Prepare copies and cast to float32 for stable equality
    meas_df = meas_df.copy()
    tracker_df = tracker_df.copy()
    meas_df.loc[:, ["true_x", "true_y", "true_z"]] = (
        meas_df[["true_x", "true_y", "true_z"]].astype(np.float32)
    )
    tracker_df.loc[:, ["x", "y", "z"]] = (
        tracker_df[["x", "y", "z"]].astype(np.float32)
    )

    # Preserve original measurement order explicitly
    meas_df["_orig_pos"] = np.arange(len(meas_df), dtype=np.int64)

    # Select minimal simulation hits columns to append (intersect with available)
    if not include_simhits_cols:
        include_simhits_cols = [
            "x", "y", "z", "time", "px", "py", "pz", "particle_id", "cellID", "detector", "EDep", "pathLength"
        ]
    rhs_cols = [c for c in include_simhits_cols if c in tracker_df.columns]
    rhs = tracker_df[rhs_cols].copy() if rhs_cols else tracker_df.copy()

    # Select minimal measurement columns to bring through (intersect with available)
    if not include_meas_cols:
        include_meas_cols = [
            "true_x", "true_y", "true_z", "rec_gx", "rec_gy", "rec_gz",
            "volume_id", "layer_id", "surface_id"
        ]
    lhs_cols = [c for c in include_meas_cols if c in meas_df.columns] + ["_orig_pos"]
    lhs = meas_df[lhs_cols].copy() if lhs_cols else meas_df[["_orig_pos"]].copy()

    # Create per-key sequence numbers to avoid cartesian product on duplicates
    lhs["_seq"] = lhs.groupby(["true_x", "true_y", "true_z"], dropna=False).cumcount()
    rhs["_seq"] = rhs.groupby(["x", "y", "z"], dropna=False).cumcount()

    # Perform LEFT merge on keys + sequence number (stable 1:1 pairing)
    merged = pd.merge(
        lhs,
        rhs,
        left_on=["true_x", "true_y", "true_z", "_seq"],
        right_on=["x", "y", "z", "_seq"],
        how="left",
        sort=False,
        copy=False,
    )

    # Drop helper columns and the measurement-side true_* prior to renaming to avoid duplicates
    merged = merged.drop(columns=["_seq", "true_x", "true_y", "true_z"], errors='ignore')

    # Simhits x,y,z become true_x, true_y, true_z
    # Measurements rec_x, rec_y, rec_z become x, y, z
    merged = merged.rename(columns={
        "x": "true_x",
        "y": "true_y",
        "z": "true_z",
        "rec_gx": "x",
        "rec_gy": "y",
        "rec_gz": "z",
        "cellID": "cell_id",
        "EDep": "e_dep",
        "pathLength": "path_length",
    }, errors='ignore')

    # Convert detector strings to integers
    merged = _convert_detector_to_int(merged)

    # Restore original measurement order explicitly
    merged = merged.sort_values("_orig_pos", kind="stable").drop(columns=["_orig_pos"], errors='ignore')

    logger.debug(f"Merged columns: {merged.columns}")

    return merged


def process_event_for_digihits(event_id: int, local_event_num: int, measurements_df: pd.DataFrame, tracker_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Build per-event digitized measurements dataframe, merged with tracker.
    """
    # Filter event slice
    if "event_id" in measurements_df.columns:
        ev_meas = measurements_df[measurements_df.event_id == local_event_num].copy()
    else:
        # Assume the file is already a single-event view
        ev_meas = measurements_df.copy()

    measurements_length = len(ev_meas)
    _t_merge = time.time()
    event_measurements = _merge_measurements_with_tracker(ev_meas, tracker_df)
    merged_length = len(event_measurements)
    logging.debug(
        f"Event {event_id}: merged measurements {measurements_length} -> {merged_length} in {time.time() - _t_merge:.3f}s"
    )

    # Event id is the global id passed in
    event_measurements["event_id"] = event_id
    return event_measurements


def build_parquet_digihits(df: pd.DataFrame, output_file: str) -> None:
    """
    Write digitized measurements to Parquet format.
    
    Args:
        df: Flat DataFrame with event_id and per-hit columns
        output_file: Path to output Parquet file
    """
    if df.empty:
        logger.warning(f"Skipping empty DataFrame for Parquet digihits: {output_file}")
        return
    
    # Use shared utility to group by event and write
    build_parquet_from_flat_df(df, output_file, compression='snappy')


def write_digihits_with_selection(
    df: pd.DataFrame,
    output_file: str,
    columns_keep: List[str] | None = None,
    output_format: str = 'hdf5',
) -> None:
    """
    Write merged digi-hits DataFrame to HDF5 or Parquet with optional column selection.
    
    Args:
        df: DataFrame with digitized hit data
        output_file: Path to output file
        columns_keep: Optional list of columns to keep
        output_format: Output format - 'hdf5' (default) or 'parquet'
    """
    if df.empty:
        return
    if columns_keep:
        cols = [c for c in columns_keep if c in df.columns]
        if 'event_id' not in cols and 'event_id' in df.columns:
            cols = cols + ['event_id']
        df = df[cols].copy()
    
    # Route to appropriate writer based on format
    if output_format == 'parquet':
        build_parquet_digihits(df, output_file)
    else:  # default to hdf5
        build_hdf5_digihits(df, output_file)


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

            # Downcast numeric dtypes for better compression
            float_cols = [c for c in data_df.columns if pd.api.types.is_float_dtype(data_df[c])]
            for c in float_cols:
                data_df[c] = data_df[c].astype('float32')

            # Cast integer columns to smaller types when safe (keep cell_id as int64)
            int_cols = [c for c in data_df.columns if pd.api.types.is_integer_dtype(data_df[c])]
            for c in int_cols:
                if c == 'cell_id':
                    continue
                # prefer int32 to reduce size
                try:
                    data_df[c] = data_df[c].astype('int32')
                except Exception:
                    # fallback if overflow
                    data_df[c] = data_df[c].astype('int64')

            # Prepare records and choose chunking
            records = data_df.to_records(index=False)
            chunk_len = max(1, min(len(records), 65536))

            event_group.create_dataset(
                'measurements',
                data=records,
                compression='gzip',
                compression_opts=6,
                shuffle=True,
                chunks=(chunk_len,)
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

    # Batch approach: caller may still use this legacy function; default to full range
    if not edm4hep_path.exists():
        logging.warning(f"Missing EDM4hep file: {edm4hep_path}")
        return []

    batch = EDM4hepEventBatch(str(edm4hep_path), events=range(run_size))
    hits_all = batch.get_tracker_hits_df()

    run_events: List[pd.DataFrame] = []
    for local_event_num in tqdm(range(run_size), desc="Processing events"):
        global_event_num = run_number * run_size + local_event_num
        ev_hits = hits_all[hits_all.event_id == local_event_num] if not hits_all.empty else None
        ev_df = process_event_for_digihits(global_event_num, local_event_num, meas_df, ev_hits)
        if not ev_df.empty:
            run_events.append(ev_df)

    return run_events


def process_chunk_for_digihits(
    run_dirs: List[Path],
    start_event: int,
    end_event: int,
    start_run: int,
    start_local: int,
    end_run: int,
    end_local: int,
    output_dir: Path,
    dataset_name: str,
    run_size: int,
    force_overwrite: bool = False,
    columns_keep: List[str] | None = None,
    output_format: str = 'hdf5',
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
    # start_event/end_event precomputed by driver (event-based chunking)
    end_run = min(end_run, len(run_dirs) - 1)

    # Determine file extension based on output format
    file_ext = '.parquet' if output_format == 'parquet' else '.h5'
    output_file = Path(output_dir) / f"{dataset_name}.reco.tracker_hits.events{start_event}-{end_event}{file_ext}"
    chunk_start = time.time()
    if output_file.exists() and not force_overwrite:
        logging.info(f"Skipping events {start_event}-{end_event} - exists: {output_file}")
        return

    all_event_dfs: List[pd.DataFrame] = []
    total_rows = 0
    for abs_run in range(start_run, end_run + 1):
        run_dir = run_dirs[abs_run]
        try:
            local_start, local_stop = local_events_for_run(
                start_run=start_run,
                start_local=start_local,
                end_run=end_run,
                end_local=end_local,
                abs_run=abs_run,
                run_size=run_size,
            )
            local_events = (local_start, local_stop)
            local_count = local_stop - local_start

            # Load measurements once per run and (optionally) prefilter to local events
            meas_path = run_dir / "measurements.root"
            if not meas_path.exists():
                logging.warning(f"Missing measurements file: {meas_path}")
                continue
            edm4hep_path = run_dir / "edm4hep.root"
            local_events_str = (
                f"{local_start}-{local_stop-1} (n={local_count})" if local_count > 0 else "<empty>"
            )
            logging.info(
                f"Run {abs_run}: dir={run_dir} edm4hep={edm4hep_path} measurements={meas_path} local_events={local_events_str}"
            )
            _t_meas = time.time()
            meas_df_all = load_root_file(str(meas_path))
            logger.debug(f"Loaded measurements.root for run {abs_run} in {time.time() - _t_meas:.3f}s")
            if "event_nr" in meas_df_all.columns:
                meas_df_all = meas_df_all[meas_df_all.event_nr.isin(range(local_events[0], local_events[1]))].copy()

            # Batch load only needed local events from edm4hep once
            edm4hep_path = run_dir / "edm4hep.root"
            if not edm4hep_path.exists():
                logging.warning(f"Missing EDM4hep file: {edm4hep_path}")
                continue
            _t_batch = time.time()
            batch = EDM4hepEventBatch(str(edm4hep_path), events=local_events)
            hits_all = batch.get_tracker_hits_df()  # load tracker collection lazily
            logger.debug(f"Loaded tracker hits batch for run {abs_run} in {time.time() - _t_batch:.3f}s")

            evs = []
            rows_run = 0
            for local_event_num in range(local_events[0], local_events[1]):
                global_event_num = abs_run * run_size + local_event_num

                # Slice measurements for this local event from in-memory DataFrame
                if "event_nr" in meas_df_all.columns:
                    ev_meas = meas_df_all[meas_df_all.event_nr == local_event_num].copy()
                else:
                    ev_meas = meas_df_all.copy()

                ev_hits = hits_all[hits_all.event_id == local_event_num] if not hits_all.empty else None
                ev_df = process_event_for_digihits(global_event_num, local_event_num, ev_meas, ev_hits)
                if not ev_df.empty:
                    evs.append(ev_df)
                    rows_run += len(ev_df)
            all_event_dfs.extend(evs)
            total_rows += sum(len(df) for df in evs)
            logging.info(
                f"Run {abs_run}: tracker_hits rows={rows_run} events={len(evs)}"
            )
        except Exception as e:
            logging.error(f"Error processing run {abs_run}: {e}")

    if all_event_dfs:
        all_df = pd.concat(all_event_dfs, ignore_index=True)
        if columns_keep:
            cols = [c for c in columns_keep if c in all_df.columns]
            if 'event_id' not in cols and 'event_id' in all_df.columns:
                cols = cols + ['event_id']
            all_df = all_df[cols].copy()
        logging.info(f"Writing {len(all_df)} measurements across {all_df.event_id.nunique()} events -> {output_file} (chunk_time={time.time() - chunk_start:.3f}s)")
        write_digihits_with_selection(all_df, str(output_file), columns_keep=None, output_format=output_format)
    else:
        logging.warning(f"No data to save for events {start_event}-{end_event}")


def convert_digihits(
    base_dir: Path | str,
    output_base_dir: Path | str,
    dataset_name: str,
    chunk_size: int = 1000,
    run_size: int = 10,
    chunk_index: int | None = None,
    max_chunks: int | None = None,
    config_for_cap: dict | None = None,
    columns_keep: List[str] | None = None,
    output_format: str = 'hdf5',
) -> None:
    """
    Convert digitized measurements to HDF5 or Parquet files grouped by event.
    
    Args:
        output_format: Output format - 'hdf5' (default) or 'parquet'
    """
    base_dir = Path(base_dir)
    output_base_dir = Path(output_base_dir)

    run_dirs = get_run_paths(base_dir)

    # Use format-specific subdirectory
    format_subdir = output_format if output_format in ['hdf5', 'parquet'] else 'hdf5'
    output_dir = make_dir(output_base_dir, f"{dataset_name}/{format_subdir}/reco/tracker_hits")
    dataset_name = dataset_name.replace("/", ".")
    
    iterate_and_process_chunks(
        run_dirs=run_dirs,
        run_size=run_size,
        chunk_size=chunk_size,
        config=(
            {"max_chunks": max_chunks} if config_for_cap is None else {**config_for_cap, **({"max_chunks": max_chunks} if max_chunks is not None else {})}
        ),
        chunk_index=chunk_index,
        process_chunk_fn=lambda start_event, end_event, start_run, start_local, end_run, end_local: process_chunk_for_digihits(
            run_dirs,
            start_event,
            end_event,
            start_run,
            start_local,
            end_run,
            end_local,
            output_dir,
            dataset_name,
            run_size,
            columns_keep=columns_keep,
            output_format=output_format,
        ),
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
    # Use common.output_base_dir for postprocessing outputs as well
    output_base_dir = Path(config["common"]["output_base_dir"]) 

    chunk_size = config.get("chunk_size", 1000)
    run_size = config.get("run_size", 10)

    # Extract output format from config (default to hdf5 for backward compatibility)
    output_format = config.get("output_format", "hdf5")

    logging.info("\nStarting digitized hit conversion with configuration:")
    logging.info(f"Campaign: {campaign}, Dataset: {dataset}, Version: {version}")
    logging.info(f"Input directory: {input_base_dir}")
    logging.info(f"Output root: {output_base_dir}")
    logging.info(f"Output format: {output_format}")
    logging.info(f"Chunk size: {chunk_size}, Run size: {run_size}")

    # Save config path for optional cap inference
    global _CONFIG_PATH_FOR_LOGGING  # type: ignore[declared-but-not-used]
    _CONFIG_PATH_FOR_LOGGING = args.config

    convert_digihits(
        input_base_dir,
        output_base_dir,
        f"{campaign}/{dataset}/{version}",
        chunk_size,
        run_size,
        args.chunk_index,
        config.get("max_chunks"),
        config,
        columns_keep=config.get("digihits_columns_keep"),
        output_format=output_format,
    )


if __name__ == "__main__":
    main()


