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
import logging

from utils.path_utils import get_run_paths, make_dir
from utils.driver import iterate_and_process_chunks, local_events_for_run
from utils.edm4hep_utils import pixel_readouts, strip_readouts

from utils.track_utils import (
    load_root_file,
    write_tracks_with_selection,
    normalize_tracksummary_df,
)
from convert_digihits import process_event_for_digihits

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
    track_fitting_df_event: pd.DataFrame,
    *,
    digihits_run_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Process a single event using tracksummary ROOT and digitized hits only.
    """
    if track_fitting_df_event is None or track_fitting_df_event.empty:
        return pd.DataFrame()

    # Use prebuilt per-event slice of track fitting summary
    track_fitting_df = (
        track_fitting_df_event.rename(columns={"track_nr": "track_id"})
        if "track_nr" in track_fitting_df_event.columns
        else track_fitting_df_event.copy()
    )

    if "track_id" not in track_fitting_df.columns:
        raise ValueError("tracksummary slice is missing track_id/track_nr column")

    # Per-event digitized measurements/hits (global event id)
    ev_meas = digihits_run_df[digihits_run_df.event_id == global_event_num].reset_index(drop=True)

    def to_hit_array(ids_list) -> np.ndarray:
        if ids_list is None:
            return np.array([], dtype=np.int32)
        return np.asarray(ids_list, dtype=np.int32)

    def majority_from_ids(ids_list) -> int | float:
        ids = to_hit_array(ids_list)
        if ids.size == 0:
            return np.nan
        try:
            labels = ev_meas.loc[ids, "particle_id"]
        except Exception:
            return np.nan
        if len(labels) == 0:
            return np.nan
        mode_vals = labels.mode()
        return mode_vals.iat[0] if len(mode_vals) else np.nan

    n_tracks = len(track_fitting_df)
    if "measurementIDs" in track_fitting_df.columns:
        # Legacy ACTS: per-track measurement-index list present → build hit_ids
        # and derive majority particle by majority vote over the matched hits.
        hit_ids_series = track_fitting_df["measurementIDs"].apply(to_hit_array)
        majority_particle_ids = track_fitting_df["measurementIDs"].apply(majority_from_ids)
    else:
        # Current ACTS (Paul's branch) dropped the measurementIDs branch from
        # RootTrackSummaryWriter — only nMeasurements (a count) survives, and the
        # per-hit list lives only in the in-memory measurement_simhits_map that
        # feeds the native Arrow track writer. We can no longer reconstruct
        # hit_ids from ROOT, so emit empty lists and take majority_particle_id
        # from the majorityParticleId_particle barcode component when present.
        logging.warning(
            "tracksummary has no measurementIDs branch (ACTS API change); "
            "hit_ids will be empty for v1 tracks — cross-check hit_ids via the "
            "native Arrow writer instead."
        )
        hit_ids_series = pd.Series([np.array([], dtype=np.int32)] * n_tracks,
                                   index=track_fitting_df.index)
        if "majorityParticleId_particle" in track_fitting_df.columns:
            majority_particle_ids = track_fitting_df["majorityParticleId_particle"]
        else:
            majority_particle_ids = pd.Series([np.nan] * n_tracks,
                                              index=track_fitting_df.index)

    # Combine data
    track_finding_data = {
        "event_id": global_event_num,
        "track_id": track_fitting_df["track_id"].values,
        "num_hits": hit_ids_series.apply(len).values,
        "hit_ids": hit_ids_series.values,
        "majority_particle_id": majority_particle_ids.values,
    }

    # Carry over standard summary quantities when present
    for col, out_name in [
        ("nMeasurements", "num_measurements"),
        ("nOutliers", "num_outliers"),
        ("nHoles", "num_holes"),
        ("nSharedHits", "num_shared_hits"),
        ("chi2", "chi2"),
        ("nMajorityHits", "nMajorityHits"),
        ("trackClassification", "trackClassification"),
    ]:
        if col in track_fitting_df.columns:
            track_finding_data[out_name] = track_fitting_df[col].values

    track_fitting_data = {
        "event_id": global_event_num,
        "track_id": track_fitting_df.track_id.values,
        "d0": track_fitting_df.eLOC0_fit.values if "eLOC0_fit" in track_fitting_df else [],
        "z0": track_fitting_df.eLOC1_fit.values if "eLOC1_fit" in track_fitting_df else [],
        "phi": track_fitting_df.ePHI_fit.values if "ePHI_fit" in track_fitting_df else [],
        "theta": track_fitting_df.eTHETA_fit.values if "eTHETA_fit" in track_fitting_df else [],
        "qop": track_fitting_df.eQOP_fit.values if "eQOP_fit" in track_fitting_df else [],
        "time": track_fitting_df.eT_fit.values if "eT_fit" in track_fitting_df else [],
        "d0_truth": track_fitting_df.t_d0.values if "t_d0" in track_fitting_df else [],
        "z0_truth": track_fitting_df.t_z0.values if "t_z0" in track_fitting_df else [],
        "phi_truth": track_fitting_df.t_phi.values if "t_phi" in track_fitting_df else [],
        "theta_truth": track_fitting_df.t_theta.values if "t_theta" in track_fitting_df else [],
        "charge_truth": track_fitting_df.t_charge.values if "t_charge" in track_fitting_df else [],
        "p_truth": track_fitting_df.t_p.values if "t_p" in track_fitting_df else [],
        "pT_truth": track_fitting_df.t_pT.values if "t_pT" in track_fitting_df else [],
        "time_truth": track_fitting_df.t_time.values if "t_time" in track_fitting_df else [],
    }

    full_track_df = pd.DataFrame(track_finding_data)
    logging.info(f"Full track dataframe columns: {full_track_df.columns}")
    event_df = full_track_df.merge(
        pd.DataFrame(track_fitting_data),
        on=["event_id", "track_id"],
    )
    logging.info(f"Event dataframe columns: {event_df.columns}")
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
    file_patterns: dict,
    local_range: tuple[int, int] | None = None,
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
    
    try:
        # Verify files exist before attempting to process
        tracksummary_path = run_dir / file_patterns["tracksummary_file"]
        edm4hep_path = run_dir / file_patterns["edm4hep_file"]
        measurements_path = run_dir / "measurements.root"
        
        if not tracksummary_path.exists():
            raise FileNotFoundError(f"Track summary file not found: {tracksummary_path}")
        if not edm4hep_path.exists():
            raise FileNotFoundError(f"EDM4hep file not found: {edm4hep_path}")
        if not measurements_path.exists():
            raise FileNotFoundError(f"Measurements file not found: {measurements_path}")
        
        # Load and normalize track summary data once for the whole run into a per-run DataFrame
        included_tracksummary_columns = [
            "event_id",
            "event_nr",
            "track_id",
            "track_nr",
            "measurementIDs",
            "eLOC0_fit",
            "eLOC1_fit",
            "ePHI_fit",
            "eTHETA_fit",
            "eQOP_fit",
            "eT_fit",
            "t_d0",
            "t_z0",
            "t_phi",
            "t_theta",
            "t_charge",
            "t_p",
            "t_pT",
            "t_time",
        ]
        df_raw = load_root_file(
            str(tracksummary_path),
            included_columns=included_tracksummary_columns,
        )
        track_fitting_df_run = normalize_tracksummary_df(df_raw)

        # Build run-level digihits by merging measurements with tracker hits once per run
        try:
            meas_df_all = uproot.open(measurements_path)
        except Exception:
            # Fallback to track_utils helper if needed
            from utils.track_utils import load_root_file as _load
            meas_df_all = _load(measurements_path)
        from pyedm4hep import EDM4hepEventBatch
        batch = EDM4hepEventBatch(str(edm4hep_path), events=local_range)
        hits_all = batch.get_tracker_hits_df()
        evs_for_run: List[pd.DataFrame] = []
        for local_event_for_merge in range(run_size):
            ev_hits = hits_all[hits_all.event_id == local_event_for_merge] if not hits_all.empty else None
            if isinstance(meas_df_all, pd.DataFrame):
                ev_meas = meas_df_all[meas_df_all.event_nr == local_event_for_merge].copy() if 'event_nr' in meas_df_all.columns else meas_df_all.copy()
            else:
                # If uproot file-like, fallback to helper
                from utils.track_utils import load_root_file as _load
                ev_meas = _load(measurements_path, event_id=local_event_for_merge)
            ev_df = process_event_for_digihits(run_number * run_size + local_event_for_merge, local_event_for_merge, ev_meas, ev_hits)
            if not ev_df.empty:
                evs_for_run.append(ev_df)
        digihits_run_df = pd.concat(evs_for_run, ignore_index=True) if evs_for_run else pd.DataFrame()
        
        run_events = []
        start_ev, stop_ev = (0, run_size) if local_range is None else local_range
        for local_event_num in range(start_ev, stop_ev):
            try:
                # Calculate global event number
                global_event_num = run_number * run_size + local_event_num
                
                # Slice per-event fitting rows by event_id == local_event_num
                if track_fitting_df_run is not None and not track_fitting_df_run.empty:
                    if "event_id" in track_fitting_df_run.columns:
                        per_event_fit = track_fitting_df_run[
                            track_fitting_df_run["event_id"] == local_event_num
                        ].copy()
                    elif "event_nr" in track_fitting_df_run.columns:
                        per_event_fit = track_fitting_df_run[
                            track_fitting_df_run["event_nr"] == local_event_num
                        ].copy()
                    else:
                        per_event_fit = pd.DataFrame()
                else:
                    per_event_fit = pd.DataFrame()
                event_df = process_event_for_tracks(
                    run_dir,
                    local_event_num,
                    global_event_num,
                    per_event_fit,
                    digihits_run_df=digihits_run_df,
                )
                run_events.append(event_df)
                
            except Exception as e:
                print(f"  Skipping event {local_event_num} in run {run_number} due to error: {str(e)}")
                continue
                
        return run_events
    except Exception as e:
        # Re-raise the exception with run information to be caught by the caller
        raise type(e)(f"Error processing run {run_number}: {str(e)}")

def process_chunk_for_tracks(
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
    file_patterns: dict,
    *,
    columns_keep: List[str] | None = None,
    output_format: str = 'hdf5',
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
    end_run = min(end_run, len(run_dirs) - 1)
    
    # Determine file extension based on output format
    file_ext = '.parquet' if output_format == 'parquet' else '.h5'
    output_file = output_dir / f"{dataset_name}.events{start_event}-{end_event}{file_ext}"
    
    # Skip if file already exists
    if output_file.exists():
        print(f"\nSkipping events {start_event}-{end_event} - output file already exists: {output_file}")
        return
        
    # Process each run in chunk (event-sliced)
    all_track_data = []
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
            local_events = range(local_start, local_stop)
            local_events_list = list(local_events)
            local_count = len(local_events_list)
            local_events_str = (
                f"{local_start}-{local_stop-1} (n={local_count})" if local_count > 0 else "<empty>"
            )
            logging.info(
                f"Run {abs_run}: dir={run_dir} local_events={local_events_str}"
            )

            run_events_all = process_run_for_tracks(run_dir, abs_run, run_size, file_patterns, (local_start, local_stop))
            run_events = []
            for df in run_events_all:
                if df.empty:
                    continue
                first_global = int(df.event_id.iloc[0])
                local_id = first_global - abs_run * run_size
                if local_id in local_events_list:
                    run_events.append(df)
            all_track_data.extend(run_events)
            rows_run = sum(len(df) for df in run_events)
            logging.info(
                f"Run {abs_run}: tracks rows={rows_run} events={len(run_events)}"
            )
        except Exception as e:
            print(f"\nSkipping run {abs_run} due to error: {str(e)}")
            continue
            
    # Save chunk to HDF5 or Parquet
    if all_track_data:
        all_events_df = pd.concat(all_track_data, ignore_index=True)
        write_tracks_with_selection(all_events_df, str(output_file), columns_keep=columns_keep, output_format=output_format)
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
    campaign = config["campaign"]
    dataset = config["dataset"]
    version = config["version"]

    # Build paths from config
    input_base_dir = Path(config["common"]["output_base_dir"]) / campaign / dataset / version
    output_base_dir = Path(config["common"]["output_base_dir"]) 
    
    # Extract output format from config (default to hdf5 for backward compatibility)
    output_format = config.get("output_format", "hdf5")
    format_subdir = output_format if output_format in ['hdf5', 'parquet'] else 'hdf5'
    output_path = config.get("output_path", f"{campaign}/{dataset}/{version}/{format_subdir}/reco/tracks")

    # Processing parameters
    chunk_size = config.get("chunk_size", 1000)
    run_size = config.get("run_size", 10)

    # File patterns
    file_patterns = {
        "tracksummary_file": config.get("tracksummary_file", "tracksummary_ambi.root"),
        "edm4hep_file": config.get("edm4hep_file", "edm4hep.root"),
    }
    
    print("\nStarting track conversion with configuration:")
    print(f"Campaign: {campaign}, Dataset: {dataset}, Version: {version}")
    print(f"Input directory: {input_base_dir}")
    print(f"Output directory: {output_base_dir}/{output_path}")
    print(f"Output format: {output_format}")
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
    output_dir = make_dir(output_base_dir, output_path)
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
        
    dataset_name = output_path.replace("/", ".")
    
    # Use shared chunk driver with caps reflected in tqdm (supports interactive caps)
    iterate_and_process_chunks(
        run_dirs=run_dirs,
        run_size=run_size,
        chunk_size=chunk_size,
        config=config,
        chunk_index=args.chunk_index,
        process_chunk_fn=lambda start_event, end_event, start_run, start_local, end_run, end_local: process_chunk_for_tracks(
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
            file_patterns,
            columns_keep=config.get("tracks_columns_keep"),
            output_format=output_format,
        ),
    )

if __name__ == "__main__":
    main()