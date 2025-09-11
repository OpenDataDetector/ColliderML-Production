#!/usr/bin/env python3
"""
Convert EDM4hep particle data to HDF5 format.

This script processes particle information from EDM4hep files and creates
structured HDF5 files with particle properties and hit count statistics.
"""

import argparse
import yaml
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

import h5py
from tqdm import tqdm
import logging
import sys

# Use relative imports to avoid conflicts with other utils modules
from utils.path_utils import get_run_paths, make_dir
from utils.driver import iterate_and_process_chunks
from utils.track_utils import load_root_file

sys.path.append("/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software/OtherLibraries/pyedm4hep")
from pyedm4hep import EDM4hepEvent, EDM4hepEventBatch

logger = logging.getLogger(__name__)

def process_event_for_particles(
    event_id: int,
    local_event_num: int,
    edm4hep_path: str,
    digi_particles_df: pd.DataFrame | None = None,
    preloaded_particles_df: pd.DataFrame | None = None,
    preloaded_parents_df: pd.DataFrame | None = None,
    min_particle_energy: float | None = None,
    min_tracker_hits: int | None = None,
) -> pd.DataFrame:
    """
    Process particle data for a single event.
    
    Args:
        event_id: Global event number
        local_event_num: Local event number within the run
        edm4hep_path: Path to EDM4hep file
        
    Returns:
        DataFrame containing particle data for this event
    """
    try:
        # Use preloaded per-event slice if provided; otherwise read from file
        if preloaded_particles_df is not None:
            particles_df = preloaded_particles_df.copy()
        else:
            event = EDM4hepEvent(edm4hep_path, event_index=local_event_num)
            particles_df = event.get_particles_df()
        
        # Reset index and add particle_id
        particles_df.reset_index(drop=True, inplace=True)
        particles_df["particle_id"] = particles_df.index
        
        # Normalize column names if needed
        if "pdg_id" not in particles_df.columns and "PDG" in particles_df.columns:
            particles_df["pdg_id"] = particles_df["PDG"]
        
        # If available, merge in vertex_primary from particles.root by matching on phase-space columns
        if digi_particles_df is not None and not digi_particles_df.empty:
            # Normalize event id column name
            if "event_id" not in digi_particles_df.columns and "event_nr" in digi_particles_df.columns:
                digi_particles_df = digi_particles_df.rename(columns={"event_nr": "event_id"})

            # Select this local event's digi particles
            local_digi = digi_particles_df[digi_particles_df.get("event_id", -1) == local_event_num].copy()

            # Ensure required merge columns exist
            merge_cols = ["vx", "vy", "vz", "px", "py", "pz"]
            if all(col in particles_df.columns for col in merge_cols) and all(col in local_digi.columns for col in merge_cols):
                # Align dtypes for robust merging
                for col in merge_cols:
                    particles_df[col] = particles_df[col].astype("float32")
                    local_digi[col] = local_digi[col].astype("float32")

                right_cols = merge_cols + [c for c in ["vertex_primary"] if c in local_digi.columns]
                if right_cols:
                    particles_df = pd.merge(
                        particles_df,
                        local_digi[right_cols],
                        on=merge_cols,
                        how="inner",
                    )

        # Assign parent_id (first parent) when link info is available
        if preloaded_parents_df is not None and not preloaded_parents_df.empty:
            # Ensure we have the link-range columns
            if {"parents_begin", "parents_end"}.issubset(particles_df.columns):
                parent_id_series = pd.Series([np.nan] * len(particles_df), dtype="float64")
                # Rows with at least one parent
                has_parent = particles_df["parents_end"].values > particles_df["parents_begin"].values
                if has_parent.any():
                    begin_idx = particles_df.loc[has_parent, "parents_begin"].astype(int).values
                    try:
                        # parents df is per-event; iloc indices refer to per-event flattened list
                        parent_ids = preloaded_parents_df.iloc[begin_idx]["particle_id"].values
                        parent_id_series.loc[has_parent] = parent_ids
                    except Exception:
                        pass
                particles_df = particles_df.copy()
                particles_df["parent_id"] = parent_id_series

        # Apply configurable minimum energy filter if requested
        if min_particle_energy is not None and "energy" in particles_df.columns:
            try:
                particles_df = particles_df[particles_df["energy"] >= float(min_particle_energy)]
            except Exception:
                pass

        # Apply configurable minimum tracker hits filter if available
        if min_tracker_hits is not None and "num_tracker_hits" in particles_df.columns:
            try:
                particles_df = particles_df[particles_df["num_tracker_hits"] >= int(min_tracker_hits)]
                print(f"Event {event_id}: {len(particles_df)} particles after min_tracker_hits filter")
            except Exception:
                pass

        # Select relevant particle columns (include vertex_primary/parent_id if present)
        desired_columns = [
            "particle_id",
            "pdg_id", 
            "mass",
            "energy",
            "charge",
            "vx", "vy", "vz",
            "time",
            "px", "py", "pz",
            "num_tracker_hits",
            "num_calo_hits",
            "vertex_primary",
            "parent_id",
        ]
        particle_columns = [c for c in desired_columns if c in particles_df.columns]
        
        # Create event particles dataframe
        event_particles = particles_df[particle_columns].copy()
        
        # Add event_id
        event_particles["event_id"] = event_id
        
        logging.debug(f"Event {event_id}: processed {len(event_particles)} particles")
        return event_particles
        
    except Exception as e:
        logging.error(f"Failed to process event {local_event_num} from {edm4hep_path}: {e}")
        return pd.DataFrame()
def build_particles_df_with_parents_and_vertex(
    batch: EDM4hepEventBatch,
    edm4hep_path: str,
    particles_root_df: pd.DataFrame | None,
    local_events: range | list,
    *,
    min_particle_energy: float | None = None,
    min_tracker_hits: int | None = None,
) -> pd.DataFrame:
    """
    Build a per-run particles dataframe using preloaded batch collections, with:
      - parent_id via parents_begin/parents_end + preloaded parents_df
      - vertex_primary merged from digi particles (particles.root) if provided
    """
    parts_all = batch.get_particles_df()
    parents_all = batch.get_parents_df()
    frames: List[pd.DataFrame] = []
    logger.debug(f"Building particles DataFrame with parents and vertex info for {len(local_events)} events")
    logger.debug(f"Particles DataFrame shape: {parts_all.shape if parts_all is not None else 'None'}, with columns {parts_all.columns if parts_all is not None else 'None'}, and unique events {parts_all.event_id.nunique() if parts_all is not None else 'None'}")
    logger.debug(f"Parents DataFrame shape: {parents_all.shape if parents_all is not None else 'None'}, with columns {parents_all.columns if parents_all is not None else 'None'}, and unique events {parents_all.event_id.nunique() if parents_all is not None else 'None'}")
    logger.debug(f"Particles root DataFrame shape: {particles_root_df.shape if particles_root_df is not None else 'None'}, with columns {particles_root_df.columns if particles_root_df is not None else 'None'}, and unique events {particles_root_df.event_id.nunique() if particles_root_df is not None else 'None'}")
    for local_event_num in local_events:
        ev_parts = parts_all[parts_all.event_id == local_event_num]
        ev_parents = parents_all[parents_all.event_id == local_event_num]
        ev_digi = None
        if particles_root_df is not None and not particles_root_df.empty:
            if "event_id" not in particles_root_df.columns and "event_nr" in particles_root_df.columns:
                particles_root_df = particles_root_df.rename(columns={"event_nr": "event_id"})
            ev_digi = particles_root_df[particles_root_df.get("event_id", -1) == local_event_num]
        ev_df = process_event_for_particles(
            event_id=local_event_num,
            local_event_num=local_event_num,
            edm4hep_path=str(edm4hep_path),
            digi_particles_df=ev_digi,
            preloaded_particles_df=ev_parts,
            preloaded_parents_df=ev_parents,
            min_particle_energy=min_particle_energy,
            min_tracker_hits=min_tracker_hits,
        )
        if not ev_df.empty:
            frames.append(ev_df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_particles_with_selection(
    df: pd.DataFrame,
    output_file: str,
    columns_keep: List[str] | None = None,
) -> None:
    """Write particles DataFrame to H5 with optional column selection."""
    if df.empty:
        return
    if columns_keep:
        cols = [c for c in columns_keep if c in df.columns]
        if 'event_id' not in cols and 'event_id' in df.columns:
            cols = cols + ['event_id']
        df = df[cols].copy()
    build_hdf5_particles(df, output_file)



def build_hdf5_particles(df: pd.DataFrame, output_file: str) -> None:
    """
    Write particle data to HDF5 under /events/event_#/particles.
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
                'particles',
                data=data_df.to_records(index=False),
                compression='gzip',
                compression_opts=6
            )


def process_run_for_particles(run_dir: Path, run_number: int, run_size: int) -> List[pd.DataFrame]:
    """
    Process all events in a run directory into a list of dataframes.
    """
    run_dir = Path(run_dir)
    edm4hep_path = run_dir / "edm4hep.root"
    particles_root_path = run_dir / "particles.root"

    if not edm4hep_path.exists():
        logging.warning(f"Missing EDM4hep file: {edm4hep_path}")
        return []

    # Load particles.root once per run (optional). If missing, continue without vertex info
    digi_particles_df: pd.DataFrame | None = None
    if particles_root_path.exists():
        try:
            digi_particles_df = load_root_file(str(particles_root_path), ignore_variable_columns=False)
        except Exception as e:
            logging.warning(f"Failed to load particles.root at {particles_root_path}: {e}")

    run_events: List[pd.DataFrame] = []
    # Batch load this full run for compatibility
    batch = EDM4hepEventBatch(str(edm4hep_path), events=range(run_size))
    parts_all = batch.get_particles_df()
    parents_all = batch.get_parents_df()

    for local_event_num in tqdm(range(run_size), desc="Processing events", leave=False):
        global_event_num = run_number * run_size + local_event_num
        ev_parts = parts_all[parts_all.event_id == local_event_num] if not parts_all.empty else pd.DataFrame()
        ev_parents = parents_all[parents_all.event_id == local_event_num] if not parents_all.empty else pd.DataFrame()
        ev_df = process_event_for_particles(
            global_event_num,
            local_event_num,
            str(edm4hep_path),
            digi_particles_df,
            preloaded_particles_df=ev_parts,
            preloaded_parents_df=ev_parents,
        )
        if not ev_df.empty:
            run_events.append(ev_df)

    return run_events


def process_chunk_for_particles(
    run_dirs: List[Path],
    start_event: int,
    end_event: int,
    start_run: int,
    start_local: int,
    end_run: int,
    end_local: int,
    output_dir: Path | str,
    dataset_name: str,
    run_size: int,
    force_overwrite: bool = False,
    *,
    min_particle_energy: float | None = None,
    min_tracker_hits: int | None = None,
    columns_keep: List[str] | None = None,
) -> None:
    """
    Process a chunk of runs and write one HDF5 file for the chunk.

    Args:
        run_dirs: List of run directories to process
        start_run: Index of the first run to process
        runs_per_chunk: Number of runs to process in each chunk
        output_dir: Directory to write the output HDF5 file (Path or str)
        dataset_name: Name of the dataset
        run_size: Number of events per run
        force_overwrite: Whether to overwrite existing output file

    Returns:
        None
    """
    # start_event/end_event precomputed; adjust end_run safe bound
    end_run = min(end_run, len(run_dirs) - 1)

    output_file = Path(output_dir) / f"{dataset_name}.truth.particles.events{start_event}-{end_event}.h5"
    if output_file.exists() and not force_overwrite:
        logging.info(f"Skipping events {start_event}-{end_event} - exists: {output_file}")
        return

    all_event_dfs: List[pd.DataFrame] = []
    total_rows = 0
    for abs_run in tqdm(range(start_run, end_run + 1), desc="Processing runs", leave=False):
        run_dir = run_dirs[abs_run]
        try:
            # Determine slice of local events for this run
            if abs_run == start_run and abs_run == end_run:
                local_events = range(start_local, end_local + 1)
            elif abs_run == start_run:
                local_events = range(start_local, run_size)
            elif abs_run == end_run:
                local_events = range(0, end_local + 1)
            else:
                local_events = range(run_size)

            # Optional per-run digi particles (particles.root)
            particles_root_path = run_dir / "particles.root"
            digi_particles_df: pd.DataFrame | None = None
            if particles_root_path.exists():
                try:
                    digi_particles_df = load_root_file(str(particles_root_path), ignore_variable_columns=False)
                    if "event_id" not in digi_particles_df.columns and "event_nr" in digi_particles_df.columns:
                        digi_particles_df = digi_particles_df.rename(columns={"event_nr": "event_id"})
                    digi_particles_df = digi_particles_df[digi_particles_df.get("event_id", -1).isin(local_events)].copy()
                except Exception as e:
                    logging.warning(f"Failed to load particles.root at {particles_root_path}: {e}")

            # Batch load only the requested events from edm4hep once
            edm4hep_path = run_dir / "edm4hep.root"
            if not edm4hep_path.exists():
                logging.warning(f"Missing EDM4hep file: {edm4hep_path}")
                continue
            batch = EDM4hepEventBatch(str(edm4hep_path), events=list(local_events))
            parts_all = batch.get_particles_df()
            parents_all = batch.get_parents_df()

            evs: List[pd.DataFrame] = []
            for local_event_num in tqdm(local_events, desc="Processing events", leave=False):
                global_event_num = abs_run * run_size + local_event_num
                ev_parts = parts_all[parts_all.event_id == local_event_num] if not parts_all.empty else pd.DataFrame()
                ev_parents = parents_all[parents_all.event_id == local_event_num] if 'parents_all' in locals() and not parents_all.empty else pd.DataFrame()
                ev_df = process_event_for_particles(
                    global_event_num,
                    local_event_num,
                    str(edm4hep_path),
                    digi_particles_df,
                    preloaded_particles_df=ev_parts,
                    preloaded_parents_df=ev_parents,
                    min_particle_energy=min_particle_energy,
                    min_tracker_hits=min_tracker_hits,
                )
                if not ev_df.empty:
                    evs.append(ev_df)
            all_event_dfs.extend(evs)
            total_rows += sum(len(df) for df in evs)
        except Exception as e:
            logging.error(f"Error processing run {abs_run}: {e}")

    if all_event_dfs:
        all_df = pd.concat(all_event_dfs, ignore_index=True)
        if columns_keep:
            # Preserve only requested columns that exist; keep event_id for grouping
            cols = [c for c in columns_keep if c in all_df.columns]
            if 'event_id' not in cols and 'event_id' in all_df.columns:
                cols = cols + ['event_id']
            all_df = all_df[cols].copy()
        logging.info(f"Writing {len(all_df)} particles across {all_df.event_id.nunique()} events -> {output_file}")
        build_hdf5_particles(all_df, str(output_file))
    else:
        logging.warning(f"No data to save for events {start_event}-{end_event}")


def convert_particles(
    base_dir: Path | str,
    output_base_dir: Path | str,
    dataset_name: str,
    chunk_size: int = 1000,
    run_size: int = 10,
    chunk_index: int | None = None,
    max_chunks: int | None = None,
    config_for_cap: dict | None = None,
    *,
    min_particle_energy: float | None = None,
    min_tracker_hits: int | None = None,
    columns_keep: List[str] | None = None,
) -> None:
    """
    Convert particle data to HDF5 files grouped by event.
    """
    base_dir = Path(base_dir)
    output_base_dir = Path(output_base_dir)

    run_dirs = get_run_paths(base_dir)

    output_dir = make_dir(output_base_dir, f"{dataset_name}/truth/particles")
    dataset_name = dataset_name.replace("/", ".")

    iterate_and_process_chunks(
        run_dirs=run_dirs,
        run_size=run_size,
        chunk_size=chunk_size,
        config=(
            {"max_chunks": max_chunks} if config_for_cap is None else {**config_for_cap, **({"max_chunks": max_chunks} if max_chunks is not None else {})}
        ),
        chunk_index=chunk_index,
        process_chunk_fn=lambda start_event, end_event, start_run, start_local, end_run, end_local: process_chunk_for_particles(
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
            min_particle_energy=min_particle_energy,
            min_tracker_hits=min_tracker_hits,
            columns_keep=columns_keep,
        ),
    )


def main():
    # Align CLI/config handling and file naming with convert_tracks.py
    parser = argparse.ArgumentParser(description="Convert EDM4HEP particle data to HDF5")
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
    # Use common.output_base_dir for outputs as well (unified root)
    output_base_dir = Path(config["common"]["output_base_dir"]) 

    chunk_size = config.get("chunk_size", 1000)
    run_size = config.get("run_size", 10)

    logging.info("\nStarting particle conversion with configuration:")
    logging.info(f"Campaign: {campaign}, Dataset: {dataset}, Version: {version}")
    logging.info(f"Input directory: {input_base_dir}")
    logging.info(f"Output root: {output_base_dir}")
    logging.info(f"Chunk size: {chunk_size}, Run size: {run_size}")

    convert_particles(
        input_base_dir,
        output_base_dir,
        f"{campaign}/{dataset}/{version}",
        chunk_size,
        run_size,
        args.chunk_index,
        config.get("max_chunks"),
        config,
        min_particle_energy=config.get("min_particle_energy"),
        min_tracker_hits=config.get("min_tracker_hits"),
        columns_keep=config.get("particles_columns_keep"),
    )


if __name__ == "__main__":
    main()