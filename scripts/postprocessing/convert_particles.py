#!/usr/bin/env python3
"""
Convert EDM4hep particle data to HDF5 or Parquet format.

This script processes particle information from EDM4hep files and creates
structured HDF5 or Parquet files with particle properties and hit count statistics.
"""

import argparse
import gc
import yaml
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

import h5py
from tqdm import tqdm
import logging
import sys
import time

# Use relative imports to avoid conflicts with other utils modules
from utils.path_utils import get_run_paths, make_dir
from utils.driver import iterate_and_process_chunks, local_events_for_run
from utils.track_utils import load_root_file
from utils.parquet_utils import build_parquet_from_flat_df
from utils.parquet_schemas import PARTICLES_PARQUET_TYPES

from pyedm4hep import EDM4hepEvent, EDM4hepEventBatch

logger = logging.getLogger(__name__)

ULP_TOLERANCE = 2   # float32 ulp per component for the D8 second pass


def _float32_ordinal(x: np.ndarray) -> np.ndarray:
    """Monotonic integer image of float32 values: adjacent floats differ by 1 (also across zero)."""
    i = np.ascontiguousarray(x, dtype=np.float32).view(np.int32).astype(np.int64)
    return np.where(i < 0, -(i & 0x7FFFFFFF), i)


def _add_ulp_matches(particles_df: pd.DataFrame, digi: pd.DataFrame, exact: pd.DataFrame,
                     merge_cols: list[str], tol: int = ULP_TOLERANCE) -> pd.DataFrame:
    """Exact join result plus the unmatched EDM4hep particles that match exactly one unmatched ACTS
    particle within ``tol`` float32 ulp in every component (and that ACTS particle matches only it).
    Ambiguous candidates are left out and logged. Without additions the exact result is returned as is."""
    left = particles_df[~particles_df["particle_id"].isin(exact["particle_id"])]
    if left.empty or digi.empty:
        return exact
    # ACTS rows consumed by the exact join (multiset by key)
    def keys(df):
        return [tuple(r) for r in df[merge_cols].to_numpy(np.float32).tolist()]
    used: dict = {}
    for k in keys(exact):
        used[k] = used.get(k, 0) + 1
    free_idx = []
    for i, k in enumerate(keys(digi)):
        if used.get(k, 0) > 0:
            used[k] -= 1
        else:
            free_idx.append(i)
    if not free_idx:
        return exact
    right = digi.iloc[free_idx]
    L = np.stack([_float32_ordinal(left[c].to_numpy()) for c in merge_cols], 1)
    R = np.stack([_float32_ordinal(right[c].to_numpy()) for c in merge_cols], 1)
    cand = {}
    for i in range(len(L)):
        ok = np.flatnonzero((np.abs(R - L[i]) <= tol).all(1))
        if len(ok):
            cand[i] = ok
    r_count: dict = {}
    for ok in cand.values():
        for j in ok:
            r_count[int(j)] = r_count.get(int(j), 0) + 1
    picks = []
    for i, ok in cand.items():
        if len(ok) == 1 and r_count[int(ok[0])] == 1:
            picks.append((i, int(ok[0])))
        else:
            logger.warning("particle %s: %d ACTS candidates within %d ulp; not matched (ambiguous)",
                           int(left["particle_id"].iloc[i]), len(ok), tol)
    if not picks:
        return exact
    extra_cols = [c for c in right.columns if c not in merge_cols]
    add = left.iloc[[i for i, _ in picks]].reset_index(drop=True)
    add_right = right.iloc[[j for _, j in picks]][extra_cols].reset_index(drop=True)
    add = pd.concat([add, add_right], axis=1)[exact.columns]
    logger.info("added %d particles matched within %d float32 ulp: %s", len(add), tol,
                add["particle_id"].tolist())
    out = pd.concat([exact, add], ignore_index=True)
    return out.sort_values("particle_id", kind="stable").reset_index(drop=True)


def process_event_for_particles(
    event_id: int,
    local_event_num: int,
    edm4hep_path: str,
    digi_particles_df: pd.DataFrame | None = None,
    preloaded_particles_df: pd.DataFrame | None = None,
    preloaded_parents_df: pd.DataFrame | None = None,
    min_particle_energy: float | None = None,
    min_tracker_hits: int | None = None,
    min_calo_hits: int | None = None,
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
        t0 = time.time()
        # Use preloaded per-event slice if provided; otherwise read from file
        if preloaded_particles_df is not None:
            particles_df = preloaded_particles_df.copy()
        else:
            _t_ev_load = time.time()
            event = EDM4hepEvent(edm4hep_path, event_index=local_event_num)
            particles_df = event.get_particles_df()
            logger.debug(f"Particles event load local={local_event_num} time={time.time() - _t_ev_load:.3f}s")
        
        # Reset index and add particle_id as an unsigned identifier
        particles_df.reset_index(drop=True, inplace=True)
        particles_df["particle_id"] = particles_df.index.astype("uint32")
        
        # Normalize column names if needed
        if "pdg_id" not in particles_df.columns and "PDG" in particles_df.columns:
            particles_df["pdg_id"] = particles_df["PDG"]
        
        # If available, merge in vertex_primary from particles.root by matching on phase-space columns
        if digi_particles_df is not None and not digi_particles_df.empty:
            # Normalize event id column name
            if "event_id" not in digi_particles_df.columns and "event_nr" in digi_particles_df.columns:
                digi_particles_df = digi_particles_df.rename(columns={"event_nr": "event_id"})

            # Select this local event's digi particles (boolean indexing creates view)
            local_digi = digi_particles_df[digi_particles_df.get("event_id", -1) == local_event_num]

            # Ensure required merge columns exist
            merge_cols = ["vx", "vy", "vz", "px", "py", "pz"]
            if all(col in particles_df.columns for col in merge_cols) and all(col in local_digi.columns for col in merge_cols):
                # Align dtypes for robust merging
                for col in merge_cols:
                    particles_df[col] = particles_df[col].astype("float32")
                    local_digi[col] = local_digi[col].astype("float32")

                extra_cols = ["vertex_primary", "perigee_d0", "perigee_z0"]
                right_cols = merge_cols + [c for c in extra_cols if c in local_digi.columns]
                if right_cols:
                    _t_merge = time.time()
                    exact = pd.merge(
                        particles_df,
                        local_digi[right_cols],
                        on=merge_cols,
                        how="inner",
                    )
                    # Drop duplicates
                    exact = exact.drop_duplicates(subset="particle_id")
                    # D8 (2026-11-28): a component on a float32 rounding tie rounds differently on the two
                    # sides, and the exact join above dropped the particle (about 0.2% of PU200 events). The
                    # leftovers are matched within 2 float32 ulp per component, one-to-one; exact rows are
                    # unchanged, added rows go into particle_id order.
                    particles_df = _add_ulp_matches(particles_df, local_digi[right_cols], exact, merge_cols)

                    # Cast perigee_d0 and perigee_z0 to float32
                    particles_df["perigee_d0"] = particles_df["perigee_d0"].astype("float32")
                    particles_df["perigee_z0"] = particles_df["perigee_z0"].astype("float32")

                    logger.debug(f"Merged vertex_primary for local={local_event_num} time={time.time() - _t_merge:.3f}s")

        # Assign parent_id (first parent) when link info is available
        if preloaded_parents_df is not None and not preloaded_parents_df.empty:
            # Ensure we have the link-range columns
            if {"parents_begin", "parents_end"}.issubset(particles_df.columns):
                parent_id_series = pd.Series([-1] * len(particles_df), dtype="int64", index=particles_df.index)
                # Rows with at least one parent
                has_parent = particles_df["parents_end"].values > particles_df["parents_begin"].values
                if has_parent.any():
                    begin_idx = particles_df.loc[has_parent, "parents_begin"].astype(int).values
                    # parents df is per-event; iloc indices refer to per-event flattened list
                    parent_ids = preloaded_parents_df.iloc[begin_idx]["particle_id"].values
                    parent_id_series.loc[has_parent] = parent_ids
                particles_df = particles_df.copy()
                particles_df["parent_id"] = parent_id_series

        # Derive primary flag from created_in_simulation when available
        if "created_in_simulation" in particles_df.columns:
            # Generator primaries are those not created in simulation
            particles_df["primary"] = ~particles_df["created_in_simulation"].astype(bool)

        # Start particle cut mask as all True
        particle_cut_mask = np.ones(len(particles_df), dtype=bool)

        # Apply configurable minimum energy filter if requested
        if min_particle_energy is not None and "energy" in particles_df.columns:
            try:
                particle_cut_mask = particle_cut_mask | (particles_df["energy"] >= float(min_particle_energy))
            except Exception:
                pass

        # Apply configurable minimum tracker hits filter if available
        if min_tracker_hits is not None and "num_tracker_hits" in particles_df.columns:
            try:
                particle_cut_mask = particle_cut_mask | (particles_df["num_tracker_hits"] >= int(min_tracker_hits))
                logger.debug(f"Event {event_id}: {particle_cut_mask.sum()} particles after min_tracker_hits filter ")
            except Exception:
                pass

        # Apply configurable minimum calo hits filter if available
        if min_calo_hits is not None and "num_calo_hits" in particles_df.columns:
            try:
                particle_cut_mask = particle_cut_mask | (particles_df["num_calo_hits"] >= int(min_calo_hits))
                logger.debug(f"Event {event_id}: {particle_cut_mask.sum()} particles after min_calo_hits filter ")
            except Exception:
                pass

        # Also ensure all generator particles are included
        if "created_in_simulation" in particles_df.columns:
            particle_cut_mask = particle_cut_mask | (particles_df["created_in_simulation"] == False)
        
        # Apply particle cut mask
        logger.debug(f"Event {event_id}: {particle_cut_mask.sum()} particles after particle cut mask")
        particles_df = particles_df[particle_cut_mask]

        # Select relevant particle columns (include vertex_primary/parent_id if present)
        desired_columns = [
            "particle_id",
            "pdg_id",
            "mass",
            "energy",
            "charge",
            "vx",
            "vy",
            "vz",
            "time",
            "px",
            "py",
            "pz",
            "num_tracker_hits",
            "num_calo_hits",
            "primary",
            "vertex_primary",
            "parent_id",
            "perigee_d0",
            "perigee_z0",
        ]
        particle_columns = [c for c in desired_columns if c in particles_df.columns]
        
        # Create event particles dataframe
        event_particles = particles_df[particle_columns].copy()
        
        # Add event_id
        event_particles["event_id"] = event_id
        
        logging.debug(f"Event {event_id}: processed {len(event_particles)} particles in {time.time() - t0:.3f}s")
        return event_particles
        
    except Exception as e:
        logging.error(f"Failed to process event {local_event_num} from {edm4hep_path}: {e}")
        return pd.DataFrame()

        
def build_particles_df_with_parents_and_vertex(
    batch: EDM4hepEventBatch,
    edm4hep_path: str,
    particles_root_df: pd.DataFrame | None,
    local_events: tuple[int, int],
    *,
    min_particle_energy: float | None = None,
    min_tracker_hits: int | None = None,
    min_calo_hits: int | None = None,
) -> pd.DataFrame:
    """
    Build a per-run particles dataframe using preloaded batch collections, with:
      - parent_id via parents_begin/parents_end + preloaded parents_df
      - vertex_primary merged from digi particles (particles.root) if provided
    """
    batch._ensure_loaded("tracker_hits") #Since we will need to count the hits per particle
    parts_all = batch.get_particles_df()
    parents_all = batch.get_parents_df()
    
    frames: List[pd.DataFrame] = []
    t_start = time.time()
    logger.debug(f"Building particles DataFrame with parents and vertex info for {len(local_events)} events")
    logger.debug(f"Particles DataFrame shape: {parts_all.shape if parts_all is not None else 'None'}, with columns {parts_all.columns if parts_all is not None else 'None'}, and unique events {parts_all.event_id.nunique() if parts_all is not None else 'None'}")
    logger.debug(f"Parents DataFrame shape: {parents_all.shape if parents_all is not None else 'None'}, with columns {parents_all.columns if parents_all is not None else 'None'}, and unique events {parents_all.event_id.nunique() if parents_all is not None else 'None'}")
    logger.debug(f"Particles root DataFrame shape: {particles_root_df.shape if particles_root_df is not None else 'None'}, with columns {particles_root_df.columns if particles_root_df is not None else 'None'}, and unique events {particles_root_df.event_id.nunique() if particles_root_df is not None else 'None'}")
    for local_event_num in range(local_events[0], local_events[1]):
        ev_parts = parts_all[parts_all.event_id == local_event_num]
        ev_parents = parents_all[parents_all.event_id == local_event_num]
        ev_digi = None
        if particles_root_df is not None and not particles_root_df.empty:
            if "event_id" not in particles_root_df.columns and "event_nr" in particles_root_df.columns:
                particles_root_df = particles_root_df.rename(columns={"event_nr": "event_id"})
            ev_digi = particles_root_df[particles_root_df.get("event_id", -1) == local_event_num]
        print("Particles event: ", local_event_num, ev_parts.shape, ev_digi.shape)
        ev_df = process_event_for_particles(
            event_id=local_event_num,
            local_event_num=local_event_num,
            edm4hep_path=str(edm4hep_path),
            digi_particles_df=ev_digi,
            preloaded_particles_df=ev_parts,
            preloaded_parents_df=ev_parents,
            min_particle_energy=min_particle_energy,
            min_tracker_hits=min_tracker_hits,
            min_calo_hits=min_calo_hits,
        )
        if not ev_df.empty:
            frames.append(ev_df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    logger.debug(f"Built run-level particles df rows={len(out)} time={time.time() - t_start:.3f}s")
    return out


def build_parquet_particles(df: pd.DataFrame, output_file: str, row_group_size: int | None = None) -> None:
    """
    Write particle data to Parquet format.
    
    Args:
        df: Flat DataFrame with event_id and per-particle columns
        output_file: Path to output Parquet file
        row_group_size: Number of rows per Parquet row group (None = PyArrow default)
    """
    if df.empty:
        logger.warning(f"Skipping empty DataFrame for Parquet particles: {output_file}")
        return
    
    # Use shared utility to group by event and write with canonical schema
    build_parquet_from_flat_df(
        df,
        output_file,
        compression='snappy',
        schema_overrides=PARTICLES_PARQUET_TYPES,
        row_group_size=row_group_size,
    )


def write_particles_with_selection(
    df: pd.DataFrame,
    output_file: str,
    columns_keep: List[str] | None = None,
    output_format: str = 'hdf5',
    row_group_size: int | None = None,
) -> None:
    """
    Write particles DataFrame to HDF5 or Parquet with optional column selection.
    
    Args:
        df: DataFrame with particle data
        output_file: Path to output file
        columns_keep: Optional list of columns to keep
        output_format: Output format - 'hdf5' (default) or 'parquet'
        row_group_size: Number of rows per Parquet row group (None = PyArrow default)
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
        build_parquet_particles(df, output_file, row_group_size=row_group_size)
    else:  # default to hdf5
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
    output_format: str = 'hdf5',
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

    # Determine file extension based on output format
    file_ext = '.parquet' if output_format == 'parquet' else '.h5'
    output_file = Path(output_dir) / f"{dataset_name}.truth.particles.events{start_event}-{end_event}{file_ext}"
    chunk_start = time.time()
    if output_file.exists() and not force_overwrite:
        logging.info(f"Skipping events {start_event}-{end_event} - exists: {output_file}")
        return

    all_event_dfs: List[pd.DataFrame] = []
    total_rows = 0
    for abs_run in tqdm(range(start_run, end_run + 1), desc="Processing runs", leave=False):
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

            # Optional per-run digi particles (particles.root)
            particles_root_path = run_dir / "particles.root"
            digi_particles_df: pd.DataFrame | None = None
            if particles_root_path.exists():
                try:
                    _t_digi = time.time()
                    digi_particles_df = load_root_file(str(particles_root_path), ignore_variable_columns=False)
                    logger.debug(f"Loaded particles.root for run {abs_run} in {time.time() - _t_digi:.3f}s")
                    if "event_id" not in digi_particles_df.columns and "event_nr" in digi_particles_df.columns:
                        digi_particles_df = digi_particles_df.rename(columns={"event_nr": "event_id"})
                    if local_events_list:
                        digi_particles_df = digi_particles_df[digi_particles_df.get("event_id", -1).isin(local_events_list)].copy()
                except Exception as e:
                    logging.warning(f"Failed to load particles.root at {particles_root_path}: {e}")

            # Batch load only the requested events from edm4hep once
            edm4hep_path = run_dir / "edm4hep.root"
            if not edm4hep_path.exists():
                logging.warning(f"Missing EDM4hep file: {edm4hep_path}")
                continue
            local_events_str = (
                f"{local_start}-{local_stop-1} (n={local_count})" if local_count > 0 else "<empty>"
            )
            logging.info(
                f"Run {abs_run}: dir={run_dir} edm4hep={edm4hep_path} particles.root={particles_root_path if particles_root_path.exists() else '<missing>'} local_events={local_events_str}"
            )
            _t_batch = time.time()
            events_selector = (local_start, local_stop) if local_count > 0 else None
            batch = EDM4hepEventBatch(str(edm4hep_path), events=events_selector)
            parts_all = batch.get_particles_df()
            parents_all = batch.get_parents_df()
            logger.debug(f"Loaded particles+parents batch for run {abs_run} in {time.time() - _t_batch:.3f}s")

            evs: List[pd.DataFrame] = []
            rows_run = 0
            ev_count = 0
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
                    rows_run += len(ev_df)
                    ev_count += 1
            all_event_dfs.extend(evs)
            total_rows += sum(len(df) for df in evs)
            logging.info(
                f"Run {abs_run}: particles rows={rows_run} events={ev_count}"
            )
            
            # Delete batch object and force garbage collection to free memory
            del batch
            gc.collect()
            
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
        logging.info(f"Writing {len(all_df)} particles across {all_df.event_id.nunique()} events -> {output_file} (chunk_time={time.time() - chunk_start:.3f}s)")
        write_particles_with_selection(all_df, str(output_file), columns_keep=None, output_format=output_format)
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
    output_format: str = 'hdf5',
) -> None:
    """
    Convert particle data to HDF5 or Parquet files grouped by event.
    
    Args:
        output_format: Output format - 'hdf5' (default) or 'parquet'
    """
    base_dir = Path(base_dir)
    output_base_dir = Path(output_base_dir)

    run_dirs = get_run_paths(base_dir)

    # Use format-specific subdirectory
    format_subdir = output_format if output_format in ['hdf5', 'parquet'] else 'hdf5'
    output_dir = make_dir(output_base_dir, f"{dataset_name}/{format_subdir}/truth/particles")
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
            output_format=output_format,
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

    # Extract output format from config (default to hdf5 for backward compatibility)
    output_format = config.get("output_format", "hdf5")
    
    logging.info("\nStarting particle conversion with configuration:")
    logging.info(f"Campaign: {campaign}, Dataset: {dataset}, Version: {version}")
    logging.info(f"Input directory: {input_base_dir}")
    logging.info(f"Output root: {output_base_dir}")
    logging.info(f"Output format: {output_format}")
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
        output_format=output_format,
    )


if __name__ == "__main__":
    main()