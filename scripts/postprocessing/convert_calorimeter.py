#!/usr/bin/env python3
"""
Convert EDM4hep calorimeter data to HDF5 or Parquet format.

This script processes calorimeter hits and contributions from EDM4hep files,
applying timing and energy thresholds, and creates structured output files
with nested contributions per cell.
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
import time

# Use relative imports to avoid conflicts with other utils modules
from utils.path_utils import get_run_paths, make_dir
from utils.driver import iterate_and_process_chunks, local_events_for_run
from utils.parquet_utils import build_parquet_from_flat_df

sys.path.append("/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software/OtherLibraries/pyedm4hep")
from pyedm4hep import EDM4hepEvent, EDM4hepEventBatch

logger = logging.getLogger(__name__)


def process_event_for_calohits(
    event_id: int,
    local_event_num: int,
    preloaded_calo_hits: pd.DataFrame | None = None,
    preloaded_calo_contributions: pd.DataFrame | None = None,
    ecal_energy_threshold: float = 5.0e-5,  # GeV
    hcal_energy_threshold: float = 2.5e-4,  # GeV
    ecal_time_min: float = -1.0,  # ns
    ecal_time_max: float = 10.0,  # ns
    hcal_time_min: float = -1.0,  # ns
    hcal_time_max: float = 10.0,  # ns
) -> pd.DataFrame:
    """
    Process calorimeter data for a single event with nested contributions.
    
    This creates a cell-level DataFrame where each row is a calorimeter cell,
    with contributions stored as nested lists (particle_id, energy, time).
    
    Args:
        event_id: Global event number
        local_event_num: Local event number within the run
        preloaded_calo_hits: Preloaded calo hits DataFrame
        preloaded_calo_contributions: Preloaded calo contributions DataFrame
        ecal_energy_threshold: ECal energy threshold in GeV (cell level)
        hcal_energy_threshold: HCal energy threshold in GeV (cell level)
        ecal_time_min: ECal minimum time in ns
        ecal_time_max: ECal maximum time in ns
        hcal_time_min: HCal minimum time in ns
        hcal_time_max: HCal maximum time in ns
        
    Returns:
        DataFrame with columns: event_id, cellID, detector, x, y, z, total_energy, 
                                contrib_particle_ids, contrib_energies, contrib_times
    """
    try:
        t0 = time.time()
        
        if preloaded_calo_contributions is None or preloaded_calo_contributions.empty:
            logger.warning(f"Event {event_id}: no calorimeter contributions")
            return pd.DataFrame()
        
        if preloaded_calo_hits is None or preloaded_calo_hits.empty:
            logger.warning(f"Event {event_id}: no calorimeter hits")
            return pd.DataFrame()
        
        # Work with contributions copy
        contribs = preloaded_calo_contributions.copy()
        
        # Step 1: Apply timing filters to contributions
        if 'time' in contribs.columns:
            # Calculate time-of-flight correction
            contribs['r'] = np.sqrt(contribs['x']**2 + contribs['y']**2 + contribs['z']**2)
            contribs['dt'] = contribs['r'] / 300.0 - 0.1  # time-of-flight in ns
            contribs['corrected_time'] = contribs['time'] - contribs['dt']
            
            # Create detector masks
            ecal_mask = contribs['detector'].str.contains('ECal', na=False)
            hcal_mask = contribs['detector'].str.contains('HCal', na=False)
            
            # Apply timing filters
            timing_mask = (
                (ecal_mask & 
                 (contribs['corrected_time'] >= ecal_time_min) & 
                 (contribs['corrected_time'] <= ecal_time_max)) |
                (hcal_mask & 
                 (contribs['corrected_time'] >= hcal_time_min) & 
                 (contribs['corrected_time'] <= hcal_time_max))
            )
            contribs = contribs[timing_mask].copy()
            logger.debug(f"Event {event_id}: {len(contribs)} contributions after timing filter")
        
        if contribs.empty:
            return pd.DataFrame()
        
        # Step 2: Aggregate contributions by (event_id, cellID, particle_id)
        # Use energy-weighted time for aggregated time per particle contribution
        contribs['energy_time'] = contribs['energy'] * contribs.get('time', 0.0)
        
        contrib_per_particle = (
            contribs.groupby(['event_id', 'cellID', 'particle_id'], sort=False)
            .agg(
                energy=('energy', 'sum'),
                energy_time=('energy_time', 'sum'),
                detector=('detector', 'first'),
            )
            .reset_index()
        )
        
        # Calculate energy-weighted time per particle contribution
        contrib_per_particle['time'] = contrib_per_particle['energy_time'] / contrib_per_particle['energy']
        contrib_per_particle = contrib_per_particle.drop(columns=['energy_time'])
        
        # Step 3a: First group by cell to get total energy (for threshold)
        cell_energy = (
            contrib_per_particle.groupby(['event_id', 'cellID'], sort=False)
            .agg(
                detector=('detector', 'first'),
                total_energy=('energy', 'sum'),
            )
            .reset_index()
        )
        
        # Step 3b: Apply cell-level energy thresholds BEFORE creating nested lists (much faster!)
        ecal_mask = cell_energy['detector'].str.contains('ECal', na=False)
        hcal_mask = cell_energy['detector'].str.contains('HCal', na=False)
        
        energy_mask = (
            (ecal_mask & (cell_energy['total_energy'] >= ecal_energy_threshold)) |
            (hcal_mask & (cell_energy['total_energy'] >= hcal_energy_threshold))
        )
        
        cells_passing_threshold = cell_energy[energy_mask][['event_id', 'cellID']].copy()
        logger.debug(f"Event {event_id}: {len(cells_passing_threshold)} cells pass energy threshold")
        
        # Step 3c: Filter contributions to only cells passing threshold
        contrib_filtered = contrib_per_particle.merge(
            cells_passing_threshold,
            on=['event_id', 'cellID'],
            how='inner'
        )
        
        if contrib_filtered.empty:
            return pd.DataFrame()
        
        # Step 3d: Now group to create nested lists (much smaller dataset!)
        cell_level = (
            contrib_filtered.groupby(['event_id', 'cellID'], sort=False)
            .agg(
                detector=('detector', 'first'),
                total_energy=('energy', 'sum'),
                contrib_particle_ids=('particle_id', list),
                contrib_energies=('energy', list),
                contrib_times=('time', list),
            )
            .reset_index()
        )
        
        # Step 4: Merge with calo_hits to get cell positions (x, y, z)
        calo_cells = cell_level.merge(
            preloaded_calo_hits[['event_id', 'cellID', 'x', 'y', 'z']],
            on=['event_id', 'cellID'],
            how='inner'
        )
        
        logger.debug(f"Event {event_id}: {len(calo_cells)} final cells with positions")
        
        # Add global event_id (already present from groupby)
        calo_cells['event_id'] = event_id
        
        logger.debug(f"Event {event_id}: processed {len(calo_cells)} calorimeter cells in {time.time() - t0:.3f}s")
        return calo_cells
        
    except Exception as e:
        logger.error(f"Failed to process calorimeter data for event {event_id}: {e}")
        return pd.DataFrame()


def build_calohits_df_batch(
    batch: EDM4hepEventBatch,
    local_events: tuple[int, int],
    ecal_energy_threshold: float = 5.0e-5,
    hcal_energy_threshold: float = 2.5e-4,
    ecal_time_min: float = -1.0,
    ecal_time_max: float = 10.0,
    hcal_time_min: float = -1.0,
    hcal_time_max: float = 10.0,
) -> pd.DataFrame:
    """
    Build calorimeter hits DataFrame for a batch of events.
    
    Args:
        batch: EDM4hepEventBatch with loaded calorimeter data
        local_events: Tuple of (start, stop) local event indices
        ecal_energy_threshold: ECal energy threshold in GeV
        hcal_energy_threshold: HCal energy threshold in GeV
        ecal_time_min: ECal minimum time in ns
        ecal_time_max: ECal maximum time in ns
        hcal_time_min: HCal minimum time in ns
        hcal_time_max: HCal maximum time in ns
        
    Returns:
        DataFrame with calorimeter cells and nested contributions
    """
    t_start = time.time()
    
    # Load batch data once
    calo_contributions_all = batch.get_calo_contributions_df()
    calo_hits_all = batch.get_calo_hits_df()
    
    if calo_contributions_all is None or calo_contributions_all.empty:
        logger.warning("No calorimeter contributions found in batch")
        return pd.DataFrame()
    
    if calo_hits_all is None or calo_hits_all.empty:
        logger.warning("No calorimeter hits found in batch")
        return pd.DataFrame()
    
    frames: List[pd.DataFrame] = []
    
    for local_event_num in range(local_events[0], local_events[1]):
        # Slice per-event data
        ev_contribs = calo_contributions_all[calo_contributions_all.event_id == local_event_num]
        ev_hits = calo_hits_all[calo_hits_all.event_id == local_event_num]
        
        if ev_contribs.empty or ev_hits.empty:
            continue
        
        ev_df = process_event_for_calohits(
            event_id=local_event_num,
            local_event_num=local_event_num,
            preloaded_calo_hits=ev_hits,
            preloaded_calo_contributions=ev_contribs,
            ecal_energy_threshold=ecal_energy_threshold,
            hcal_energy_threshold=hcal_energy_threshold,
            ecal_time_min=ecal_time_min,
            ecal_time_max=ecal_time_max,
            hcal_time_min=hcal_time_min,
            hcal_time_max=hcal_time_max,
        )
        
        if not ev_df.empty:
            frames.append(ev_df)
    
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    logger.debug(f"Built calorimeter hits df rows={len(out)} time={time.time() - t_start:.3f}s")
    return out


def build_parquet_calohits(df: pd.DataFrame, output_file: str) -> None:
    """
    Write calorimeter hits to Parquet format with nested contributions.
    
    The nested lists (contrib_particle_ids, contrib_energies, contrib_times) 
    will be preserved in Parquet format.
    
    Args:
        df: Flat DataFrame with event_id, cell-level columns, and list columns
        output_file: Path to output Parquet file
    """
    if df.empty:
        logger.warning(f"Skipping empty DataFrame for Parquet calorimeter: {output_file}")
        return
    
    df = df.copy()
    
    # Standardize column name to cell_id (with underscore)
    # This matches the naming convention used for other ID columns
    if 'cellID' in df.columns:
        df.rename(columns={'cellID': 'cell_id'}, inplace=True)
    
    # Convert cell_id to string to handle large bitfield-encoded values
    # These can exceed int64 range (2^63-1) due to bitfield encoding, and PyArrow
    # has issues with uint64 in lists. Storing as string is the most reliable approach.
    if 'cell_id' in df.columns:
        df['cell_id'] = df['cell_id'].astype(str)
    
    # Use shared utility to group by event and write
    # The list columns will automatically become list[list[...]] after groupby
    build_parquet_from_flat_df(df, output_file, compression='snappy')
    logger.info(f"Wrote calorimeter parquet file: {output_file}")


def write_calohits_with_selection(
    df: pd.DataFrame,
    output_file: str,
    columns_keep: List[str] | None = None,
    output_format: str = 'hdf5',
) -> None:
    """
    Write calorimeter hits DataFrame to HDF5 or Parquet with optional column selection.
    
    Args:
        df: DataFrame with calorimeter cell data and nested contributions
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
        # Ensure we keep the nested contribution columns
        for contrib_col in ['contrib_particle_ids', 'contrib_energies', 'contrib_times']:
            if contrib_col in df.columns and contrib_col not in cols:
                cols.append(contrib_col)
        df = df[cols].copy()
    
    # Route to appropriate writer based on format
    if output_format == 'parquet':
        build_parquet_calohits(df, output_file)
    else:  # default to hdf5
        build_hdf5_calohits(df, output_file)


def build_hdf5_calohits(df: pd.DataFrame, output_file: str) -> None:
    """
    Write calorimeter data to HDF5 under /events/event_#/calo_hits.
    
    Nested contributions are stored as separate datasets per event.
    """
    with h5py.File(output_file, 'a') as f:
        events_group = f.create_group('events') if 'events' not in f else f['events']

        for event_id, event_df in df.groupby('event_id'):
            event_group_name = f'event_{event_id}'
            if event_group_name in events_group:
                # Remove existing group to avoid conflicts
                del events_group[event_group_name]
            event_group = events_group.create_group(event_group_name)

            # Store scalar cell-level data
            scalar_cols = ['cellID', 'detector', 'x', 'y', 'z', 'energy']
            scalar_data = event_df[[c for c in scalar_cols if c in event_df.columns]].copy()
            
            # Convert detector strings to integers for HDF5 compatibility
            if 'detector' in scalar_data.columns:
                detector_mapping = {det: i for i, det in enumerate(scalar_data['detector'].unique())}
                scalar_data['detector'] = scalar_data['detector'].map(detector_mapping)
            
            event_group.create_dataset(
                'calo_hits',
                data=scalar_data.to_records(index=False),
                compression='gzip',
                compression_opts=6
            )
            
            # Store nested contributions as variable-length arrays
            if 'contrib_particle_ids' in event_df.columns:
                contrib_particle_ids = event_df['contrib_particle_ids'].values
                event_group.create_dataset(
                    'contrib_particle_ids',
                    data=contrib_particle_ids,
                    dtype=h5py.vlen_dtype(np.dtype('int64')),
                    compression="gzip",
                    compression_opts=6
                )
            
            if 'contrib_energies' in event_df.columns:
                contrib_energies = event_df['contrib_energies'].values
                event_group.create_dataset(
                    'contrib_energies',
                    data=contrib_energies,
                    dtype=h5py.vlen_dtype(np.dtype('float32')),
                    compression="gzip",
                    compression_opts=6
                )
            
            if 'contrib_times' in event_df.columns:
                contrib_times = event_df['contrib_times'].values
                event_group.create_dataset(
                    'contrib_times',
                    data=contrib_times,
                    dtype=h5py.vlen_dtype(np.dtype('float32')),
                    compression="gzip",
                    compression_opts=6
                )


def process_chunk_for_calohits(
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
    ecal_energy_threshold: float = 5.0e-5,
    hcal_energy_threshold: float = 2.5e-4,
    ecal_time_min: float = -1.0,
    ecal_time_max: float = 10.0,
    hcal_time_min: float = -1.0,
    hcal_time_max: float = 10.0,
) -> None:
    """
    Process a chunk of runs and write one output file for the chunk.
    
    Args:
        run_dirs: List of run directories to process
        start_event: First global event number in chunk
        end_event: Last global event number in chunk
        start_run: Index of first run
        start_local: First local event in start_run
        end_run: Index of last run
        end_local: Last local event in end_run (exclusive)
        output_dir: Directory to write output file
        dataset_name: Name of the dataset
        run_size: Number of events per run
        force_overwrite: Whether to overwrite existing output file
        columns_keep: Optional list of columns to keep
        output_format: Output format - 'hdf5' or 'parquet'
        ecal_energy_threshold: ECal energy threshold in GeV
        hcal_energy_threshold: HCal energy threshold in GeV
        ecal_time_min: ECal minimum time in ns
        ecal_time_max: ECal maximum time in ns
        hcal_time_min: HCal minimum time in ns
        hcal_time_max: HCal maximum time in ns
    """
    end_run = min(end_run, len(run_dirs) - 1)
    
    # Determine file extension based on output format
    file_ext = '.parquet' if output_format == 'parquet' else '.h5'
    output_file = Path(output_dir) / f"{dataset_name}.reco.calo_hits.events{start_event}-{end_event}{file_ext}"
    
    chunk_start = time.time()
    if output_file.exists() and not force_overwrite:
        logger.info(f"Skipping events {start_event}-{end_event} - exists: {output_file}")
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
            local_count = local_stop - local_start
            local_events = (local_start, local_stop)
            
            edm4hep_path = run_dir / "edm4hep.root"
            if not edm4hep_path.exists():
                logger.warning(f"Missing EDM4hep file: {edm4hep_path}")
                continue
            
            local_events_str = (
                f"{local_start}-{local_stop-1} (n={local_count})" if local_count > 0 else "<empty>"
            )
            logger.info(
                f"Run {abs_run}: dir={run_dir} edm4hep={edm4hep_path} local_events={local_events_str}"
            )
            
            # Batch load calorimeter data once per run
            _t_batch = time.time()
            batch = EDM4hepEventBatch(str(edm4hep_path), events=local_events)
            logger.debug(f"Loaded EDM4hep batch for run {abs_run} in {time.time() - _t_batch:.3f}s")
            
            # Process events in this run
            run_df = build_calohits_df_batch(
                batch,
                local_events,
                ecal_energy_threshold=ecal_energy_threshold,
                hcal_energy_threshold=hcal_energy_threshold,
                ecal_time_min=ecal_time_min,
                ecal_time_max=ecal_time_max,
                hcal_time_min=hcal_time_min,
                hcal_time_max=hcal_time_max,
            )
            
            if not run_df.empty:
                # Update event_ids to global numbering
                run_df['event_id'] = run_df['event_id'] + abs_run * run_size
                all_event_dfs.append(run_df)
                total_rows += len(run_df)
                logger.info(
                    f"Run {abs_run}: calo_hits rows={len(run_df)} events={run_df.event_id.nunique()}"
                )
                
        except Exception as e:
            logger.error(f"Error processing run {abs_run}: {e}")

    if all_event_dfs:
        all_df = pd.concat(all_event_dfs, ignore_index=True)
        logger.info(
            f"Writing {len(all_df)} calorimeter cells across {all_df.event_id.nunique()} events -> {output_file} "
            f"(chunk_time={time.time() - chunk_start:.3f}s)"
        )
        write_calohits_with_selection(all_df, str(output_file), columns_keep=columns_keep, output_format=output_format)
    else:
        logger.warning(f"No data to save for events {start_event}-{end_event}")


def convert_calorimeter(
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
    ecal_energy_threshold: float = 5.0e-5,
    hcal_energy_threshold: float = 2.5e-4,
    ecal_time_min: float = -1.0,
    ecal_time_max: float = 10.0,
    hcal_time_min: float = -1.0,
    hcal_time_max: float = 10.0,
) -> None:
    """
    Convert calorimeter data to HDF5 or Parquet files grouped by event.
    
    Args:
        base_dir: Input directory containing run folders
        output_base_dir: Base output directory
        dataset_name: Dataset name (campaign/dataset/version)
        chunk_size: Number of events per output file
        run_size: Number of events per run
        chunk_index: Optional chunk index for parallel processing
        max_chunks: Optional maximum number of chunks to process
        config_for_cap: Optional config dict for capping
        columns_keep: Optional list of columns to keep
        output_format: Output format - 'hdf5' (default) or 'parquet'
        ecal_energy_threshold: ECal energy threshold in GeV
        hcal_energy_threshold: HCal energy threshold in GeV
        ecal_time_min: ECal minimum time in ns
        ecal_time_max: ECal maximum time in ns
        hcal_time_min: HCal minimum time in ns
        hcal_time_max: HCal maximum time in ns
    """
    base_dir = Path(base_dir)
    output_base_dir = Path(output_base_dir)

    run_dirs = get_run_paths(base_dir)

    # Use format-specific subdirectory
    format_subdir = output_format if output_format in ['hdf5', 'parquet'] else 'hdf5'
    output_dir = make_dir(output_base_dir, f"{dataset_name}/{format_subdir}/reco/calo_hits")
    dataset_name = dataset_name.replace("/", ".")
    
    iterate_and_process_chunks(
        run_dirs=run_dirs,
        run_size=run_size,
        chunk_size=chunk_size,
        config=(
            {"max_chunks": max_chunks} if config_for_cap is None 
            else {**config_for_cap, **({"max_chunks": max_chunks} if max_chunks is not None else {})}
        ),
        chunk_index=chunk_index,
        process_chunk_fn=lambda start_event, end_event, start_run, start_local, end_run, end_local: process_chunk_for_calohits(
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
            ecal_energy_threshold=ecal_energy_threshold,
            hcal_energy_threshold=hcal_energy_threshold,
            ecal_time_min=ecal_time_min,
            ecal_time_max=ecal_time_max,
            hcal_time_min=hcal_time_min,
            hcal_time_max=hcal_time_max,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="Convert EDM4HEP calorimeter data to HDF5 or Parquet")
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
    output_base_dir = Path(config["common"]["output_base_dir"]) 

    chunk_size = config.get("chunk_size", 1000)
    run_size = config.get("run_size", 10)

    # Extract output format from config (default to hdf5 for backward compatibility)
    output_format = config.get("output_format", "hdf5")
    
    # Extract calorimeter-specific thresholds from config
    calo_config = config.get("calorimeter", {})
    ecal_energy_threshold = calo_config.get("ecal_energy_threshold", 5.0e-5)
    hcal_energy_threshold = calo_config.get("hcal_energy_threshold", 2.5e-4)
    ecal_time_min = calo_config.get("ecal_time_min", -1.0)
    ecal_time_max = calo_config.get("ecal_time_max", 10.0)
    hcal_time_min = calo_config.get("hcal_time_min", -1.0)
    hcal_time_max = calo_config.get("hcal_time_max", 10.0)
    
    logging.info("\nStarting calorimeter conversion with configuration:")
    logging.info(f"Campaign: {campaign}, Dataset: {dataset}, Version: {version}")
    logging.info(f"Input directory: {input_base_dir}")
    logging.info(f"Output root: {output_base_dir}")
    logging.info(f"Output format: {output_format}")
    logging.info(f"Chunk size: {chunk_size}, Run size: {run_size}")
    logging.info(f"ECal energy threshold: {ecal_energy_threshold} GeV")
    logging.info(f"HCal energy threshold: {hcal_energy_threshold} GeV")
    logging.info(f"ECal time window: [{ecal_time_min}, {ecal_time_max}] ns")
    logging.info(f"HCal time window: [{hcal_time_min}, {hcal_time_max}] ns")

    convert_calorimeter(
        input_base_dir,
        output_base_dir,
        f"{campaign}/{dataset}/{version}",
        chunk_size,
        run_size,
        args.chunk_index,
        config.get("max_chunks"),
        config,
        columns_keep=config.get("calohits_columns_keep"),
        output_format=output_format,
        ecal_energy_threshold=ecal_energy_threshold,
        hcal_energy_threshold=hcal_energy_threshold,
        ecal_time_min=ecal_time_min,
        ecal_time_max=ecal_time_max,
        hcal_time_min=hcal_time_min,
        hcal_time_max=hcal_time_max,
    )


if __name__ == "__main__":
    main()
