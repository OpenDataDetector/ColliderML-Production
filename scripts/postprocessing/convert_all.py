#!/usr/bin/env python3
"""
Run all EDM4HEP to HDF5 conversions in sequence, driven by a YAML config.
"""

import argparse
import time
from pathlib import Path
import yaml
import logging
import logging.config

from convert_particles import convert_particles, build_particles_df_with_parents_and_vertex, write_particles_with_selection
# from convert_calorimeter import convert_calorimeter
# from convert_tracks import convert_tracks
from convert_digihits import convert_digihits, process_event_for_digihits, write_digihits_with_selection

from utils.path_utils import make_dir

logger = logging.getLogger(__name__)


def convert_all(config: dict) -> None:
    campaign = config["campaign"]
    dataset = config["dataset"]
    version = config["version"]
    
    logger.debug(f"Starting conversion with config: campaign={campaign}, dataset={dataset}, version={version}")

    common_cfg = config.get("common", {})
    # Use a single root for both sim and postprocessing
    input_base_dir = Path(common_cfg["output_base_dir"]) / campaign / dataset / version
    output_base_dir = Path(config.get("h5_output_dir", common_cfg["output_base_dir"]))
    
    logger.debug(f"Input base directory: {input_base_dir}")
    logger.debug(f"Output base directory: {output_base_dir}")

    # Chunking
    chunk_size = int(config.get("chunk_size", 1000))
    run_size = int(config.get("run_size", 10))
    runs_per_chunk = max(1, chunk_size // run_size)
    max_chunks = config.get("max_chunks")
    max_runs = runs_per_chunk * int(max_chunks) if max_chunks is not None else None
    
    logger.debug(f"Chunk size: {chunk_size}, Run size: {run_size}")

    # Objects to convert
    objects = config.get("objects", ["tracker_hits", "tracks", "particles", "calorimeter"])  # default set
    objects = [obj.lower() for obj in objects]
    
    logger.debug(f"Objects to convert: {objects}")

    dataset_base = f"{campaign}/{dataset}/{version}"
    dataset_name_dot = dataset_base.replace("/", ".")
    logger.debug(f"Dataset base path: {dataset_base}")

    start_time = time.time()

    # Unified single-pass batch loading per run, iterating events, per object
    # Use pyedm4hep for batch loading
    import pandas as pd
    import numpy as np
    from pyedm4hep import EDM4hepEventBatch
    from utils.path_utils import get_run_paths
    from utils.track_utils import load_root_file

    logger.debug("Importing required modules completed")

    run_dirs = get_run_paths(input_base_dir)
    logger.debug(f"Retrieved {len(run_dirs)} run directories from {input_base_dir}")
    
    particles_out_dir = make_dir(output_base_dir, f"{dataset_base}/truth/particles")
    trkhits_out_dir = make_dir(output_base_dir, f"{dataset_base}/reco/tracker_hits")
    
    logger.debug(f"Created output directories: particles={particles_out_dir}, tracker_hits={trkhits_out_dir}")

    particles_columns_keep = config.get("particles_columns_keep")
    digihits_columns_keep = config.get("digihits_columns_keep")
    
    logger.debug(f"Column selection - particles: {particles_columns_keep}, digihits: {digihits_columns_keep}")

    logger.info(f"Found {len(run_dirs)} runs. Processing with run_size={run_size}, runs_per_chunk={runs_per_chunk}, max_chunks={max_chunks}")
    for abs_run, run_dir in enumerate(run_dirs):
        if max_runs is not None and abs_run >= max_runs:
            logger.info(f"Reached max_chunks limit: processed {abs_run} runs (runs_per_chunk={runs_per_chunk}, max_chunks={max_chunks})")
            break
        logger.debug(f"Processing run {abs_run}: {run_dir}")
        
        edm4hep_path = Path(run_dir) / "edm4hep.root"
        if not edm4hep_path.exists():
            logger.warning(f"Missing EDM4hep file: {edm4hep_path}")
            continue

        logger.debug(f"Found EDM4hep file: {edm4hep_path}")

        # Single batch open per run
        logger.debug(f"Creating EDM4hepEventBatch for events 0-{run_size-1}")
        batch = EDM4hepEventBatch(str(edm4hep_path), events=range(run_size))
        logger.debug("EDM4hepEventBatch created successfully")
        # Preload collections lazily when getters are called

        # Particles: optional digi merge
        particles_df_all = pd.DataFrame()
        if "particles" in objects:
            logger.debug("Processing particles object")
            particles_root_path = Path(run_dir) / "particles.root"
            digi_particles_df_run = None
            if particles_root_path.exists():
                logger.debug(f"Found particles.root file: {particles_root_path}")
                try:
                    included_columns = [
                        "event_id",
                        "vx", "vy", "vz",
                        "px", "py", "pz",
                        "vertex_primary",
                    ]
                    logger.debug(f"Loading particles.root with columns: {included_columns}")
                    digi_particles_df_run = load_root_file(str(particles_root_path), included_columns=included_columns)
                    logger.debug(f"Loaded particles.root successfully, shape: {digi_particles_df_run.shape if digi_particles_df_run is not None else 'None'}")
                except Exception as e:
                    logger.warning(f"Failed to load particles.root at {particles_root_path}: {e}")
            else:
                logger.debug(f"No particles.root file found at: {particles_root_path}")
                
            logger.debug("Building particles DataFrame with parents and vertex info")
            particles_df_all = build_particles_df_with_parents_and_vertex(
                batch,
                str(edm4hep_path),
                digi_particles_df_run,
                local_events=range(run_size),
                min_particle_energy=config.get("min_particle_energy"),
                min_tracker_hits=config.get("min_tracker_hits"),
            )
            logger.debug(f"Built particles DataFrame, shape: {particles_df_all.shape}")
            
            start_event = abs_run * run_size
            end_event = start_event + run_size - 1
            # Match convert_particles.py naming:
            #   <dataset_name>.truth.particles.events<start>-<end>.h5
            particles_out = Path(particles_out_dir) / (
                f"{dataset_name_dot}.truth.particles.events{start_event}-{end_event}.h5"
            )
            logger.debug(f"Writing particles to: {particles_out}")
            write_particles_with_selection(particles_df_all, str(particles_out), columns_keep=particles_columns_keep)
            logger.debug(f"Successfully wrote particles file")

        # Digi hits: needs measurements merge per event
        if "tracker_hits" in objects:
            logger.debug("Processing tracker_hits object")
            measurements_path = Path(run_dir) / "measurements.root"
            if not measurements_path.exists():
                logger.warning(f"Missing measurements file: {measurements_path}")
            else:
                logger.debug(f"Found measurements file: {measurements_path}")
                try:
                    # Load only subset of measurement columns to save IO
                    included_columns = config.get("digihits_measurements_columns", [
                        "event_nr",
                        "volume_id", "layer_id", "surface_id",
                        "rec_x", "rec_y", "rec_z",
                        "true_x", "true_y", "true_z",
                    ])
                    logger.debug(f"Loading measurements.root with columns: {included_columns}")
                    digi_measurements_df_all = load_root_file(str(measurements_path), included_columns=included_columns)
                    logger.debug(f"Loaded measurements.root successfully, shape: {digi_measurements_df_all.shape if digi_measurements_df_all is not None else 'None'}")
                except Exception as e:
                    logger.error(f"Failed to load measurements for run {abs_run}: {e}")
                    digi_measurements_df_all = pd.DataFrame()
                
                logger.debug("Getting tracker hits DataFrame from batch")
                hits_all = batch.get_tracker_hits_df()
                logger.debug(f"Retrieved tracker hits DataFrame, shape: {hits_all.shape if not hits_all.empty else 'Empty'}")
                
                # Build per-event merged frames
                merged_frames = []
                logger.debug(f"Processing {run_size} events for tracker hits merge")
                for local_event_num in range(run_size):
                    logger.debug(f"Processing local event {local_event_num}")
                    ev_hits = hits_all[hits_all.event_id == local_event_num] if not hits_all.empty else None
                    ev_meas = digi_measurements_df_all[digi_measurements_df_all.event_nr == local_event_num].copy() if 'event_nr' in digi_measurements_df_all.columns else digi_measurements_df_all.copy()
                    
                    logger.debug(f"Event {local_event_num}: hits shape={ev_hits.shape if ev_hits is not None else 'None'}, measurements shape={ev_meas.shape if ev_meas is not None else 'None'}")
                    
                    ev_df = process_event_for_digihits(abs_run * run_size + local_event_num, local_event_num, ev_meas, ev_hits)
                    logger.debug(f"Event {local_event_num}: merged DataFrame shape={ev_df.shape}")
                    
                    if not ev_df.empty:
                        merged_frames.append(ev_df)
                        logger.debug(f"Added event {local_event_num} to merged frames")
                
                if merged_frames:
                    logger.debug(f"Concatenating {len(merged_frames)} merged frames")
                    merged_all = pd.concat(merged_frames, ignore_index=True)
                    logger.debug(f"Final merged DataFrame shape: {merged_all.shape}")
                    
                    start_event = abs_run * run_size
                    end_event = start_event + run_size - 1
                    # Match convert_digihits.py naming:
                    #   <dataset_name>.reco.tracker_hits.events<start>-<end>.h5
                    trkhits_out = Path(trkhits_out_dir) / (
                        f"{dataset_name_dot}.reco.tracker_hits.events{start_event}-{end_event}.h5"
                    )
                    logger.debug(f"Writing tracker hits to: {trkhits_out}")
                    write_digihits_with_selection(merged_all, str(trkhits_out), columns_keep=digihits_columns_keep)
                    logger.debug(f"Successfully wrote tracker hits file")
                else:
                    logger.debug("No merged frames to write for tracker hits")

        # Other objects can be integrated similarly into this single-pass if needed
        logger.debug(f"Completed processing run {abs_run}")

    end_time = time.time()
    logger.info(f"\nTotal conversion time: {end_time - start_time:.2f} seconds")
    logger.debug("Conversion process completed successfully")


def main():
    parser = argparse.ArgumentParser(description="Convert all EDM4HEP data to HDF5 (config-driven)")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    args = parser.parse_args()
    
    logger.debug(f"Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    logger.debug("Config loaded successfully")
    
    # One-liner logging control: honor simple config key "log_level" (default INFO)
    level_name = str(config.get("log_level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        force=True,
    )
    
    logger.debug("Starting convert_all function")
    convert_all(config)
    logger.debug("convert_all function completed")

if __name__ == "__main__":
    main() 