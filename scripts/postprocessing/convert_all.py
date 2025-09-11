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
from tqdm import tqdm

from convert_particles import convert_particles, build_particles_df_with_parents_and_vertex, write_particles_with_selection
# from convert_calorimeter import convert_calorimeter
# from convert_tracks import convert_tracks
from convert_digihits import convert_digihits, process_event_for_digihits, write_digihits_with_selection

from utils.path_utils import make_dir
from utils.track_utils import (
    load_root_file,
    create_particle_barcode_map,
    get_majority_particle_id,
    convert_hit_ids,
    load_track_summary,
    build_hdf5_tracks,
)

logger = logging.getLogger(__name__)


def convert_all(config: dict, chunk_index: int | None = None) -> None:
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
    # Event-based totals
    num_runs = 0
    
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
    num_runs = len(run_dirs)
    total_events = num_runs * run_size
    logger.debug(f"Retrieved {len(run_dirs)} run directories from {input_base_dir}")
    
    particles_out_dir = make_dir(output_base_dir, f"{dataset_base}/truth/particles")
    trkhits_out_dir = make_dir(output_base_dir, f"{dataset_base}/reco/tracker_hits")
    tracks_out_dir = make_dir(output_base_dir, f"{dataset_base}/reco/tracks")
    
    logger.debug(f"Created output directories: particles={particles_out_dir}, tracker_hits={trkhits_out_dir}")

    particles_columns_keep = config.get("particles_columns_keep")
    digihits_columns_keep = config.get("digihits_columns_keep")
    # Track I/O patterns (optional; used if 'tracks' requested)
    tracks_csv_pattern = config.get("tracks_csv_pattern", "event{:09d}-tracks_ambi.csv")
    tracksummary_file = config.get("tracksummary_file", "tracksummary_ambi.root")
    simhits_file = config.get("simhits_file", "simhits.root")
    
    logger.debug(f"Column selection - particles: {particles_columns_keep}, digihits: {digihits_columns_keep}")

    logger.info(f"Found {num_runs} runs. Processing with run_size={run_size}, runs_per_chunk={runs_per_chunk}, max_chunks={max_chunks}, chunk_index={chunk_index}")

    # Determine event window based on chunk_index (no overlap across processes)
    if chunk_index is not None:
        start_event = chunk_index * chunk_size
        if start_event >= total_events:
            logger.info(f"Chunk {chunk_index} start_event {start_event} >= total_events {total_events}; nothing to do")
            return
        end_event = min(total_events - 1, start_event + chunk_size - 1)
        logger.info(f"Processing chunk_index={chunk_index}: events {start_event}-{end_event}")
    else:
        start_event = 0
        end_event = total_events - 1
        logger.info(f"Processing full range of events {start_event}-{end_event}")

    # Map event window to runs and local event ranges
    start_run, start_local = divmod(start_event, run_size)
    end_run, end_local = divmod(end_event, run_size)
    logger.debug(f"Event window maps to runs {start_run}..{end_run}, start_local={start_local}, end_local={end_local}")

    # Accumulators for this chunk
    particles_frames = []
    digihits_frames = []
    tracks_frames = []
    seen_pairs_particles: set[tuple[int,int]] = set()
    seen_pairs_hits: set[tuple[int,int]] = set()

    for abs_run in tqdm(range(start_run, end_run + 1)):
        run_dir = run_dirs[abs_run]
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

        # Optional: Preload supporting inputs for tracks once per run
        run_tracks_enabled = ("tracks" in objects)
        tracksummary_arrays = None
        simhits_df_all = None
        if run_tracks_enabled:
            # Load tracksummary arrays (per-run container of per-event records)
            ts_path = Path(run_dir) / tracksummary_file
            if ts_path.exists():
                try:
                    tracksummary_arrays = load_track_summary(str(ts_path))
                    logger.debug(f"Loaded tracksummary from {ts_path}")
                except Exception as e:
                    logger.warning(f"Failed to load tracksummary at {ts_path}: {e}")
            else:
                logger.debug(f"No tracksummary file found at: {ts_path}")

            # Load simhits once per run
            simhits_path = Path(run_dir) / simhits_file
            if simhits_path.exists():
                try:
                    simhits_df_all = load_root_file(str(simhits_path))
                    logger.debug(f"Loaded simhits from {simhits_path} shape={simhits_df_all.shape if simhits_df_all is not None else 'None'}")
                except Exception as e:
                    logger.warning(f"Failed to load simhits at {simhits_path}: {e}")
            else:
                logger.debug(f"No simhits file found at: {simhits_path}")

        # Particles: optional digi merge
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
                
            # Determine local events for this run within the event window
            if start_run == end_run:
                local_events = range(start_local, end_local + 1)
            elif abs_run == start_run:
                local_events = range(start_local, run_size)
            elif abs_run == end_run:
                local_events = range(0, end_local + 1)
            else:
                local_events = range(0, run_size)
            logger.debug(f"Particles local_events for run {abs_run}: {list(local_events)[:3]}... (len={len(list(local_events))})")

            if len(list(local_events)) > 0:
                logger.debug("Building particles DataFrame with parents and vertex info")
                df_run = build_particles_df_with_parents_and_vertex(
                    batch,
                    str(edm4hep_path),
                    digi_particles_df_run,
                    local_events=local_events,
                    min_particle_energy=config.get("min_particle_energy"),
                    min_tracker_hits=config.get("min_tracker_hits"),
                )
                if not df_run.empty and "event_id" in df_run.columns:
                    df_run = df_run.copy()
                    df_run["event_id"] = df_run["event_id"] + abs_run * run_size
                if not df_run.empty:
                    # Overlap guard for particles: ensure no duplicate (run,local_event)
                    for le in local_events:
                        pair = (abs_run, le)
                        if pair in seen_pairs_particles:
                            logger.error(f"Overlap detected for particles on (run,local_event)=({abs_run},{le})")
                        seen_pairs_particles.add(pair)
                    particles_frames.append(df_run)

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

                # Determine local events for this run within the event window
                if start_run == end_run:
                    local_events = range(start_local, end_local + 1)
                elif abs_run == start_run:
                    local_events = range(start_local, run_size)
                elif abs_run == end_run:
                    local_events = range(0, end_local + 1)
                else:
                    local_events = range(0, run_size)

                # Build per-event merged frames
                logger.debug(f"Processing {len(list(local_events))} events for tracker hits merge in run {abs_run}")
                for local_event_num in local_events:
                    logger.debug(f"Processing local event {local_event_num}")
                    ev_hits = hits_all[hits_all.event_id == local_event_num] if not hits_all.empty else None
                    ev_meas = digi_measurements_df_all[digi_measurements_df_all.event_nr == local_event_num].copy() if 'event_nr' in digi_measurements_df_all.columns else digi_measurements_df_all.copy()
                    
                    logger.debug(f"Event {local_event_num}: hits shape={ev_hits.shape if ev_hits is not None else 'None'}, measurements shape={ev_meas.shape if ev_meas is not None else 'None'}")
                    
                    ev_df = process_event_for_digihits(abs_run * run_size + local_event_num, local_event_num, ev_meas, ev_hits)
                    logger.debug(f"Event {local_event_num}: merged DataFrame shape={ev_df.shape}")
                    
                    if not ev_df.empty:
                        # Overlap guard for hits
                        pair = (abs_run, local_event_num)
                        if pair in seen_pairs_hits:
                            logger.error(f"Overlap detected for tracker_hits on (run,local_event)=({abs_run},{local_event_num})")
                        seen_pairs_hits.add(pair)
                        digihits_frames.append(ev_df)
                        logger.debug(f"Added event {local_event_num} from run {abs_run} to merged frames")

        # Tracks: combine track-finding CSV + tracksummary ROOT, map to MC via simhits/edm4hep hits
        if "tracks" in objects and tracksummary_arrays is not None and simhits_df_all is not None:
            logger.debug("Processing tracks object")
            try:
                hits_all = batch.get_tracker_hits_df()
            except Exception as e:
                logger.warning(f"Failed to get tracker hits for tracks mapping in run {abs_run}: {e}")
                hits_all = None

            # Determine local events similar to other objects
            if start_run == end_run:
                local_events_for_tracks = range(start_local, end_local + 1)
            elif abs_run == start_run:
                local_events_for_tracks = range(start_local, run_size)
            elif abs_run == end_run:
                local_events_for_tracks = range(0, end_local + 1)
            else:
                local_events_for_tracks = range(0, run_size)

            for local_event_num in local_events_for_tracks:
                # CSV presence gate
                tracks_csv_path = Path(run_dir) / tracks_csv_pattern.format(local_event_num)
                if not tracks_csv_path.exists():
                    continue
                # Build per-event inputs
                try:
                    import pandas as pd
                    tracks_csv = pd.read_csv(tracks_csv_path)
                except Exception as e:
                    logger.warning(f"Failed to read tracks CSV for event {local_event_num} in run {abs_run}: {e}")
                    continue

                try:
                    arrays = tracksummary_arrays[local_event_num]
                    track_data = {}
                    for field in getattr(arrays, 'fields', []):
                        if field == 'event_nr':
                            continue
                        try:
                            import awkward as ak
                            array_np = ak.to_numpy(arrays[field])
                            if len(getattr(array_np, 'shape', ())) == 1:
                                track_data[field] = array_np
                        except Exception:
                            continue
                    import numpy as np
                    track_fitting_df = pd.DataFrame(track_data).rename(columns={"track_nr": "track_id"})
                except Exception as e:
                    logger.warning(f"Failed to build tracksummary DF for event {local_event_num} in run {abs_run}: {e}")
                    continue

                # Local slices for mapping
                if hits_all is None or hits_all.empty:
                    logger.warning(f"Missing tracker hits for event {local_event_num}; skipping track MC mapping")
                    continue
                local_event_edm4hep_hits = hits_all[hits_all.event_id == local_event_num].copy()

                # Simhits slice: prefer 'event_id', else 'event_nr'
                if simhits_df_all is None or simhits_df_all.empty:
                    continue
                if 'event_id' in simhits_df_all.columns:
                    local_event_simhits = simhits_df_all[simhits_df_all.event_id == local_event_num].copy()
                elif 'event_nr' in simhits_df_all.columns:
                    local_event_simhits = simhits_df_all[simhits_df_all.event_nr == local_event_num].copy()
                else:
                    logger.warning("Simhits dataframe missing event id column; skipping tracks")
                    continue

                try:
                    particle_barcode_map = create_particle_barcode_map(local_event_edm4hep_hits, local_event_simhits)
                except Exception as e:
                    logger.warning(f"Failed to create particle barcode map for event {local_event_num}: {e}")
                    continue

                try:
                    majority_particle_ids = tracks_csv.Hits_ID.apply(
                        get_majority_particle_id, args=(local_event_simhits, particle_barcode_map)
                    )
                except Exception as e:
                    logger.warning(f"Failed to compute majority particle id for event {local_event_num}: {e}")
                    continue

                global_event_num = abs_run * run_size + local_event_num
                try:
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
                except Exception as e:
                    logger.warning(f"Failed to assemble track-finding data for event {local_event_num}: {e}")
                    continue

                try:
                    track_fitting_data = {
                        "event_id": global_event_num,
                        "track_id": track_fitting_df.track_id.values if not track_fitting_df.empty else [],
                        "d0": track_fitting_df.eLOC0_fit.values if 'eLOC0_fit' in track_fitting_df else [],
                        "z0": track_fitting_df.eLOC1_fit.values if 'eLOC1_fit' in track_fitting_df else [],
                        "phi": track_fitting_df.ePHI_fit.values if 'ePHI_fit' in track_fitting_df else [],
                        "theta": track_fitting_df.eTHETA_fit.values if 'eTHETA_fit' in track_fitting_df else [],
                        "qop": track_fitting_df.eQOP_fit.values if 'eQOP_fit' in track_fitting_df else [],
                        "time": track_fitting_df.eT_fit.values if 'eT_fit' in track_fitting_df else [],
                        "d0_truth": track_fitting_df.t_d0.values if 't_d0' in track_fitting_df else [],
                        "z0_truth": track_fitting_df.t_z0.values if 't_z0' in track_fitting_df else [],
                        "phi_truth": track_fitting_df.t_phi.values if 't_phi' in track_fitting_df else [],
                        "theta_truth": track_fitting_df.t_theta.values if 't_theta' in track_fitting_df else [],
                        "charge_truth": track_fitting_df.t_charge.values if 't_charge' in track_fitting_df else [],
                        "p_truth": track_fitting_df.t_p.values if 't_p' in track_fitting_df else [],
                        "pT_truth": track_fitting_df.t_pT.values if 't_pT' in track_fitting_df else [],
                        "time_truth": track_fitting_df.t_time.values if 't_time' in track_fitting_df else [],
                    }
                except Exception as e:
                    logger.warning(f"Failed to assemble track-fitting data for event {local_event_num}: {e}")
                    continue

                import pandas as pd
                try:
                    full_track_df = pd.DataFrame(track_finding_data)
                    event_df = full_track_df.merge(pd.DataFrame(track_fitting_data), on=["event_id", "track_id"], how="left")
                except Exception as e:
                    logger.warning(f"Failed to merge track data for event {local_event_num}: {e}")
                    continue

                # Overlap guard
                pair = (abs_run, local_event_num)
                if pair in seen_pairs_hits:  # reuse seen set name to avoid new var? Use dedicated set
                    logger.error(f"Overlap detected for tracks on (run,local_event)=({abs_run},{local_event_num})")
                tracks_frames.append(event_df)

        # Other objects can be integrated similarly into this single-pass if needed
        logger.debug(f"Completed processing run {abs_run}")

    # After all runs for this chunk, write combined outputs
    expected_events = end_event - start_event + 1
    if "particles" in objects:
        if particles_frames:
            particles_all = pd.concat(particles_frames, ignore_index=True)
            particles_out = Path(particles_out_dir) / (
                f"{dataset_name_dot}.truth.particles.events{start_event}-{end_event}.h5"
            )
            # Validate expected events vs processed set
            processed_events_particles = len(seen_pairs_particles)
            if processed_events_particles != expected_events:
                logger.warning(f"Particles chunk events expected={expected_events}, processed={processed_events_particles}")
            logger.info(f"Writing particles to: {particles_out} (rows={len(particles_all)})")
            write_particles_with_selection(particles_all, str(particles_out), columns_keep=particles_columns_keep)
            if particles_out.exists():
                logger.info(f"Wrote particles file: {particles_out}")
            else:
                logger.warning(f"Particles file not created (possibly filtered to empty): {particles_out}")
        else:
            logger.info("No particles to write for this chunk")

    if "tracker_hits" in objects:
        if digihits_frames:
            digihits_all = pd.concat(digihits_frames, ignore_index=True)
            trkhits_out = Path(trkhits_out_dir) / (
                f"{dataset_name_dot}.reco.tracker_hits.events{start_event}-{end_event}.h5"
            )
            processed_events_hits = len(seen_pairs_hits)
            if processed_events_hits != expected_events:
                logger.warning(f"Tracker hits chunk events expected={expected_events}, processed={processed_events_hits}")
            logger.info(f"Writing tracker hits to: {trkhits_out} (rows={len(digihits_all)})")
            write_digihits_with_selection(digihits_all, str(trkhits_out), columns_keep=digihits_columns_keep)
            if trkhits_out.exists():
                logger.info(f"Wrote tracker hits file: {trkhits_out}")
            else:
                logger.warning(f"Tracker hits file not created (possibly filtered to empty): {trkhits_out}")
        else:
            logger.info("No tracker hits to write for this chunk")

    if "tracks" in objects:
        if tracks_frames:
            import pandas as pd
            tracks_all = pd.concat(tracks_frames, ignore_index=True)
            tracks_out = Path(tracks_out_dir) / (
                f"{dataset_name_dot}.reco.tracks.events{start_event}-{end_event}.h5"
            )
            logger.info(f"Writing tracks to: {tracks_out} (rows={len(tracks_all)})")
            build_hdf5_tracks(tracks_all, str(tracks_out))
            if tracks_out.exists():
                logger.info(f"Wrote tracks file: {tracks_out}")
            else:
                logger.warning(f"Tracks file not created (possibly filtered to empty): {tracks_out}")
        else:
            logger.info("No tracks to write for this chunk")

    end_time = time.time()
    logger.info(f"\nTotal conversion time: {end_time - start_time:.2f} seconds")
    logger.debug("Conversion process completed successfully")


def main():
    parser = argparse.ArgumentParser(description="Convert all EDM4HEP data to HDF5 (config-driven)")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    parser.add_argument("--chunk-index", type=int, default=None, help="Optional chunk index to process (for distributed runs)")
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
    convert_all(config, chunk_index=args.chunk_index)
    logger.debug("convert_all function completed")

if __name__ == "__main__":
    main() 