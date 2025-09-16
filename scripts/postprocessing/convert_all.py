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
# Reuse per-event tracks processing from dedicated module
from convert_tracks import process_event_for_tracks
from convert_digihits import convert_digihits, process_event_for_digihits, write_digihits_with_selection

from utils.path_utils import make_dir, get_run_paths
from utils.driver import iterate_and_process_chunks, local_events_for_run
from utils.track_utils import (
    load_root_file,
    load_track_summary,
    build_hdf5_tracks,
    write_tracks_with_selection,
    build_track_fitting_df_run,
)

logger = logging.getLogger(__name__)


def _get_objects(config: dict) -> list[str]:
    objs = config.get("objects", ["tracker_hits", "tracks", "particles", "calorimeter"])  # default set
    return [obj.lower() for obj in objs]


def _compute_paths(config: dict) -> tuple[Path, Path, str, str]:
    campaign = config["campaign"]
    dataset = config["dataset"]
    version = config["version"]
    common_cfg = config.get("common", {})
    input_base_dir = Path(common_cfg["output_base_dir"]) / campaign / dataset / version
    output_base_dir = Path(config.get("h5_output_dir", common_cfg["output_base_dir"]))
    dataset_base = f"{campaign}/{dataset}/{version}"
    dataset_name_dot = dataset_base.replace("/", ".")
    return input_base_dir, output_base_dir, dataset_base, dataset_name_dot


def _prepare_output_dirs(output_base_dir: Path, dataset_base: str) -> tuple[Path, Path, Path]:
    particles_out_dir = make_dir(output_base_dir, f"{dataset_base}/truth/particles")
    trkhits_out_dir = make_dir(output_base_dir, f"{dataset_base}/reco/tracker_hits")
    tracks_out_dir = make_dir(output_base_dir, f"{dataset_base}/reco/tracks")
    return particles_out_dir, trkhits_out_dir, tracks_out_dir


def _process_chunk_for_all(
    run_dirs: list[Path],
    start_event: int,
    end_event: int,
    start_run: int,
    start_local: int,
    end_run: int,
    end_local: int,
    *,
    run_size: int,
    objects: list[str],
    dataset_name_dot: str,
    particles_out_dir: Path,
    trkhits_out_dir: Path,
    tracks_out_dir: Path,
    particles_columns_keep: list[str] | None,
    digihits_columns_keep: list[str] | None,
    min_particle_energy: float | None,
    min_tracker_hits: int | None,
    digihits_measurements_columns: list[str] | None,
    tracks_csv_pattern: str,
    tracksummary_file: str,
    simhits_file: str,
    # new optional selection for tracks output
    tracks_columns_keep: list[str] | None = None,
) -> None:
    chunk_start_time = time.time()
    logger.info(f"Starting chunk processing for events {start_event}-{end_event}")
    
    import pandas as pd
    from pyedm4hep import EDM4hepEventBatch
    import awkward as ak

    particles_frames: list[pd.DataFrame] = []
    digihits_frames: list[pd.DataFrame] = []
    tracks_frames: list[pd.DataFrame] = []
    seen_pairs_tracks: set[tuple[int, int]] = set()
    seen_pairs_particles: set[tuple[int, int]] = set()
    seen_pairs_hits: set[tuple[int, int]] = set()

    run_processing_time = 0.0
    
    for abs_run in tqdm(range(start_run, end_run + 1), leave=False):
        run_start_time = time.time()
        run_dir = run_dirs[abs_run]
        edm4hep_path = Path(run_dir) / "edm4hep.root"
        if not edm4hep_path.exists():
            logger.warning(f"Missing EDM4hep file: {edm4hep_path}")
            continue

        local_events = local_events_for_run(
            start_run=start_run,
            start_local=start_local,
            end_run=end_run,
            end_local=end_local,
            abs_run=abs_run,
            run_size=run_size,
        )

        # Load only local events for this run to reduce I/O
        batch_load_start = time.time()
        batch = EDM4hepEventBatch(str(edm4hep_path), events=list(local_events))
        logger.debug(
            f"EDM4hep batch load for run {abs_run} (events={len(list(local_events))}): {time.time() - batch_load_start:.3f}s"
        )

        tracksummary_arrays = None
        track_fitting_df_run = None
        if "tracks" in objects:
            tracks_load_start = time.time()
            ts_path = Path(run_dir) / tracksummary_file
            if ts_path.exists():
                try:
                    tracksummary_arrays = load_track_summary(str(ts_path))
                    # Normalize into a per-run DataFrame with event linkage
                    import pandas as pd
                    track_fitting_df_run = build_track_fitting_df_run(tracksummary_arrays, run_size)
                except Exception as e:
                    logger.warning(f"Failed to load tracksummary at {ts_path}: {e}")
            tracks_load_time = time.time() - tracks_load_start
            logger.debug(f"Track summary loading for run {abs_run}: {tracks_load_time:.3f}s")

        if "particles" in objects:
            particles_start_time = time.time()
            particles_root_path = Path(run_dir) / "particles.root"
            digi_particles_df_run = None
            if particles_root_path.exists():
                try:
                    included_columns = [
                        "event_id",
                        "vx",
                        "vy",
                        "vz",
                        "px",
                        "py",
                        "pz",
                        "vertex_primary",
                    ]
                    digi_particles_df_run = load_root_file(str(particles_root_path), included_columns=included_columns)
                except Exception as e:
                    logger.warning(f"Failed to load particles.root at {particles_root_path}: {e}")

            if len(list(local_events)) > 0:
                df_run = build_particles_df_with_parents_and_vertex(
                    batch,
                    str(edm4hep_path),
                    digi_particles_df_run,
                    local_events=local_events,
                    min_particle_energy=min_particle_energy,
                    min_tracker_hits=min_tracker_hits,
                )
                if not df_run.empty and "event_id" in df_run.columns:
                    df_run = df_run.copy()
                    df_run["event_id"] = df_run["event_id"] + abs_run * run_size
                if not df_run.empty:
                    for le in local_events:
                        pair = (abs_run, le)
                        if pair in seen_pairs_particles:
                            logger.error(f"Overlap detected for particles on (run,local_event)=({abs_run},{le})")
                        seen_pairs_particles.add(pair)
                    particles_frames.append(df_run)
            particles_time = time.time() - particles_start_time
            logger.debug(f"Particles processing for run {abs_run}: {particles_time:.3f}s")

        digihits_run_df = None
        if "tracker_hits" in objects:
            digihits_start_time = time.time()
            measurements_path = Path(run_dir) / "measurements.root"
            if measurements_path.exists():
                try:
                    included_columns = (
                        digihits_measurements_columns
                        if digihits_measurements_columns is not None
                        else [
                        "event_nr",
                            "volume_id",
                            "layer_id",
                            "surface_id",
                            "rec_x",
                            "rec_y",
                            "rec_z",
                            "true_x",
                            "true_y",
                            "true_z",
                        ]
                    )
                    digi_measurements_df_all = load_root_file(str(measurements_path), included_columns=included_columns)
                except Exception as e:
                    logger.error(f"Failed to load measurements for run {abs_run}: {e}")
                    digi_measurements_df_all = pd.DataFrame()
                hits_fetch_start = time.time()
                hits_all = batch.get_tracker_hits_df()
                logger.debug(f"Loaded tracker hits DataFrame for run {abs_run} in {time.time() - hits_fetch_start:.3f}s")
                evs_for_run = []
                for local_event_num in local_events:
                    ev_hits = hits_all[hits_all.event_id == local_event_num] if not hits_all.empty else None
                    ev_meas = (
                        digi_measurements_df_all[digi_measurements_df_all.event_nr == local_event_num].copy()
                        if "event_nr" in getattr(digi_measurements_df_all, "columns", [])
                        else digi_measurements_df_all.copy()
                    )
                    ev_df = process_event_for_digihits(abs_run * run_size + local_event_num, local_event_num, ev_meas, ev_hits)
                    if not ev_df.empty:
                        pair = (abs_run, local_event_num)
                        if pair in seen_pairs_hits:
                            logger.error(
                                f"Overlap detected for tracker_hits on (run,local_event)=({abs_run},{local_event_num})"
                            )
                        seen_pairs_hits.add(pair)
                        digihits_frames.append(ev_df)
                        evs_for_run.append(ev_df)
                if evs_for_run:
                    digihits_run_df = pd.concat(evs_for_run, ignore_index=True)
            else:
                logger.warning(f"Missing measurements file: {measurements_path}")
            digihits_time = time.time() - digihits_start_time
            logger.debug(f"Digihits processing for run {abs_run}: {digihits_time:.3f}s")

        if "tracks" in objects and track_fitting_df_run is not None and digihits_run_df is not None:
            tracks_proc_start_time = time.time()
            for local_event_num in local_events:
                global_event_num = abs_run * run_size + local_event_num
                try:
                    per_ev_start = time.time()
                    event_df = process_event_for_tracks(
                        run_dir=Path(run_dir),
                        local_event_num=local_event_num,
                        global_event_num=global_event_num,
                        track_fitting_df_event=track_fitting_df_run[track_fitting_df_run.get('event_nr', -1) == local_event_num].copy(),
                        tracks_csv_pattern=tracks_csv_pattern,
                        digihits_run_df=digihits_run_df,
                    )
                    logger.debug(
                        f"Tracks event processed run={abs_run} local={local_event_num} rows={0 if event_df is None else len(event_df)} in {time.time() - per_ev_start:.3f}s"
                    )
                except Exception as e:
                    logger.warning(
                        f"Tracks processing failed for (run,local)=({abs_run},{local_event_num}): {e}"
                    )
                    continue
                if event_df is None or event_df.empty:
                    continue
                pair = (abs_run, local_event_num)
                if pair in seen_pairs_tracks:
                    logger.error(
                        f"Overlap detected for tracks on (run,local_event)=({abs_run},{local_event_num})"
                    )
                seen_pairs_tracks.add(pair)
                tracks_frames.append(event_df)
            tracks_proc_time = time.time() - tracks_proc_start_time
            logger.debug(f"Tracks processing for run {abs_run}: {tracks_proc_time:.3f}s")
        
        run_time = time.time() - run_start_time
        run_processing_time += run_time
        logger.debug(f"Run {abs_run} total processing time: {run_time:.3f}s")

    # File writing phase
    writing_start_time = time.time()
    expected_events = end_event - start_event + 1
    
    if "particles" in objects and particles_frames:
        particles_write_start = time.time()
        particles_all = pd.concat(particles_frames, ignore_index=True)
        particles_out = Path(particles_out_dir) / (
            f"{dataset_name_dot}.truth.particles.events{start_event}-{end_event}.h5"
        )
        processed_events_particles = len(seen_pairs_particles)
        if processed_events_particles != expected_events:
            logger.warning(
                f"Particles chunk events expected={expected_events}, processed={processed_events_particles}"
            )
        logger.info(f"Writing particles to: {particles_out} (rows={len(particles_all)})")
        write_particles_with_selection(particles_all, str(particles_out), columns_keep=particles_columns_keep)
        if particles_out.exists():
            logger.info(f"Wrote particles file: {particles_out}")
        else:
            logger.warning(f"Particles file not created (possibly filtered to empty): {particles_out}")
        particles_write_time = time.time() - particles_write_start
        logger.debug(f"Particles file writing time: {particles_write_time:.3f}s")
        
    if "tracker_hits" in objects and digihits_frames:
        digihits_write_start = time.time()
        digihits_all = pd.concat(digihits_frames, ignore_index=True)
        trkhits_out = Path(trkhits_out_dir) / (
            f"{dataset_name_dot}.reco.tracker_hits.events{start_event}-{end_event}.h5"
        )
        processed_events_hits = len(seen_pairs_hits)
        if processed_events_hits != expected_events:
            logger.warning(
                f"Tracker hits chunk events expected={expected_events}, processed={processed_events_hits}"
            )
        logger.info(f"Writing tracker hits to: {trkhits_out} (rows={len(digihits_all)})")
        write_digihits_with_selection(digihits_all, str(trkhits_out), columns_keep=digihits_columns_keep)
        if trkhits_out.exists():
            logger.info(f"Wrote tracker hits file: {trkhits_out}")
        else:
            logger.warning(
                f"Tracker hits file not created (possibly filtered to empty): {trkhits_out}"
            )
        digihits_write_time = time.time() - digihits_write_start
        logger.debug(f"Tracker hits file writing time: {digihits_write_time:.3f}s")
        
    if "tracks" in objects and tracks_frames:
        tracks_write_start = time.time()
        import pandas as pd
        tracks_all = pd.concat(tracks_frames, ignore_index=True)
        tracks_out = Path(tracks_out_dir) / (
            f"{dataset_name_dot}.reco.tracks.events{start_event}-{end_event}.h5"
        )
        processed_events_tracks = len(seen_pairs_tracks)
        if processed_events_tracks != expected_events:
            logger.warning(
                f"Tracks chunk events expected={expected_events}, processed={processed_events_tracks}"
            )
        logger.info(f"Writing tracks to: {tracks_out} (rows={len(tracks_all)})")
        write_tracks_with_selection(tracks_all, str(tracks_out), columns_keep=tracks_columns_keep)
        if tracks_out.exists():
            logger.info(f"Wrote tracks file: {tracks_out}")
        else:
            logger.warning(f"Tracks file not created (possibly filtered to empty): {tracks_out}")
        tracks_write_time = time.time() - tracks_write_start
        logger.debug(f"Tracks file writing time: {tracks_write_time:.3f}s")

    writing_time = time.time() - writing_start_time
    chunk_total_time = time.time() - chunk_start_time
    
    logger.info(f"Chunk {start_event}-{end_event} timing summary:")
    logger.info(f"  Run processing: {run_processing_time:.3f}s")
    logger.info(f"  File writing: {writing_time:.3f}s")
    logger.info(f"  Total chunk time: {chunk_total_time:.3f}s")


 


def convert_all(config: dict, chunk_index: int | None = None) -> None:
    logger.debug(
        f"Starting conversion with config: campaign={config.get('campaign')}, dataset={config.get('dataset')}, version={config.get('version')}"
    )

    input_base_dir, output_base_dir, dataset_base, dataset_name_dot = _compute_paths(config)
    chunk_size = int(config.get("chunk_size", 1000))
    run_size = int(config.get("run_size", 10))
    objects = _get_objects(config)

    logger.debug(f"Input base directory: {input_base_dir}")
    logger.debug(f"Output base directory: {output_base_dir}")
    logger.debug(f"Objects to convert: {objects}")

    start_time = time.time()

    run_dirs = get_run_paths(input_base_dir)
    logger.info(f"Found {len(run_dirs)} runs. chunk_size={chunk_size}, run_size={run_size}, chunk_index={chunk_index}")

    particles_out_dir, trkhits_out_dir, tracks_out_dir = _prepare_output_dirs(output_base_dir, dataset_base)

    particles_columns_keep = config.get("particles_columns_keep")
    digihits_columns_keep = config.get("digihits_columns_keep")
    tracks_csv_pattern = config.get("tracks_csv_pattern", "event{:09d}-tracks_ambi.csv")
    tracksummary_file = config.get("tracksummary_file", "tracksummary_ambi.root")
    simhits_file = config.get("simhits_file", "simhits.root")
    tracks_columns_keep = config.get("tracks_columns_keep")
    min_particle_energy = config.get("min_particle_energy")
    min_tracker_hits = config.get("min_tracker_hits")
    digihits_measurements_columns = config.get("digihits_measurements_columns")

    processing_start_time = time.time()
    
    iterate_and_process_chunks(
        run_dirs=run_dirs,
        run_size=run_size,
        chunk_size=chunk_size,
        config=config,
        chunk_index=chunk_index,
        process_chunk_fn=lambda start_event, end_event, start_run, start_local, end_run, end_local: _process_chunk_for_all(
            run_dirs,
            start_event,
            end_event,
            start_run,
            start_local,
            end_run,
            end_local,
            run_size=run_size,
            objects=objects,
            dataset_name_dot=dataset_name_dot,
            particles_out_dir=particles_out_dir,
            trkhits_out_dir=trkhits_out_dir,
            tracks_out_dir=tracks_out_dir,
            particles_columns_keep=particles_columns_keep,
            digihits_columns_keep=digihits_columns_keep,
            min_particle_energy=min_particle_energy,
            min_tracker_hits=min_tracker_hits,
            digihits_measurements_columns=digihits_measurements_columns,
            tracks_csv_pattern=tracks_csv_pattern,
            tracksummary_file=tracksummary_file,
            simhits_file=simhits_file,
            tracks_columns_keep=tracks_columns_keep,
        ),
    )

    processing_time = time.time() - processing_start_time
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"\nConversion timing summary:")
    logger.info(f"  Setup time: {processing_start_time - start_time:.2f}s")
    logger.info(f"  Processing time: {processing_time:.2f}s")
    logger.info(f"  Total conversion time: {total_time:.2f}s")
    logger.debug("Conversion process completed successfully")


def main():
    main_start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Convert all EDM4HEP data to HDF5 (config-driven)")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    parser.add_argument("--chunk-index", type=int, default=None, help="Optional chunk index to process (for distributed runs)")
    args = parser.parse_args()
    
    config_load_start = time.time()
    logger.debug(f"Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    logger.debug("Config loaded successfully")
    config_load_time = time.time() - config_load_start
    logger.debug(f"Config loading time: {config_load_time:.3f}s")
    
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
    
    main_total_time = time.time() - main_start_time
    logger.info(f"Total script execution time: {main_total_time:.2f}s")

if __name__ == "__main__":
    main() 