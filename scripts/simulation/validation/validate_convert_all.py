#!/usr/bin/env python3
"""
Validation script for the convert_all postprocessing stage.

This script is invoked after the convert_all stage completes (via job_submission).
It validates that expected HDF5 outputs exist and have basic structural integrity.

Invocation (from job_submission):
  python validate_convert_all.py --stage convert_all --runs-dir <version_dir>/runs

Checks performed:
  - Locate version directory from runs directory
  - Load the latest config snapshot from <version_dir>/configs if available
  - Determine which objects were requested (particles, tracker_hits, tracks)
  - Parse the convert_all stage log to extract processed chunk event ranges
  - For each chunk and enabled object, verify output H5 file exists
  - Perform structural checks on each file (groups/datasets present, minimal sanity)

Exit codes:
  0: success (no issues)
  1: validation failures were found
  2: unexpected error during validation
  3: publish requested but failed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import logging
import yaml
import h5py
import os


logger = logging.getLogger(__name__)


def _load_config_snapshot(version_dir: Path) -> dict | None:
    """Load the most recent YAML config snapshot under <version_dir>/configs, if any."""
    configs_dir = version_dir / "configs"
    if not configs_dir.exists():
        return None
    yaml_files = sorted(configs_dir.glob("*.yaml"))
    if not yaml_files:
        return None
    # Use the newest by mtime
    yaml_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    try:
        with open(yaml_files[0], "r") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _infer_dataset_name_dot(version_dir: Path, cfg: dict | None) -> str:
    """Infer campaign.dataset.version from config when available, else from directory."""
    if isinstance(cfg, dict):
        c = cfg.get("campaign")
        d = cfg.get("dataset")
        v = cfg.get("version")
        if all(isinstance(x, str) for x in (c, d, v)):
            return f"{c}.{d}.{v}"
    try:
        campaign, dataset, version = version_dir.parts[-3:]
        return f"{campaign}.{dataset}.{version}"
    except Exception:
        return ".".join(version_dir.parts[-3:])


def _expected_chunks_from_config(cfg: dict | None) -> list[tuple[int, int]]:
    """Compute expected event chunk ranges from config (no log parsing).

    Uses: job_config.n_runs, run_size, chunk_size.
    """
    if not isinstance(cfg, dict):
        return []
    job_cfg = cfg.get("job_config", {}) if isinstance(cfg.get("job_config"), dict) else {}
    try:
        n_runs = int(job_cfg.get("n_runs"))
        run_size = int(cfg.get("run_size"))
        chunk_size = int(cfg.get("chunk_size"))
        if n_runs <= 0 or run_size <= 0 or chunk_size <= 0:
            return []
    except Exception:
        return []
    num_events = n_runs * run_size
    num_chunks = (num_events + chunk_size - 1) // chunk_size
    # Apply optional cap if present
    try:
        cap = cfg.get("max_chunks")
        if cap is None:
            # driver.determine_chunk_cap also allows interactive cap via job_config.n_runs; here we honor explicit max_chunks only
            cap_int = None
        else:
            cap_int = int(cap)
        if cap_int is not None and cap_int >= 0:
            num_chunks = min(num_chunks, cap_int)
    except Exception:
        pass
    ranges: list[tuple[int, int]] = []
    for idx in range(num_chunks):
        start_event = idx * chunk_size
        end_event = min(num_events, start_event + chunk_size) - 1
        ranges.append((start_event, end_event))
    return ranges


def _validate_h5_file(file_path: Path, required_dataset: str, *, require_csr: bool = False) -> list[str]:
    """Generic HDF5 validator for convert_all outputs under /events/event_*/..."""
    errors: list[str] = []
    try:
        with h5py.File(file_path, "r") as f:
            if "events" not in f:
                errors.append(f"missing group /events in {file_path}")
                return errors
            events = f["events"]
            has_any = False
            for name in events.keys():
                if not name.startswith("event_"):
                    continue
                has_any = True
                grp = events[name]
                if required_dataset not in grp:
                    errors.append(f"{file_path}:{name} missing dataset '{required_dataset}'")
                if require_csr:
                    if "hit_ids_data" not in grp or "hit_ids_indptr" not in grp:
                        errors.append(f"{file_path}:{name} missing CSR datasets 'hit_ids_data'/'hit_ids_indptr'")
            if not has_any:
                errors.append(f"{file_path} has no event_* groups")
    except Exception as e:
        errors.append(f"failed to open {file_path}: {e}")
    return errors


def _parse_args():
    """Parse CLI arguments for validation (minimal args)."""
    parser = argparse.ArgumentParser(description="Validate convert_all outputs")
    parser.add_argument("--stage", required=True, help="Stage name (expected: convert_all)")
    parser.add_argument("--runs-dir", required=True, help="Path to <version_dir>/runs")
    return parser.parse_args()


def _infer_objects_from_config(cfg: dict | None) -> list[str]:
    """Return list of objects to validate based on config snapshot (lowercased)."""
    default_objs = ["particles", "tracker_hits", "tracks"]
    if not isinstance(cfg, dict):
        return default_objs
    objs_cfg = cfg.get("objects")
    if isinstance(objs_cfg, list) and objs_cfg:
        return [str(o).lower() for o in objs_cfg]
    return default_objs


def _resolve_data_root(version_dir: Path) -> Path:
    """Return base directory where outputs live: new fixed structure under 'hdf5/'"""
    return version_dir / "hdf5"


def _validate_for_objects(version_dir: Path, dataset_name_dot: str, objects: list[str], chunks: list[tuple[int, int]]) -> list[str]:
    """Validate existence and structure for produced files; return list of error messages."""
    errors: list[str] = []
    base_root = _resolve_data_root(version_dir)
    if chunks:
        for (start_event, end_event) in chunks:
            if "particles" in objects:
                p = base_root / "truth" / "particles" / f"{dataset_name_dot}.truth.particles.events{start_event}-{end_event}.h5"
                if not p.exists():
                    errors.append(f"missing particles file: {p}")
                else:
                    errors.extend(_validate_h5_file(p, "particles"))
            if "tracker_hits" in objects:
                h = base_root / "reco" / "tracker_hits" / f"{dataset_name_dot}.reco.tracker_hits.events{start_event}-{end_event}.h5"
                if not h.exists():
                    errors.append(f"missing tracker_hits file: {h}")
                else:
                    errors.extend(_validate_h5_file(h, "measurements"))
            if "tracks" in objects:
                t = base_root / "reco" / "tracks" / f"{dataset_name_dot}.reco.tracks.events{start_event}-{end_event}.h5"
                if not t.exists():
                    errors.append(f"missing tracks file: {t}")
                else:
                    errors.extend(_validate_h5_file(t, "tracks", require_csr=True))
    else:
        # Without chunks, do not guess; prefer explicit ranges to avoid picking up stale files
        errors.append("no expected chunk ranges available from config; aborting to avoid stale files")
    return errors


def _resolve_publish_settings(cfg: dict | None) -> tuple[bool, str | None]:
    """Determine whether to publish and resolve public root from config only."""
    if not isinstance(cfg, dict):
        return False, None
    val_cfg = cfg.get("validation_config", {}) if isinstance(cfg.get("validation_config"), dict) else {}
    do_publish = bool(val_cfg.get("publish_after_validation", False))
    public_root = val_cfg.get("public_output_base_dir")
    if not public_root:
        common_cfg = cfg.get("common", {}) if isinstance(cfg.get("common"), dict) else {}
        public_root = common_cfg.get("public_output_base_dir")
    return do_publish, public_root


def _collect_produced_files(version_dir: Path, dataset_name_dot: str, objects: list[str], chunks: list[tuple[int, int]], public_root_path: Path) -> list[tuple[Path, Path]]:
    """Build list of (src, dst) files for publishing under public_root."""
    produced_files: list[tuple[Path, Path]] = []
    base_root = _resolve_data_root(version_dir)
    base_truth = base_root / "truth" / "particles"
    base_hits = base_root / "reco" / "tracker_hits"
    base_tracks = base_root / "reco" / "tracks"
    target_base = public_root_path / version_dir.parts[-3] / version_dir.parts[-2] / version_dir.parts[-1]
    target_truth = target_base / "truth" / "particles"
    target_hits = target_base / "reco" / "tracker_hits"
    target_tracks = target_base / "reco" / "tracks"

    for (start_event, end_event) in chunks:
        if "particles" in objects:
            src = base_truth / f"{dataset_name_dot}.truth.particles.events{start_event}-{end_event}.h5"
            produced_files.append((src, target_truth / src.name))
        if "tracker_hits" in objects:
            src = base_hits / f"{dataset_name_dot}.reco.tracker_hits.events{start_event}-{end_event}.h5"
            produced_files.append((src, target_hits / src.name))
        if "tracks" in objects:
            src = base_tracks / f"{dataset_name_dot}.reco.tracks.events{start_event}-{end_event}.h5"
            produced_files.append((src, target_tracks / src.name))
    return produced_files


def _publish_hardlinks(file_pairs: list[tuple[Path, Path]]) -> list[str]:
    """Publish via hardlinks and set perms; return list of error messages."""
    failed: list[str] = []
    for src, dst in file_pairs:
        try:
            if not src.exists():
                failed.append(f"source missing: {src}")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                dst.unlink()
            os.link(src, dst)
            os.chmod(dst, 0o775)
            for parent in [dst.parent, dst.parent.parent, dst.parent.parent.parent]:
                if parent and parent.exists():
                    os.chmod(parent, 0o775)
            logger.info(f"Published (hardlink): {dst} -> {src}")
        except OSError as e:
            failed.append(f"link failed: {src} -> {dst}: {e}")
        except Exception as e:
            failed.append(f"unexpected: {src} -> {dst}: {e}")
    return failed


def main():
    """Entry point that orchestrates validation and optional publishing."""
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        if args.stage != "convert_all":
            logger.warning(f"Validation invoked for stage '{args.stage}', expected 'convert_all'")

        runs_dir = Path(args.runs_dir).resolve()
        if not runs_dir.exists() or not runs_dir.is_dir():
            logger.error(f"runs-dir not found or not a directory: {runs_dir}")
            sys.exit(1)

        version_dir = runs_dir.parent
        logger.info(f"Version dir inferred: {version_dir}")

        cfg = _load_config_snapshot(version_dir)
        objects = _infer_objects_from_config(cfg)
        dataset_name_dot = _infer_dataset_name_dot(version_dir, cfg)
        chunks = _expected_chunks_from_config(cfg)
        if not chunks:
            logger.error("Could not derive expected chunk ranges from config. Ensure job_config.n_runs, run_size, and chunk_size are present.")
            sys.exit(1)

        errors = _validate_for_objects(version_dir, dataset_name_dot, objects, chunks)
        if errors:
            logger.error("Validation failures detected:")
            for e in errors:
                logger.error(f"  - {e}")
            sys.exit(1)

        logger.info("convert_all validation passed: all expected outputs found and structurally sound.")

        do_publish, public_root = _resolve_publish_settings(cfg)
        if not do_publish:
            sys.exit(0)

        if not public_root:
            logger.error("Publish requested but public_output_base_dir not provided (neither CLI nor config).")
            sys.exit(3)

        public_root_path = Path(public_root)
        file_pairs = _collect_produced_files(version_dir, dataset_name_dot, objects, chunks, public_root_path)
        publish_errors = _publish_hardlinks(file_pairs)
        if publish_errors:
            logger.error("Publishing encountered errors:")
            for m in publish_errors:
                logger.error(f"  - {m}")
            sys.exit(3)

        logger.info("Publishing completed successfully.")
        sys.exit(0)

    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during validation: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()


