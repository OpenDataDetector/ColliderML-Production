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
import re
import yaml
import h5py
import os
import stat


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


def _infer_dataset_name_dot(version_dir: Path) -> str:
    """Infer campaign.dataset.version from directory tail components."""
    try:
        campaign, dataset, version = version_dir.parts[-3:]
        return f"{campaign}.{dataset}.{version}"
    except Exception:
        # Fallback to dot-joined last three parts if unusual layout
        return ".".join(version_dir.parts[-3:])


def _parse_chunks_from_log(version_dir: Path) -> list[tuple[int, int]]:
    """Parse event chunk ranges from convert_all log file.

    Looks for lines like: "Chunk X-Y timing summary:" under logs/stage_convert_all.
    """
    logs_dir = version_dir / "logs" / "stage_convert_all"
    if not logs_dir.exists():
        return []
    # Prefer a consolidated log.txt if present
    log_candidates = []
    txt_path = logs_dir / "log.txt"
    if txt_path.exists():
        log_candidates.append(txt_path)
    # Also consider *.out files
    log_candidates.extend(sorted(logs_dir.glob("*.out"), key=lambda p: p.stat().st_mtime, reverse=True))
    if not log_candidates:
        return []

    chunk_ranges: list[tuple[int, int]] = []
    pat = re.compile(r"Chunk\s+(\d+)-(\d+)\s+timing summary:")
    try:
        # Scan a few largest/most recent files first
        for path in log_candidates[:5]:
            with open(path, "r", errors="ignore") as f:
                for line in f:
                    m = pat.search(line)
                    if m:
                        a, b = int(m.group(1)), int(m.group(2))
                        chunk_ranges.append((a, b))
        # Deduplicate
        chunk_ranges = sorted(set(chunk_ranges))
    except Exception:
        pass
    return chunk_ranges


def _check_particles_file(h5_path: Path) -> list[str]:
    errors: list[str] = []
    try:
        with h5py.File(h5_path, "r") as f:
            if "events" not in f:
                errors.append(f"missing group /events in {h5_path}")
                return errors
            events = f["events"]
            has_any = False
            for name in events.keys():
                if not name.startswith("event_"):
                    continue
                has_any = True
                grp = events[name]
                if "particles" not in grp:
                    errors.append(f"{h5_path}:{name} missing dataset 'particles'")
            if not has_any:
                errors.append(f"{h5_path} has no event_* groups")
    except Exception as e:
        errors.append(f"failed to open {h5_path}: {e}")
    return errors


def _check_tracker_hits_file(h5_path: Path) -> list[str]:
    errors: list[str] = []
    try:
        with h5py.File(h5_path, "r") as f:
            if "events" not in f:
                errors.append(f"missing group /events in {h5_path}")
                return errors
            events = f["events"]
            has_any = False
            for name in events.keys():
                if not name.startswith("event_"):
                    continue
                has_any = True
                grp = events[name]
                if "measurements" not in grp:
                    errors.append(f"{h5_path}:{name} missing dataset 'measurements'")
            if not has_any:
                errors.append(f"{h5_path} has no event_* groups")
    except Exception as e:
        errors.append(f"failed to open {h5_path}: {e}")
    return errors


def _check_tracks_file(h5_path: Path) -> list[str]:
    errors: list[str] = []
    try:
        with h5py.File(h5_path, "r") as f:
            if "events" not in f:
                errors.append(f"missing group /events in {h5_path}")
                return errors
            events = f["events"]
            has_any = False
            for name in events.keys():
                if not name.startswith("event_"):
                    continue
                has_any = True
                grp = events[name]
                if "tracks" not in grp:
                    errors.append(f"{h5_path}:{name} missing dataset 'tracks'")
                # CSR arrays are recommended; warn if missing but do not hard-fail
                if "hit_ids_data" not in grp or "hit_ids_indptr" not in grp:
                    errors.append(f"{h5_path}:{name} missing CSR datasets 'hit_ids_data'/'hit_ids_indptr'")
            if not has_any:
                errors.append(f"{h5_path} has no event_* groups")
    except Exception as e:
        errors.append(f"failed to open {h5_path}: {e}")
    return errors


def _parse_args():
    """Parse CLI arguments for validation."""
    parser = argparse.ArgumentParser(description="Validate convert_all outputs")
    parser.add_argument("--stage", required=True, help="Stage name (expected: convert_all)")
    parser.add_argument("--runs-dir", required=True, help="Path to <version_dir>/runs")
    parser.add_argument("--publish", action="store_true", help="Publish (link) outputs to public directory after validation")
    parser.add_argument("--public-root", default=None, help="Public output base dir (overrides config common.public_output_base_dir)")
    parser.add_argument("--publish-mode", default=None, choices=["hardlink"], help="Publish mode; currently supports 'hardlink' only")
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


def _validate_for_objects(version_dir: Path, dataset_name_dot: str, objects: list[str], chunks: list[tuple[int, int]]) -> list[str]:
    """Validate existence and structure for produced files; return list of error messages."""
    errors: list[str] = []
    if chunks:
        for (start_event, end_event) in chunks:
            if "particles" in objects:
                p = version_dir / "truth" / "particles" / f"{dataset_name_dot}.truth.particles.events{start_event}-{end_event}.h5"
                if not p.exists():
                    errors.append(f"missing particles file: {p}")
                else:
                    errors.extend(_check_particles_file(p))
            if "tracker_hits" in objects:
                h = version_dir / "reco" / "tracker_hits" / f"{dataset_name_dot}.reco.tracker_hits.events{start_event}-{end_event}.h5"
                if not h.exists():
                    errors.append(f"missing tracker_hits file: {h}")
                else:
                    errors.extend(_check_tracker_hits_file(h))
            if "tracks" in objects:
                t = version_dir / "reco" / "tracks" / f"{dataset_name_dot}.reco.tracks.events{start_event}-{end_event}.h5"
                if not t.exists():
                    errors.append(f"missing tracks file: {t}")
                else:
                    errors.extend(_check_tracks_file(t))
    else:
        # Fallback: scan files directly
        if "particles" in objects:
            for p in sorted((version_dir / "truth" / "particles").glob(f"{dataset_name_dot}.truth.particles.events*.h5")):
                errors.extend(_check_particles_file(p))
        if "tracker_hits" in objects:
            for h in sorted((version_dir / "reco" / "tracker_hits").glob(f"{dataset_name_dot}.reco.tracker_hits.events*.h5")):
                errors.extend(_check_tracker_hits_file(h))
        if "tracks" in objects:
            for t in sorted((version_dir / "reco" / "tracks").glob(f"{dataset_name_dot}.reco.tracks.events*.h5")):
                errors.extend(_check_tracks_file(t))
    return errors


def _resolve_publish_settings(args, cfg: dict | None) -> tuple[bool, str | None, str]:
    """Determine whether to publish and resolve public root and mode.

    Returns (do_publish, public_root, publish_mode)
    """
    want_publish = bool(args.publish)
    publish_mode = args.publish_mode
    public_root_cli = args.public_root
    public_root_cfg = None
    public_root_override = None
    publish_cfg_flag = False
    if isinstance(cfg, dict):
        common_cfg = cfg.get("common", {})
        public_root_cfg = common_cfg.get("public_output_base_dir")
        val_cfg = cfg.get("validation_config", {})
        publish_cfg_flag = bool(val_cfg.get("publish_after_validation", False))
        # Allow validation_config to override the public root
        public_root_override = val_cfg.get("public_output_base_dir")
        if publish_mode is None:
            publish_mode = val_cfg.get("publish_mode")
    do_publish = want_publish or publish_cfg_flag
    public_root = public_root_cli or public_root_override or public_root_cfg
    if publish_mode is None:
        publish_mode = "hardlink"
    return do_publish, public_root, publish_mode


def _collect_produced_files(version_dir: Path, dataset_name_dot: str, objects: list[str], chunks: list[tuple[int, int]], public_root_path: Path) -> list[tuple[Path, Path]]:
    """Build list of (src, dst) files for publishing under public_root."""
    produced_files: list[tuple[Path, Path]] = []
    base_truth = version_dir / "truth" / "particles"
    base_hits = version_dir / "reco" / "tracker_hits"
    base_tracks = version_dir / "reco" / "tracks"
    target_base = public_root_path / version_dir.parts[-3] / version_dir.parts[-2] / version_dir.parts[-1]
    target_truth = target_base / "truth" / "particles"
    target_hits = target_base / "reco" / "tracker_hits"
    target_tracks = target_base / "reco" / "tracks"

    if chunks:
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
    else:
        if "particles" in objects and base_truth.exists():
            for p in base_truth.glob(f"{dataset_name_dot}.truth.particles.events*.h5"):
                produced_files.append((p, target_truth / p.name))
        if "tracker_hits" in objects and base_hits.exists():
            for h in base_hits.glob(f"{dataset_name_dot}.reco.tracker_hits.events*.h5"):
                produced_files.append((h, target_hits / h.name))
        if "tracks" in objects and base_tracks.exists():
            for t in base_tracks.glob(f"{dataset_name_dot}.reco.tracks.events*.h5"):
                produced_files.append((t, target_tracks / t.name))
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
                try:
                    if src.stat().st_size == dst.stat().st_size:
                        logger.info(f"Publish exists (size match), skipping: {dst}")
                        os.chmod(dst, 0o775)
                        for parent in [dst.parent, dst.parent.parent, dst.parent.parent.parent]:
                            if parent and parent.exists():
                                os.chmod(parent, 0o775)
                        continue
                    else:
                        dst.unlink()
                except Exception:
                    try:
                        dst.unlink()
                    except Exception:
                        pass
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
        dataset_name_dot = _infer_dataset_name_dot(version_dir)
        chunks = _parse_chunks_from_log(version_dir)
        if not chunks:
            logger.warning("No chunk ranges found in logs. Will attempt to infer from files.")

        errors = _validate_for_objects(version_dir, dataset_name_dot, objects, chunks)
        if errors:
            logger.error("Validation failures detected:")
            for e in errors:
                logger.error(f"  - {e}")
            sys.exit(1)

        logger.info("convert_all validation passed: all expected outputs found and structurally sound.")

        do_publish, public_root, publish_mode = _resolve_publish_settings(args, cfg)
        if not do_publish:
            sys.exit(0)

        if not public_root:
            logger.error("Publish requested but public_output_base_dir not provided (neither CLI nor config).")
            sys.exit(3)
        if publish_mode != "hardlink":
            logger.error(f"Unsupported publish_mode: {publish_mode}")
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


