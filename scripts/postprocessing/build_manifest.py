#!/usr/bin/env python3
"""
Build a single global event manifest with modality flags for postprocessing.

This stage scans all runs under the version directory and writes
<version_dir>/manifests/events_manifest.csv with columns:
  run_id, run_dir, local_event_id, global_event_id,
  has_hits, has_tracks, has_particles, has_calo

Usage (via run_stage.py):
  stage: build_manifest
  n_runs should be 1; no chunk-index is used for this stage.
"""

import argparse
import yaml
from pathlib import Path
from typing import Any, Dict

from utils.event_manifest import (
    build_event_manifest,
    write_event_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a global event manifest with modality flags")
    parser.add_argument("--config", required=True, type=str, help="Path to YAML config file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing manifest if present")
    return parser.parse_args()


def get_version_directory(config: Dict[str, Any]) -> Path:
    common = config.get("common", {})
    base_dir = Path(common["output_base_dir"])  # required
    campaign = config["campaign"]
    dataset = config["dataset"]
    version = config["version"]
    return base_dir / campaign / dataset / version


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Resolve paths
    version_dir = get_version_directory(config)
    manifests_dir = version_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = manifests_dir / "events_manifest.csv"

    # Respect existing manifest unless overwrite is requested
    if manifest_csv.exists() and not args.overwrite:
        print(f"Manifest already exists: {manifest_csv}. Use --overwrite to rebuild.")
        return

    # Patterns (defaults aligned with converters)
    edm4hep_file = config.get("edm4hep_file", "edm4hep.root")
    tracks_csv_pattern = config.get("tracks_csv_pattern", "event{:09d}-tracks_ambi.csv")
    simhits_file = config.get("simhits_file", "simhits.root")
    tracksummary_file = config.get("tracksummary_file", "tracksummary_ambi.root")

    # Build and write manifest. Input base is the version dir (expects 'runs' subdir).
    manifest_df = build_event_manifest(
        version_dir,
        edm4hep_file=edm4hep_file,
        tracks_csv_pattern=tracks_csv_pattern,
        simhits_file=simhits_file,
        tracksummary_file=tracksummary_file,
    )

    if manifest_df.empty:
        print("No events discovered. Manifest not written.")
        return

    out_path = write_event_manifest(manifest_df, version_dir)
    # Summary
    total = len(manifest_df)
    hits = int(manifest_df.has_hits.sum())
    tracks = int(manifest_df.has_tracks.sum())
    particles = int(manifest_df.has_particles.sum())
    calo = int(manifest_df.has_calo.sum())
    print(f"Wrote manifest to {out_path}")
    print(f"Total indexed rows: {total}")
    print(f"has_hits={hits}, has_tracks={tracks}, has_particles={particles}, has_calo={calo}")


if __name__ == "__main__":
    main()


