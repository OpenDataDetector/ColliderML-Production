"""
Event manifest utilities.

Build a single, global event manifest that indexes all available events across
runs and marks per-modality availability with boolean flags. This allows all
postprocessing converters (hits, tracks, particles, calo) to consume the same
manifest and select their events by filtering on the relevant modality column.

The manifest schema:
  - run_id: int             # numeric run directory name
  - run_dir: str            # absolute path to the run directory
  - local_event_id: int     # event index local to the run (0-based)
  - global_event_id: int    # globally contiguous event index (0-based)
  - has_hits: bool          # EDM4hep event exists (optionally with tracker hits)
  - has_tracks: bool        # per-event tracks CSV exists and required run-level files exist
  - has_particles: bool     # EDM4hep event exists (placeholder: same as has_hits)
  - has_calo: bool          # EDM4hep event exists (placeholder: same as has_hits)

Chunk metadata helpers are also provided to split the manifest into fixed-size
event chunks for deterministic HDF5 file creation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
import re

import pandas as pd
import uproot

from .utils import get_run_paths


_TRACKS_EVENT_RE = re.compile(r"event(\d{9})-tracks_ambi\\.csv$")


def _discover_hits_local_events(edm4hep_path: Path) -> Set[int]:
    """
    Discover available local event indices for hits from EDM4hep ROOT file.

    This uses the number of entries in the "events" TTree as the available
    local event indices (0..n-1). It does not validate tracker collection
    non-emptiness to keep the scan fast.
    """
    if not edm4hep_path.exists():
        return set()
    try:
        with uproot.open(edm4hep_path) as f:
            if "events" not in f:
                return set()
            n_entries = f["events"].num_entries
            return set(range(int(n_entries)))
    except Exception:
        return set()


def _discover_tracks_local_events(run_dir: Path, tracks_csv_pattern: str) -> Set[int]:
    """
    Discover available local event indices for tracks by listing CSV files.

    Parameters
    - tracks_csv_pattern: e.g. "event{:09d}-tracks_ambi.csv". When provided,
      we synthesize a regex to match and extract the 9-digit event number.
    """
    # Derive a regex from pattern if possible, otherwise fall back to a default
    # known pattern.
    regex: Optional[re.Pattern[str]] = None
    if "{:09d}" in tracks_csv_pattern and tracks_csv_pattern.startswith("event"):
        # Escape dots and build regex
        escaped = re.escape(tracks_csv_pattern.replace("{:09d}", "PLACEHOLDER"))
        escaped = escaped.replace("PLACEHOLDER", r"(\\d{9})")
        regex = re.compile(escaped + r"$")
    else:
        regex = _TRACKS_EVENT_RE

    local_ids: Set[int] = set()
    for csv_path in run_dir.glob("*.csv"):
        m = regex.search(csv_path.name)
        if not m:
            continue
        try:
            local_ids.add(int(m.group(1)))
        except Exception:
            continue
    return local_ids


def _file_exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False


def build_event_manifest(
    input_base_dir: Path | str,
    *,
    edm4hep_file: str = "edm4hep.root",
    tracks_csv_pattern: str = "event{:09d}-tracks_ambi.csv",
    simhits_file: str = "simhits.root",
    tracksummary_file: str = "tracksummary_ambi.root",
) -> pd.DataFrame:
    """
    Scan all runs below input_base_dir and build a global event manifest with
    per-modality availability flags.

    The manifest includes union of local event IDs present in any modality.
    """
    input_base_dir = Path(input_base_dir)
    run_dirs = get_run_paths(input_base_dir)

    records: List[Dict] = []
    global_event_counter = 0

    for run_dir in run_dirs:
        # Determine run_id from directory name
        try:
            run_id = int(run_dir.name)
        except ValueError:
            # Skip non-numeric run directories for safety
            continue

        edm4hep_path = run_dir / edm4hep_file
        simhits_path = run_dir / simhits_file
        tracksummary_path = run_dir / tracksummary_file

        hits_events = _discover_hits_local_events(edm4hep_path)
        tracks_events = _discover_tracks_local_events(run_dir, tracks_csv_pattern)

        # Union of all discovered local event IDs in this run
        all_local_ids = sorted(hits_events | tracks_events)
        if not all_local_ids:
            continue

        # Run-level presence required for tracks beyond the CSV itself
        has_run_level_tracks_inputs = _file_exists(simhits_path) and _file_exists(tracksummary_path)

        for local_event_id in all_local_ids:
            has_hits = local_event_id in hits_events
            has_tracks = (local_event_id in tracks_events) and has_run_level_tracks_inputs

            # For now, assume particles and calo presence correlates with EDM4hep event presence.
            has_particles = has_hits
            has_calo = has_hits

            records.append(
                {
                    "run_id": run_id,
                    "run_dir": str(run_dir.resolve()),
                    "local_event_id": int(local_event_id),
                    "global_event_id": int(global_event_counter),
                    "has_hits": bool(has_hits),
                    "has_tracks": bool(has_tracks),
                    "has_particles": bool(has_particles),
                    "has_calo": bool(has_calo),
                }
            )
            global_event_counter += 1

    manifest_df = pd.DataFrame.from_records(records)
    if not manifest_df.empty:
        # Ensure deterministic ordering
        manifest_df = manifest_df.sort_values(["global_event_id"]).reset_index(drop=True)
    return manifest_df


def write_event_manifest(manifest_df: pd.DataFrame, output_dir: Path | str) -> Path:
    """
    Write the manifest to a CSV file under output_dir/manifests/events_manifest.csv.
    Returns the path to the written CSV.
    """
    output_dir = Path(output_dir)
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests_dir / "events_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    return manifest_path


def build_chunks_from_manifest(
    manifest_df: pd.DataFrame, *, chunk_size: int
) -> pd.DataFrame:
    """
    Build a chunk mapping DataFrame with one row per chunk:
      - chunk_index: int
      - start_global_event_id: int
      - end_global_event_id: int (inclusive)
      - num_events: int
    """
    if manifest_df.empty:
        return pd.DataFrame(columns=[
            "chunk_index", "start_global_event_id", "end_global_event_id", "num_events"
        ])

    total_events = len(manifest_df)
    chunk_rows: List[Dict] = []
    chunk_index = 0
    for start in range(0, total_events, chunk_size):
        end = min(start + chunk_size, total_events) - 1
        chunk_rows.append(
            {
                "chunk_index": chunk_index,
                "start_global_event_id": int(manifest_df.iloc[start]["global_event_id"]),
                "end_global_event_id": int(manifest_df.iloc[end]["global_event_id"]),
                "num_events": int(end - start + 1),
            }
        )
        chunk_index += 1
    return pd.DataFrame(chunk_rows)


def write_chunks_manifest(chunks_df: pd.DataFrame, output_dir: Path | str) -> Path:
    """
    Write the chunks manifest CSV under output_dir/manifests/chunks_manifest.csv.
    Returns the path to the written CSV.
    """
    output_dir = Path(output_dir)
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = manifests_dir / "chunks_manifest.csv"
    chunks_df.to_csv(chunks_path, index=False)
    return chunks_path


