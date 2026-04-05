"""
Data loading from HuggingFace Hub.

Streams pre-generated ColliderML datasets without requiring Docker or
any simulation software. Uses pyarrow for efficient Parquet reading
and huggingface_hub for download/caching.
"""

import os
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

# Known datasets on HuggingFace
HF_REPO = "CERN/ColliderML-Release-1"

# Dataset naming convention: {channel}_pu{pileup}
# e.g., "ttbar_pu200" -> channel=ttbar, pileup=200
KNOWN_DATASETS = {
    "ttbar_pu0", "ttbar_pu40", "ttbar_pu200",
    "higgs_portal_pu0", "higgs_portal_pu10", "higgs_portal_pu200",
    "zmumu_pu0", "zmumu_pu200",
    "zee_pu0", "zee_pu200",
    "diphoton_pu0", "diphoton_pu200",
    "jets_pu0", "jets_pu200",
    "susy_gmsb_pu0", "susy_gmsb_pu200",
    "hidden_valley_pu0", "hidden_valley_pu200",
    "zprime_pu0", "zprime_pu200",
    "single_muon_pu0",
}

VALID_TABLES = {"tracks", "tracker_hits", "particles", "calo_hits"}


def load(dataset, tables=None, columns=None, max_events=None, cache_dir=None):
    """
    Load pre-generated ColliderML data from HuggingFace.

    Downloads and caches Parquet files, then returns them as a dictionary
    of pyarrow Tables (or pandas DataFrames if pandas is available).

    Args:
        dataset: Dataset name, e.g. "ttbar_pu200", "higgs_portal_pu10".
        tables: List of table names to load. Options: "tracks", "tracker_hits",
                "particles", "calo_hits". Defaults to all available.
        columns: Optional dict mapping table name -> list of columns to read.
                 Reduces memory usage for large datasets.
        max_events: Maximum number of row groups to read (approximate event limit).
        cache_dir: Directory for caching downloaded files.
                   Defaults to ~/.cache/huggingface/hub.

    Returns:
        dict: Mapping of table name -> pyarrow Table.
              If only one table is requested, returns it directly.

    Examples:
        # Load all tables for ttbar with pileup 200
        data = colliderml.load("ttbar_pu200")
        data["tracks"]  # pyarrow Table

        # Load specific tables with column selection
        data = colliderml.load("ttbar_pu200",
                               tables=["tracks", "particles"],
                               columns={"tracks": ["d0", "z0", "phi", "theta", "qop"]})

        # Convert to pandas
        df = colliderml.load("ttbar_pu200", tables=["tracks"]).to_pandas()
    """
    if tables is None:
        tables = list(VALID_TABLES)
    elif isinstance(tables, str):
        tables = [tables]

    for t in tables:
        if t not in VALID_TABLES:
            raise ValueError(
                f"Unknown table '{t}'. Valid tables: {sorted(VALID_TABLES)}"
            )

    columns = columns or {}

    # Discover available Parquet files in the HF repo
    file_map = _discover_files(dataset, tables, cache_dir)

    result = {}
    for table_name in tables:
        files = file_map.get(table_name, [])
        if not files:
            continue
        table = _read_parquet_files(files, columns.get(table_name), max_events)
        result[table_name] = table

    if not result:
        available = ", ".join(sorted(KNOWN_DATASETS))
        raise FileNotFoundError(
            f"No data found for dataset '{dataset}'. "
            f"Known datasets: {available}"
        )

    # If only one table requested, return it directly
    if len(tables) == 1 and len(result) == 1:
        return next(iter(result.values()))

    return result


def _discover_files(dataset, tables, cache_dir):
    """Find and download Parquet files for the given dataset and tables."""
    try:
        all_files = list_repo_files(HF_REPO, repo_type="dataset")
    except Exception as e:
        raise ConnectionError(
            f"Failed to list files in {HF_REPO}. "
            f"Check your internet connection. Error: {e}"
        ) from e

    file_map = {}
    for table_name in tables:
        # Match files like: ttbar_pu200/tracks/part-00000.parquet
        # or: ttbar_pu200/tracks.parquet
        matching = []
        prefix = f"{dataset}/{table_name}"
        for f in all_files:
            if f.startswith(prefix) and f.endswith(".parquet"):
                matching.append(f)

        if not matching:
            continue

        # Download files
        local_paths = []
        for remote_path in sorted(matching):
            local_path = hf_hub_download(
                HF_REPO,
                filename=remote_path,
                repo_type="dataset",
                cache_dir=cache_dir,
            )
            local_paths.append(local_path)

        file_map[table_name] = local_paths

    return file_map


def _read_parquet_files(paths, columns, max_events):
    """Read Parquet files into a single pyarrow Table."""
    tables = []
    for path in paths:
        pf = pq.ParquetFile(path)
        if max_events is not None:
            # Read only up to max_events row groups
            n_groups = min(max_events, pf.metadata.num_row_groups)
            table = pf.read_row_groups(range(n_groups), columns=columns)
        else:
            table = pf.read(columns=columns)
        tables.append(table)

    if len(tables) == 1:
        return tables[0]

    import pyarrow as pa
    return pa.concat_tables(tables)


def list_datasets():
    """List available datasets on HuggingFace."""
    return sorted(KNOWN_DATASETS)
