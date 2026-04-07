"""
Pre-cache a small set of events per dataset for instant Space load.

Run this once at build time (or in a HF Space build hook) to populate
`_cached_events/{dataset}/{table}.parquet` with ~50 events each.

Usage:
    python cache_events.py
"""

from pathlib import Path

import pyarrow.parquet as pq

import colliderml
from app import DATASETS, EVENTS_PER_DATASET, CACHE_DIR


def main():
    CACHE_DIR.mkdir(exist_ok=True)
    for dataset in DATASETS:
        out_dir = CACHE_DIR / dataset
        out_dir.mkdir(exist_ok=True)
        print(f"Caching {dataset}...")
        try:
            tables = colliderml.load(
                dataset,
                tables=["tracker_hits", "particles", "tracks"],
                max_events=EVENTS_PER_DATASET,
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        if not isinstance(tables, dict):
            tables = {"data": tables}

        for name, table in tables.items():
            out_path = out_dir / f"{name}.parquet"
            pq.write_table(table, str(out_path))
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"  {name}: {len(table)} rows ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
