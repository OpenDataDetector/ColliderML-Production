#!/usr/bin/env python3
"""Test load_dataset with streaming=True vs streaming=False on a pu0 config.

Run from repo root or with:
  python scripts/dataset/test_load_dataset_streaming.py

We measure time to first event and (where possible) how many Parquet files
are touched, to see if streaming=True loads only the first file.

Cache location: All Hugging Face caches are forced to TEST_CACHE (default
/tmp/colliderml_hf_test_cache) so we do not fill the user's home quota.
- HF_HUB_CACHE = raw downloads from the Hub (Parquet files) — default ~/.cache/huggingface/hub
- HF_DATASETS_CACHE = processed dataset cache — default ~/.cache/huggingface/datasets
"""

import os
import sys
import time
from pathlib import Path

# Force ALL HF caches to /tmp so we never fill the user's home quota.
# HF_HUB_CACHE = where raw files (Parquet) are downloaded; defaults to ~/.cache/huggingface/hub
# HF_DATASETS_CACHE = where datasets library writes processed data; we set per run below.
TEST_CACHE = Path(os.environ.get("COLLIDERML_TEST_CACHE", "/tmp/colliderml_hf_test_cache"))
os.environ["HF_HUB_CACHE"] = str(TEST_CACHE / "hub")
os.environ["HF_DATASETS_CACHE"] = str(TEST_CACHE / "datasets")

REPO_ID = "CERN/ColliderML-Release-1"
CONFIG = "ttbar_pu0_particles"


def count_parquet_in_cache(cache_root: Path) -> int:
    """Count .parquet files under cache_root (for this dataset)."""
    n = 0
    for p in cache_root.rglob("*.parquet"):
        n += 1
    return n


def size_mb(cache_root: Path) -> float:
    """Total size in MB of parquet files under cache_root."""
    total = 0
    for p in cache_root.rglob("*.parquet"):
        total += p.stat().st_size
    return total / (1024 * 1024)


def run_streaming_true():
    """Load with streaming=True, get first event only."""
    # HF_HUB_CACHE already set at top; datasets cache can share or stay under TEST_CACHE
    Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    start = time.perf_counter()
    ds = load_dataset(REPO_ID, CONFIG, split="train", streaming=True)
    first = next(iter(ds))
    elapsed = time.perf_counter() - start

    hub_p = Path(os.environ["HF_HUB_CACHE"])
    ds_p = Path(os.environ["HF_DATASETS_CACHE"])
    n_parquet = count_parquet_in_cache(hub_p) + count_parquet_in_cache(ds_p)
    mb = size_mb(hub_p) + size_mb(ds_p)
    return elapsed, n_parquet, mb, first


def run_streaming_false():
    """Load with streaming=False, split='train[:1]' (first row only)."""
    Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    start = time.perf_counter()
    ds = load_dataset(REPO_ID, CONFIG, split="train[:1]")
    first = ds[0]
    elapsed = time.perf_counter() - start

    hub_p = Path(os.environ["HF_HUB_CACHE"])
    ds_p = Path(os.environ["HF_DATASETS_CACHE"])
    n_parquet = count_parquet_in_cache(hub_p) + count_parquet_in_cache(ds_p)
    mb = size_mb(hub_p) + size_mb(ds_p)
    return elapsed, n_parquet, mb, first


def main():
    print("ColliderML load_dataset test: streaming=True vs streaming=False")
    print("=" * 60)
    print(f"  Repo: {REPO_ID}")
    print(f"  Config: {CONFIG}")
    print(f"  All caches under: {TEST_CACHE} (not home — HF_HUB_CACHE + HF_DATASETS_CACHE)")
    print()

    # Run streaming=True first (often we want to recommend this for "first file only")
    print("1) streaming=True — get first event only")
    print("   (Expectation: only first Parquet file should be downloaded.)")
    try:
        t1, n1, mb1, ev1 = run_streaming_true()
        print(f"   Time to first event: {t1:.2f} s")
        print(f"   Parquet files in cache: {n1}")
        print(f"   Cache size: {mb1:.2f} MB")
        print(f"   First event keys: {list(ev1.keys())[:6]}...")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        n1 = mb1 = t1 = -1

    print()

    print("2) streaming=False, split='train[:1]' — load then take first row")
    print("   (Expectation: all Parquet files for train may be downloaded.)")
    try:
        t2, n2, mb2, ev2 = run_streaming_false()
        print(f"   Time to first event: {t2:.2f} s")
        print(f"   Parquet files in cache: {n2}")
        print(f"   Cache size: {mb2:.2f} MB")
        print(f"   First event keys: {list(ev2.keys())[:6]}...")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        n2 = mb2 = t2 = -1

    print()
    print("=" * 60)
    print("Summary")
    print("-" * 60)
    if t1 >= 0:
        print(f"  streaming=True:  {t1:.2f} s to first event (cache: {n1} parquet, {mb1:.2f} MB)")
        print("  => Streaming does not cache all files; only data for iterated rows is fetched.")
    if t2 >= 0:
        print(f"  streaming=False: {t2:.2f} s to first event (cache: {n2} parquet, {mb2:.2f} MB)")
        if n2 > 1:
            print("  => Without streaming, HF downloads many/all split files before applying split slice.")
    elif t1 >= 0:
        print("  streaming=False: (run failed or incomplete; typically downloads all split files.)")
    if t1 >= 0 and t2 >= 0:
        print("  => Recommendation: use streaming=True to get first events without downloading the full split.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
