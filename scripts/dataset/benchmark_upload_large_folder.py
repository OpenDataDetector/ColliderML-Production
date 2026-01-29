"""
Benchmark script to test upload_large_folder performance.

This script tests uploading ttbar_pu200_particles (~800GB, 1000 files)
using the upload_large_folder method with multi-threaded hashing.

Expected improvement: 17-19 min (serial) → 1-2 min (parallel)

Usage:
    python benchmark_upload_large_folder.py [--dry-run]
"""

import argparse
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import List

# Set HuggingFace cache to LOCAL disk (not network FS) to avoid lock contention
# Using /tmp which is local to the compute node, not shared CFS
HF_CACHE_DIR = "/tmp/hf_cache_upload"
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_DIR, "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")

# Enable verbose logging for HuggingFace Hub
os.environ["HF_HUB_VERBOSITY"] = "info"

try:
    from huggingface_hub import HfApi
except ImportError as e:
    print(f"ERROR: Required libraries not available: {e}")
    print("Install with: pip install huggingface_hub")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

REPO_ID = "CERN/ColliderML-Release-1"
CONFIG_NAME = "ttbar_pu200_particles"

# Path to source data
DATA_BASE = Path("/global/cfs/cdirs/m4958/data/ColliderML/simulation")
SOURCE_PATH = DATA_BASE / "full_pileup" / "ttbar" / "v1" / "parquet" / "truth" / "particles"

# Staging directory for upload_large_folder (use /tmp for local disk, not CFS)
STAGING_BASE = Path("/tmp/upload_staging")
STAGING_PATH = STAGING_BASE / "data" / CONFIG_NAME


# ============================================================================
# Helper Functions
# ============================================================================

def sort_parquet_files_numerically(parquet_files: List[Path]) -> List[Path]:
    """Sort parquet files by event range (e.g., events0-999.parquet)."""
    def extract_event_start(filepath: Path) -> int:
        match = re.search(r'events(\d+)-', filepath.name)
        return int(match.group(1)) if match else 0

    return sorted(parquet_files, key=extract_event_start)


def create_staging_directory(source_path: Path, staging_path: Path, dry_run: bool = False) -> int:
    """
    Create staging directory with symlinks to source files.

    Files are renamed to HF convention: train-00000-of-01000.parquet

    Returns number of files staged.
    """
    logger.info(f"Creating staging directory at: {staging_path}")

    # Find all parquet files in source
    parquet_files = list(source_path.glob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in: {source_path}")

    # Sort files numerically
    sorted_files = sort_parquet_files_numerically(parquet_files)
    n_files = len(sorted_files)

    total_size = sum(f.stat().st_size for f in sorted_files)
    logger.info(f"Found {n_files} parquet files ({total_size / 1e9:.2f} GB)")

    if dry_run:
        logger.info("[DRY RUN] Would create staging directory with symlinks")
        return n_files

    # Create staging directory
    staging_path.mkdir(parents=True, exist_ok=True)

    # Create symlinks with HF naming convention
    logger.info(f"Creating {n_files} symlinks...")
    for i, src_file in enumerate(sorted_files):
        dst_name = f"train-{i:05d}-of-{n_files:05d}.parquet"
        dst_path = staging_path / dst_name

        # Create relative symlink
        # (use absolute paths for simplicity on shared filesystem)
        if dst_path.exists():
            dst_path.unlink()
        dst_path.symlink_to(src_file.absolute())

        if (i + 1) % 100 == 0:
            logger.info(f"  Created {i + 1}/{n_files} symlinks")

    logger.info(f"✓ Staging directory ready: {staging_path}")
    return n_files


def cleanup_staging_directory(staging_base: Path, dry_run: bool = False):
    """Remove staging directory and all symlinks."""
    if dry_run:
        logger.info(f"[DRY RUN] Would remove staging directory: {staging_base}")
        return

    if staging_base.exists():
        logger.info(f"Cleaning up staging directory: {staging_base}")
        shutil.rmtree(staging_base)
        logger.info("✓ Staging directory removed")


def benchmark_upload_large_folder(
    repo_id: str,
    staging_base: Path,
    token: str,
    dry_run: bool = False
) -> float:
    """
    Upload using upload_large_folder and measure time.

    Returns elapsed time in seconds.
    """
    logger.info("=" * 80)
    logger.info(f"Uploading with upload_large_folder: {repo_id}")
    logger.info(f"Staging directory: {staging_base}")
    logger.info("=" * 80)

    if dry_run:
        logger.info("[DRY RUN] Would call upload_large_folder")
        return 0.0

    start_time = time.time()

    try:
        api = HfApi(token=token)

        # Upload entire staging directory
        # This will upload everything under data/{config_name}/
        num_workers = 32  # Reduced from default (cpu_count // 2 = 128) to avoid lock contention
        logger.info(f"Starting upload_large_folder (multi-threaded)...")
        logger.info(f"  Using {num_workers} worker threads (reduced to avoid lock contention on network FS)")
        logger.info(f"  Status reports will print every 10 seconds")

        api.upload_large_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(staging_base),
            print_report=True,           # Print progress reports
            print_report_every=10,       # Print every 10 seconds (instead of default 60)
            num_workers=32,              # Limit to 32 workers (default was 128, causing lock contention)
            # allow_patterns can be used to filter files if needed
        )

        elapsed = time.time() - start_time
        logger.info(f"✓ Upload complete!")
        logger.info(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

        return elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"✗ Upload failed after {elapsed:.1f}s: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return elapsed


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark upload_large_folder performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script tests uploading ttbar_pu200_particles (~800GB) using upload_large_folder.
Current approach takes ~17-19 minutes. Expected: 1-2 minutes with parallel hashing.

Examples:
  # Dry run (no actual upload)
  python benchmark_upload_large_folder.py --dry-run

  # Real upload
  python benchmark_upload_large_folder.py

  # Keep staging directory for inspection
  python benchmark_upload_large_folder.py --keep-staging
        """
    )

    parser.add_argument('--dry-run', action='store_true',
                       help='Preview without uploading')
    parser.add_argument('--keep-staging', action='store_true',
                       help='Keep staging directory after upload (for debugging)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Get HuggingFace token
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    if not token and not args.dry_run:
        logger.error("HuggingFace token not found!")
        logger.error("Set with: export HF_TOKEN=<your_token>")
        sys.exit(1)

    try:
        # Validate source data exists
        if not SOURCE_PATH.exists():
            logger.error(f"Source data not found: {SOURCE_PATH}")
            sys.exit(1)

        logger.info("=" * 80)
        logger.info("BENCHMARK: upload_large_folder")
        logger.info("=" * 80)
        logger.info(f"Config: {CONFIG_NAME}")
        logger.info(f"Source: {SOURCE_PATH}")
        logger.info(f"Staging: {STAGING_PATH}")
        logger.info(f"Repo: {REPO_ID}")
        logger.info(f"CPU cores: {os.cpu_count()} (will use {os.cpu_count() // 2} for hashing)")
        logger.info("=" * 80)

        # Step 1: Create staging directory with symlinks
        logger.info("\n[Step 1/3] Creating staging directory...")
        staging_start = time.time()
        n_files = create_staging_directory(SOURCE_PATH, STAGING_PATH, args.dry_run)
        staging_elapsed = time.time() - staging_start
        logger.info(f"✓ Staging complete: {staging_elapsed:.1f}s")

        # Step 2: Upload with upload_large_folder
        logger.info("\n[Step 2/3] Uploading with upload_large_folder...")
        upload_elapsed = benchmark_upload_large_folder(
            REPO_ID,
            STAGING_BASE,
            token,
            args.dry_run
        )

        # Step 3: Cleanup
        if not args.keep_staging:
            logger.info("\n[Step 3/3] Cleaning up staging directory...")
            cleanup_staging_directory(STAGING_BASE, args.dry_run)
        else:
            logger.info(f"\n[Step 3/3] Keeping staging directory: {STAGING_BASE}")

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * 80)
        logger.info(f"Config: {CONFIG_NAME}")
        logger.info(f"Files: {n_files}")
        logger.info(f"Staging time: {staging_elapsed:.1f}s ({staging_elapsed/60:.1f} min)")
        logger.info(f"Upload time: {upload_elapsed:.1f}s ({upload_elapsed/60:.1f} min)")
        logger.info(f"Total time: {staging_elapsed + upload_elapsed:.1f}s ({(staging_elapsed + upload_elapsed)/60:.1f} min)")
        logger.info("")
        logger.info("Comparison with current approach:")
        logger.info("  Current (serial): ~17-19 minutes")
        logger.info(f"  upload_large_folder: {upload_elapsed/60:.1f} minutes")
        if upload_elapsed > 0 and not args.dry_run:
            speedup = (17.5 * 60) / upload_elapsed  # Use 17.5 min as baseline
            logger.info(f"  Speedup: {speedup:.1f}x faster")
        logger.info("=" * 80)

        if not args.dry_run:
            logger.info(f"\nView at: https://huggingface.co/datasets/{REPO_ID}")

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        if not args.keep_staging:
            logger.info("Cleaning up staging directory...")
            cleanup_staging_directory(STAGING_BASE, dry_run=False)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        if not args.keep_staging:
            logger.info("Cleaning up staging directory...")
            cleanup_staging_directory(STAGING_BASE, dry_run=False)
        sys.exit(1)


if __name__ == "__main__":
    main()
