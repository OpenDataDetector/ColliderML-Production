#!/usr/bin/env python3
"""
Upload unified ColliderML dataset to HuggingFace Hub (Simplified Version).

This script uploads parquet files directly to HuggingFace without re-sharding.
Files are organized as data/{config_name}/*.parquet in the repo.

Usage:
    python upload_to_hf_unified_simple.py unified_dataset_config.yaml

    # Dry run to see what would be uploaded
    python upload_to_hf_unified_simple.py unified_dataset_config.yaml --dry-run

    # Upload specific configs only
    python upload_to_hf_unified_simple.py unified_dataset_config.yaml --configs ttbar_pu0_particles

Features:
    - Direct upload: fast, no memory-intensive re-sharding
    - Streaming validation: checks event IDs without loading full dataset
    - Progressive uploads: add configs incrementally
    - Automatic README.md config generation for HF dataset viewer
"""

import argparse
import logging
import os
import re
import sys
import tempfile
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set HuggingFace cache to LOCAL disk to avoid lock contention on network FS
# Using /tmp for local disk instead of shared CFS
HF_CACHE_DIR = "/tmp/hf_cache_upload"
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_DIR, "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")

try:
    from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
    from datasets import get_dataset_config_names
    import pyarrow.parquet as pq
    HF_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: Required libraries not available: {e}")
    print("Install with: pip install datasets huggingface_hub pyarrow")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Management
# ============================================================================

def load_config(config_path: str) -> Dict:
    """Load and validate YAML configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required fields
    required = ['huggingface', 'data', 'object_types', 'campaigns']
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required config section: {field}")

    if 'repo_id' not in config['huggingface']:
        raise ValueError("Missing huggingface.repo_id in config")

    logger.info(f"Loaded config from: {config_path}")
    return config


def build_config_list(config: Dict) -> List[Dict]:
    """
    Build list of all configs to upload based on yaml configuration.

    Returns list of dicts with keys: config_name, path, dataset_name, pileup, object_type
    """
    configs = []
    base_dir = Path(config['data']['base_dir'])
    version = config['data']['version']
    format_subdir = config['data']['format_subdir']

    for campaign in config['campaigns']:
        campaign_name = campaign['campaign_name']
        pileup_label = campaign['pileup_label']

        for dataset in campaign['datasets']:
            if not dataset.get('enabled', True):
                continue

            dataset_name = dataset['name']

            for object_type in config['object_types']:
                config_name = f"{dataset_name}_{pileup_label}_{object_type}"

                # Construct path: base/campaign/dataset/version/parquet/truth|reco/object_type/
                if object_type == "particles":
                    subdir = "truth/particles"
                else:
                    subdir = f"reco/{object_type}"

                data_path = base_dir / campaign_name / dataset_name / version / format_subdir / subdir

                # Check if path exists and has parquet files
                if not data_path.exists():
                    logger.warning(f"Path does not exist: {data_path}")
                    continue

                parquet_files = list(data_path.glob("*.parquet"))
                if not parquet_files:
                    logger.warning(f"No parquet files in: {data_path}")
                    continue

                configs.append({
                    'config_name': config_name,
                    'path': data_path,
                    'dataset_name': dataset_name,
                    'campaign_name': campaign_name,
                    'pileup': campaign['pileup'],
                    'pileup_label': pileup_label,
                    'object_type': object_type,
                    'description': dataset.get('description', '')
                })

    logger.info(f"Built list of {len(configs)} configs")
    return configs


# ============================================================================
# Event ID Validation (Streaming)
# ============================================================================

def sort_parquet_files_numerically(parquet_files: List[Path]) -> List[Path]:
    """Sort parquet files by event range (e.g., events0-999.parquet)."""
    def extract_event_start(filepath: Path) -> int:
        match = re.search(r'events(\d+)-', filepath.name)
        return int(match.group(1)) if match else 0

    return sorted(parquet_files, key=extract_event_start)


def validate_event_ids_streaming(parquet_files: List[Path], config_name: str, object_type: str) -> bool:
    """
    Validate event IDs by streaming only the event_id column from parquet files.

    This avoids loading the entire dataset into memory.

    Validation rules:
    - Truth particles: All event IDs must be present and continuous (every event has particles)
    - Reco objects: Event IDs only need to be unique (events can have 0 tracks/hits)

    Returns True if valid, False otherwise.
    """
    logger.info(f"  Validating event IDs (streaming mode)...")

    try:
        all_event_ids = []

        # Stream through each parquet file, reading only event_id column
        for pf in parquet_files:
            parquet_file = pq.ParquetFile(pf)
            # Read only event_id column
            table = parquet_file.read(columns=['event_id'])
            event_ids = table['event_id'].to_pylist()
            all_event_ids.extend(event_ids)

        n_events = len(all_event_ids)
        logger.info(f"  Loaded {n_events} event IDs from {len(parquet_files)} files")

        # Check uniqueness
        unique_ids = set(all_event_ids)
        if len(unique_ids) != n_events:
            logger.error(f"  ✗ Found {n_events - len(unique_ids)} duplicate event IDs!")
            return False

        # Check range
        min_id = min(all_event_ids)
        max_id = max(all_event_ids)

        # For truth particles: require completeness and continuity
        # For reco objects: allow missing events (means 0 objects for that event)
        is_truth = (object_type == "particles")

        if is_truth:
            # Truth particles: all events must be present
            if min_id != 0:
                logger.warning(f"  ⚠ Event IDs start at {min_id}, not 0")

            if max_id != n_events - 1:
                logger.warning(f"  ⚠ Event IDs end at {max_id}, expected {n_events - 1}")

            # Check for gaps
            expected_ids = set(range(min_id, max_id + 1))
            missing_ids = expected_ids - unique_ids

            if missing_ids:
                missing_sample = sorted(missing_ids)[:10]
                logger.error(f"  ✗ Found {len(missing_ids)} missing event IDs!")
                logger.error(f"    Sample missing: {missing_sample}...")
                return False

            logger.info(f"  ✓ Event IDs validated: {n_events} unique, continuous from {min_id} to {max_id}")
        else:
            # Reco objects: missing events are OK (means 0 objects)
            n_missing = (max_id - min_id + 1) - n_events
            if n_missing > 0:
                logger.info(f"  ✓ Event IDs validated: {n_events} unique from {min_id} to {max_id}")
                logger.info(f"    {n_missing} events absent (likely 0 {object_type} for those events)")
            else:
                logger.info(f"  ✓ Event IDs validated: {n_events} unique, continuous from {min_id} to {max_id}")

        return True

    except Exception as e:
        logger.error(f"  ✗ Failed to validate event IDs: {e}")
        return False


# ============================================================================
# Direct Upload to HuggingFace
# ============================================================================

def upload_config_direct(
    config_info: Dict,
    repo_id: str,
    token: str,
    num_workers: int = 5,
    skip_validation: bool = False,
    dry_run: bool = False
) -> Tuple[bool, float]:
    """
    Upload a single config directly using HF API (no re-sharding).

    Files are uploaded to: data/{config_name}/ folder in the repo.
    Uses streaming validation to avoid memory issues.

    Returns (success: bool, elapsed_time: float).
    """
    config_name = config_info['config_name']
    data_path = config_info['path']

    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Uploading: {config_name}")
    logger.info(f"  Data path: {data_path}")

    parquet_files = list(data_path.glob("*.parquet"))
    total_size = sum(f.stat().st_size for f in parquet_files)
    logger.info(f"  Found {len(parquet_files)} parquet files ({total_size / 1e9:.2f} GB)")

    if dry_run:
        logger.info(f"  Would upload to: data/{config_name}/")
        return True, 0.0

    start_time = time.time()

    try:
        # Sort files numerically
        sorted_files = sort_parquet_files_numerically(parquet_files)

        # Validate event IDs using streaming (memory-efficient)
        if not skip_validation:
            object_type = config_info['object_type']
            if not validate_event_ids_streaming(sorted_files, config_name, object_type):
                logger.error(f"  ✗ Validation failed - aborting upload")
                return False, time.time() - start_time
        else:
            logger.info(f"  Skipping validation (--skip-validation)")

        # Prepare upload operations with HF naming convention
        # Files will be: data/{config_name}/train-00000-of-00100.parquet
        logger.info(f"  Preparing upload operations...")
        n_files = len(sorted_files)
        operations = []

        for i, src_file in enumerate(sorted_files):
            dst_name = f"train-{i:05d}-of-{n_files:05d}.parquet"
            operations.append(
                CommitOperationAdd(
                    path_in_repo=f"data/{config_name}/{dst_name}",
                    path_or_fileobj=str(src_file)
                )
            )

        # Upload with multi-threaded API
        logger.info(f"  Uploading to HuggingFace ({num_workers} threads)...")
        api = HfApi(token=token)
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Add config: {config_name}",
            num_threads=num_workers
        )

        elapsed = time.time() - start_time
        logger.info(f"✓ Successfully uploaded: {config_name} ({elapsed:.1f}s)")
        return True, elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"✗ Failed to upload {config_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, elapsed


# ============================================================================
# README Configuration Management
# ============================================================================

def update_readme_with_all_configs(
    repo_id: str,
    local_readme_path: Optional[Path],
    token: str,
    dry_run: bool = False
) -> bool:
    """
    Update README.md with ALL configs currently in the repository.

    This merges:
    - Markdown content from local README file (if provided)
    - YAML frontmatter for ALL configs currently on HuggingFace

    This ensures the README is always complete and up-to-date.

    Returns True if successful.
    """
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Updating README.md with all configs")

    if dry_run:
        logger.info("  Would query HF for all configs and merge with local README")
        return True

    try:
        api = HfApi(token=token)

        # Get ALL configs by looking at data/ directory structure
        # (not from README, since README may be out of date!)
        try:
            from huggingface_hub.hf_api import RepoFolder
            repo_tree = list(api.list_repo_tree(
                repo_id=repo_id,
                repo_type="dataset",
                path_in_repo="data",
                recursive=False
            ))
            all_config_names = []
            for item in repo_tree:
                # Check if it's a folder (RepoFolder type)
                if isinstance(item, RepoFolder):
                    config_name = item.path.split('/')[-1]
                    all_config_names.append(config_name)
            logger.info(f"  Found {len(all_config_names)} total configs in data/ folder")
        except Exception as e:
            logger.warning(f"  Could not fetch configs from data/ folder: {e}")
            all_config_names = []

        # Get markdown content AND existing frontmatter from local README file
        markdown_content = ""
        existing_frontmatter = {}

        if local_readme_path and local_readme_path.exists():
            logger.info(f"  Using content from: {local_readme_path}")
            with open(local_readme_path, 'r') as f:
                local_content = f.read()

            # Extract existing frontmatter if present
            yaml_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', local_content, re.DOTALL)
            if yaml_match:
                # Parse existing YAML frontmatter
                try:
                    existing_frontmatter = yaml.safe_load(yaml_match.group(1))
                    if existing_frontmatter is None:
                        existing_frontmatter = {}
                    logger.info(f"  Parsed existing frontmatter with {len(existing_frontmatter)} keys")
                except Exception as e:
                    logger.warning(f"  Could not parse existing frontmatter: {e}")
                    existing_frontmatter = {}

                # Get markdown content after frontmatter
                markdown_content = local_content[yaml_match.end():]
            else:
                markdown_content = local_content
        else:
            logger.warning("  No local README file found, using empty markdown content")

        # Build config list for frontmatter
        configs = []
        for config_name in sorted(all_config_names):
            configs.append({
                'config_name': config_name,
                'data_files': [{
                    'split': 'train',
                    'path': f"data/{config_name}/*.parquet"
                }]
            })

        # MERGE: Update configs in existing frontmatter, keep everything else
        yaml_data = existing_frontmatter.copy()
        yaml_data['configs'] = configs
        logger.info(f"  Merged configs with existing frontmatter ({len(yaml_data)} total keys)")

        # Build complete README
        new_yaml = yaml.dump(yaml_data, default_flow_style=False, sort_keys=False, allow_unicode=True)
        new_readme = f"---\n{new_yaml}---\n{markdown_content}"

        # Upload README
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(new_readme)
            tmp_path = f.name

        try:
            api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Update README: {len(configs)} config(s)"
            )
            logger.info(f"✓ Updated README.md with {len(configs)} total config(s)")
        finally:
            os.unlink(tmp_path)

        return True

    except Exception as e:
        logger.error(f"✗ Failed to update README: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


# ============================================================================
# Repository Management
# ============================================================================

def create_repo_if_needed(repo_id: str, config: Dict, token: str, dry_run: bool = False) -> bool:
    """Create repository if it doesn't exist."""
    if dry_run:
        logger.info(f"[DRY RUN] Would create/verify repo: {repo_id}")
        return True

    try:
        api = HfApi(token=token)
        try:
            api.repo_info(repo_id=repo_id, repo_type="dataset")
            logger.info(f"✓ Repository exists: {repo_id}")
        except Exception:
            logger.info(f"Creating repository: {repo_id}")
            api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=config['huggingface'].get('private', False)
            )
            logger.info(f"✓ Repository created: {repo_id}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to create/verify repository: {e}")
        return False


def clean_repo(repo_id: str, token: str, dry_run: bool = False) -> bool:
    """
    Delete all parquet files from the repository.
    Keeps README.md and .gitattributes.
    """
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Cleaning repository: {repo_id}")

    try:
        api = HfApi(token=token)
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

        files_to_delete = [
            f for f in repo_files
            if f.endswith('.parquet') and f not in ['README.md', '.gitattributes']
        ]

        if not files_to_delete:
            logger.info("  No parquet files to delete")
            return True

        logger.info(f"  Found {len(files_to_delete)} parquet files to delete")

        if dry_run:
            return True

        operations = [CommitOperationDelete(path_in_repo=f) for f in files_to_delete]

        logger.info(f"  Deleting {len(operations)} files...")
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Clean repo: delete {len(files_to_delete)} parquet files"
        )

        logger.info(f"✓ Successfully cleaned repository")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to clean repository: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def get_existing_configs(repo_id: str, token: str) -> List[str]:
    """Get list of existing configs by checking data/ directory structure."""
    try:
        from huggingface_hub.hf_api import RepoFolder
        api = HfApi(token=token)
        repo_tree = list(api.list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo="data",
            recursive=False
        ))
        existing = []
        for item in repo_tree:
            if isinstance(item, RepoFolder):
                config_name = item.path.split('/')[-1]
                existing.append(config_name)
        logger.info(f"Found {len(existing)} existing configs")
        return existing
    except Exception:
        logger.info("Could not fetch existing configs (repo may not exist yet)")
        return []


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Upload ColliderML dataset to HuggingFace Hub (Simplified)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload all enabled configs
  python upload_to_hf_unified_simple.py unified_dataset_config.yaml

  # Dry run
  python upload_to_hf_unified_simple.py unified_dataset_config.yaml --dry-run

  # Upload specific configs
  python upload_to_hf_unified_simple.py unified_dataset_config.yaml --configs ttbar_pu0_particles

  # Skip validation for large datasets
  python upload_to_hf_unified_simple.py unified_dataset_config.yaml --skip-validation
        """
    )

    parser.add_argument('config', help='Path to YAML configuration file')
    parser.add_argument('--dry-run', action='store_true', help='Preview without uploading')
    parser.add_argument('--configs', nargs='+', help='Upload only these specific configs')
    parser.add_argument('--start-from', help='Start from this config (skip earlier ones)')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip configs that already exist (default: true)')
    parser.add_argument('--no-skip-existing', dest='skip_existing', action='store_false',
                        help='Re-upload existing configs')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip event ID validation (faster but risky)')
    parser.add_argument('--clean-repo', action='store_true',
                        help='Delete all parquet files before uploading')
    parser.add_argument('--num-workers', type=int, default=5,
                        help='Number of concurrent upload threads (default: 5)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Load config
        config = load_config(args.config)

        # Get HuggingFace token
        token_env = config['huggingface'].get('token_env', 'HF_TOKEN')
        token = os.environ.get(token_env)
        if not token and not args.dry_run:
            logger.error(f"HuggingFace token not found: {token_env}")
            logger.error("Set with: export HF_TOKEN=<your_token>")
            sys.exit(1)

        repo_id = config['huggingface']['repo_id']

        # Create/verify repo
        if not create_repo_if_needed(repo_id, config, token, args.dry_run):
            sys.exit(1)

        # Clean repo if requested
        if args.clean_repo:
            if not clean_repo(repo_id, token, args.dry_run):
                sys.exit(1)

        # Build config list
        all_configs = build_config_list(config)
        if not all_configs:
            logger.warning("No configs found to upload")
            sys.exit(0)

        # Filter configs
        configs_to_upload = all_configs

        if args.configs:
            configs_to_upload = [c for c in configs_to_upload if c['config_name'] in args.configs]
            logger.info(f"Filtered to {len(configs_to_upload)} specific configs")

        if args.start_from:
            idx = next((i for i, c in enumerate(configs_to_upload) if c['config_name'] == args.start_from), None)
            if idx is not None:
                configs_to_upload = configs_to_upload[idx:]
                logger.info(f"Starting from {args.start_from}: {len(configs_to_upload)} remaining")

        if args.skip_existing and not args.dry_run:
            existing = get_existing_configs(repo_id, token)
            before = len(configs_to_upload)
            configs_to_upload = [c for c in configs_to_upload if c['config_name'] not in existing]
            if before > len(configs_to_upload):
                logger.info(f"Skipping {before - len(configs_to_upload)} existing configs")

        # Upload configs if any need uploading
        success_count = 0
        failed_count = 0
        total_time = 0.0
        successful_configs = []

        if configs_to_upload:
            # Print summary
            logger.info("=" * 80)
            logger.info(f"Configs to upload: {len(configs_to_upload)}")
            for c in configs_to_upload:
                logger.info(f"  - {c['config_name']}")
            logger.info("=" * 80)

            # Upload each config
            for i, config_info in enumerate(configs_to_upload, 1):
                logger.info(f"\n[{i}/{len(configs_to_upload)}] {config_info['config_name']}...")

                success, elapsed = upload_config_direct(
                    config_info,
                    repo_id,
                    token,
                    num_workers=args.num_workers,
                    skip_validation=args.skip_validation,
                    dry_run=args.dry_run
                )

                if success:
                    success_count += 1
                    total_time += elapsed
                    successful_configs.append({'config_name': config_info['config_name']})
                else:
                    failed_count += 1
        else:
            logger.info("No configs to upload (all exist), but will update README")

        # Update README with ALL configs (existing + newly uploaded)
        # This runs even if no new uploads happened
        if not args.dry_run:
            logger.info(f"\nUpdating README.md with all configs...")
            readme_path_str = config['huggingface'].get('readme_path')
            readme_path = Path(readme_path_str) if readme_path_str else None
            update_readme_with_all_configs(repo_id, readme_path, token, args.dry_run)

        # Summary
        logger.info("\n" + "=" * 80)
        if args.dry_run:
            logger.info("✓ Dry run complete")
        else:
            logger.info(f"✓ Upload complete!")
            logger.info(f"  Successful: {success_count}")
            logger.info(f"  Failed: {failed_count}")
            logger.info(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
            if success_count > 1:
                logger.info(f"  Average: {total_time/success_count:.1f}s per config")
            logger.info(f"\nView at: https://huggingface.co/datasets/{repo_id}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
