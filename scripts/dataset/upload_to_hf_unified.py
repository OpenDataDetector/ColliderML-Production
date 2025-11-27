#!/usr/bin/env python3
"""
Upload unified ColliderML dataset to HuggingFace Hub.

This script uploads all physics channels and object types to a single unified
HuggingFace dataset using individual configs for each combination.

Usage:
    python scripts/dataset/upload_to_hf_unified.py scripts/dataset/unified_dataset_config.yaml

    # Dry run to see what would be uploaded
    python scripts/dataset/upload_to_hf_unified.py scripts/dataset/unified_dataset_config.yaml --dry-run

    # Upload specific configs only
    python scripts/dataset/upload_to_hf_unified.py scripts/dataset/unified_dataset_config.yaml --configs ttbar_pu0_particles ttbar_pu0_tracks

Features:
    - Progressive uploads: add configs one at a time without affecting existing data
    - Uses datasets library for automatic Parquet handling
    - Automatic README.md generation and updates
    - Skip existing configs to avoid re-uploading
"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from datasets import load_dataset, get_dataset_config_names
    from huggingface_hub import HfApi
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("ERROR: huggingface_hub and datasets libraries required.")
    print("Install with: pip install datasets huggingface_hub")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from: {config_path}")
    return config


def validate_config(config: Dict) -> None:
    """Validate configuration has required fields."""
    required = ['huggingface', 'data', 'object_types', 'campaigns']
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required config section: {field}")

    if 'repo_id' not in config['huggingface']:
        raise ValueError("Missing huggingface.repo_id in config")

    logger.info("✓ Config validation passed")


def get_data_path(
    base_dir: Path,
    campaign_name: str,
    dataset_name: str,
    version: str,
    format_subdir: str,
    object_type: str
) -> Path:
    """Construct path to parquet files for a given object type."""
    # Determine subdirectory (truth vs reco)
    if object_type == "particles":
        subdir = "truth/particles"
    else:
        subdir = f"reco/{object_type}"

    # Full path
    data_path = base_dir / campaign_name / dataset_name / version / format_subdir / subdir

    return data_path


def check_path_exists(path: Path, config_name: str) -> bool:
    """Check if data path exists and contains parquet files."""
    if not path.exists():
        logger.warning(f"Path does not exist for {config_name}: {path}")
        return False

    parquet_files = list(path.glob("*.parquet"))
    if not parquet_files:
        logger.warning(f"No parquet files found for {config_name} in: {path}")
        return False

    logger.debug(f"Found {len(parquet_files)} parquet files for {config_name}")
    return True


def get_existing_configs(repo_id: str, token: str) -> List[str]:
    """Get list of existing configs in the HuggingFace dataset."""
    try:
        existing = get_dataset_config_names(repo_id, token=token)
        logger.info(f"Found {len(existing)} existing configs in {repo_id}")
        return existing
    except Exception as e:
        logger.info(f"Could not fetch existing configs (repo may not exist yet): {e}")
        return []


def build_config_list(config: Dict) -> List[Dict]:
    """
    Build list of all configs to upload based on yaml configuration.

    Returns list of dicts with keys: config_name, path, dataset_name, pileup, object_type
    """
    configs_to_upload = []

    base_dir = Path(config['data']['base_dir'])
    version = config['data']['version']
    format_subdir = config['data']['format_subdir']
    object_types = config['object_types']

    for campaign in config['campaigns']:
        campaign_name = campaign['campaign_name']
        pileup_label = campaign['pileup_label']
        pileup = campaign['pileup']

        for dataset in campaign['datasets']:
            dataset_name = dataset['name']

            # Skip if not enabled
            if not dataset.get('enabled', True):
                logger.info(f"Skipping disabled dataset: {dataset_name} (campaign: {campaign_name})")
                continue

            # Create configs for each object type
            for object_type in object_types:
                # Config naming: {dataset}_pu{pileup}_{object_type}
                # But use pileup_label instead of raw number for flexibility
                config_name = f"{dataset_name}_{pileup_label}_{object_type}"

                # Get path to data
                data_path = get_data_path(
                    base_dir, campaign_name, dataset_name,
                    version, format_subdir, object_type
                )

                # Check if path exists
                if not check_path_exists(data_path, config_name):
                    continue

                configs_to_upload.append({
                    'config_name': config_name,
                    'path': data_path,
                    'dataset_name': dataset_name,
                    'campaign_name': campaign_name,
                    'pileup': pileup,
                    'pileup_label': pileup_label,
                    'object_type': object_type,
                    'description': dataset.get('description', '')
                })

    logger.info(f"Built list of {len(configs_to_upload)} configs to upload")
    return configs_to_upload


def upload_config(
    config_info: Dict,
    repo_id: str,
    max_shard_size: str,
    private: bool,
    dry_run: bool = False
) -> bool:
    """
    Upload a single config to HuggingFace.

    Returns True if successful, False otherwise.
    """
    config_name = config_info['config_name']
    data_path = config_info['path']

    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Uploading config: {config_name}")
    logger.info(f"  Data path: {data_path}")

    if dry_run:
        logger.info(f"  Would upload {len(list(data_path.glob('*.parquet')))} parquet files")
        return True

    try:
        # Load dataset from parquet files
        logger.info(f"  Loading parquet files...")
        dataset = load_dataset(
            "parquet",
            data_files=str(data_path / "*.parquet"),
            split="train"
        )

        logger.info(f"  Loaded {len(dataset)} events")

        # Push to hub
        logger.info(f"  Pushing to HuggingFace...")
        dataset.push_to_hub(
            repo_id,
            config_name=config_name,
            max_shard_size=max_shard_size,
            private=private
        )

        logger.info(f"✓ Successfully uploaded: {config_name}")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to upload {config_name}: {e}")
        return False


def create_or_update_repo(repo_id: str, config: Dict, token: str, dry_run: bool = False) -> bool:
    """
    Create repository if it doesn't exist, or verify it exists.

    Returns True if repo exists/was created, False otherwise.
    """
    if dry_run:
        logger.info(f"[DRY RUN] Would create/verify repo: {repo_id}")
        return True

    try:
        api = HfApi(token=token)

        # Check if repo exists
        try:
            api.repo_info(repo_id=repo_id, repo_type="dataset")
            logger.info(f"✓ Repository exists: {repo_id}")
            return True
        except Exception:
            # Repo doesn't exist, create it
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


def main():
    parser = argparse.ArgumentParser(
        description="Upload unified ColliderML dataset to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload all enabled configs
  python upload_to_hf_unified.py unified_dataset_config.yaml

  # Dry run to see what would be uploaded
  python upload_to_hf_unified.py unified_dataset_config.yaml --dry-run

  # Upload specific configs only
  python upload_to_hf_unified.py unified_dataset_config.yaml --configs ttbar_pu0_particles ttbar_pu0_tracks

  # Continue from a specific config (skip earlier ones)
  python upload_to_hf_unified.py unified_dataset_config.yaml --start-from ttbar_pu200_particles
        """
    )

    parser.add_argument(
        'config',
        type=str,
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what would be done without making changes'
    )

    parser.add_argument(
        '--configs',
        nargs='+',
        help='Upload only these specific configs (by name)'
    )

    parser.add_argument(
        '--start-from',
        type=str,
        help='Start uploading from this config (skip all before it)'
    )

    parser.add_argument(
        '--skip-existing',
        action='store_true',
        default=True,
        help='Skip configs that already exist on HuggingFace (default: true)'
    )

    parser.add_argument(
        '--no-skip-existing',
        dest='skip_existing',
        action='store_false',
        help='Re-upload configs even if they already exist'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Load and validate config
        config = load_config(args.config)
        validate_config(config)

        # Get HuggingFace token (not required for dry-run)
        token_env = config['huggingface'].get('token_env', 'HF_TOKEN')
        token = os.environ.get(token_env)
        if not token and not args.dry_run:
            logger.error(f"HuggingFace token not found in environment variable: {token_env}")
            logger.error("Set token with: export HF_TOKEN=<your_token>")
            logger.error("Get token from: https://huggingface.co/settings/tokens")
            sys.exit(1)
        elif not token and args.dry_run:
            logger.info(f"No HF token found, but running in dry-run mode - continuing...")

        repo_id = config['huggingface']['repo_id']
        max_shard_size = config.get('upload', {}).get('max_shard_size', '500MB')
        private = config['huggingface'].get('private', False)
        skip_existing = args.skip_existing

        # Create or verify repository
        if not create_or_update_repo(repo_id, config, token, args.dry_run):
            logger.error("Failed to create/verify repository. Exiting.")
            sys.exit(1)

        # Get existing configs if skip_existing is enabled
        existing_configs = []
        if skip_existing and not args.dry_run:
            existing_configs = get_existing_configs(repo_id, token)

        # Build list of configs to upload
        all_configs = build_config_list(config)

        if not all_configs:
            logger.warning("No configs found to upload. Check your configuration.")
            sys.exit(0)

        # Filter configs based on command line arguments
        configs_to_upload = all_configs

        # Filter: specific configs only
        if args.configs:
            configs_to_upload = [c for c in configs_to_upload if c['config_name'] in args.configs]
            logger.info(f"Filtered to {len(configs_to_upload)} specific configs")

        # Filter: start from specific config
        if args.start_from:
            found_start = False
            filtered = []
            for c in configs_to_upload:
                if c['config_name'] == args.start_from:
                    found_start = True
                if found_start:
                    filtered.append(c)
            configs_to_upload = filtered
            logger.info(f"Starting from {args.start_from}: {len(configs_to_upload)} configs remaining")

        # Filter: skip existing
        if skip_existing and existing_configs:
            before_count = len(configs_to_upload)
            configs_to_upload = [c for c in configs_to_upload if c['config_name'] not in existing_configs]
            skipped = before_count - len(configs_to_upload)
            if skipped > 0:
                logger.info(f"Skipping {skipped} existing configs")

        if not configs_to_upload:
            logger.info("No configs to upload after filtering. All done!")
            sys.exit(0)

        # Print summary
        logger.info("=" * 80)
        logger.info(f"Configs to upload: {len(configs_to_upload)}")
        for c in configs_to_upload:
            logger.info(f"  - {c['config_name']}")
        logger.info("=" * 80)

        if args.dry_run:
            logger.info("[DRY RUN] No changes will be made")

        # Upload each config
        success_count = 0
        failed_count = 0

        for i, config_info in enumerate(configs_to_upload, 1):
            logger.info(f"\n[{i}/{len(configs_to_upload)}] Processing {config_info['config_name']}...")

            success = upload_config(
                config_info,
                repo_id,
                max_shard_size,
                private,
                args.dry_run
            )

            if success:
                success_count += 1
            else:
                failed_count += 1

        # Final summary
        logger.info("\n" + "=" * 80)
        if args.dry_run:
            logger.info("✓ Dry run complete")
        else:
            logger.info(f"✓ Upload complete!")
            logger.info(f"  Successfully uploaded: {success_count}")
            logger.info(f"  Failed: {failed_count}")
            logger.info(f"\nView dataset at: https://huggingface.co/datasets/{repo_id}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
