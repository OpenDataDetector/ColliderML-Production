#!/usr/bin/env python3
"""
Upload ColliderML datasets to HuggingFace with NERSC hosting.

This tool uploads Parquet dataset metadata to HuggingFace while keeping the actual
data files hosted on NERSC infrastructure for HTTP streaming.

Usage:
    python upload_to_huggingface.py --config configs/hf_upload_example.yaml [--dry-run]

Features:
    - Progressive updates: add new configs/splits without affecting existing data
    - Auto-generated README with statistics
    - Multiple sharding strategies
    - NERSC URL integration for streaming
"""

import os
import sys
import yaml
import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.hf_utils import (
    build_file_inventory,
    compute_inventory_statistics,
    generate_dataset_card,
    merge_hf_configs
)

try:
    from huggingface_hub import HfApi, hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not installed. Install with: pip install huggingface_hub")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded config from: {config_path}")
    return config


def validate_config(config: Dict) -> None:
    """
    Validate configuration has required fields.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If required fields are missing
    """
    required = ['huggingface', 'nersc', 'upload']
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required config section: {field}")
    
    if 'repo_id' not in config['huggingface']:
        raise ValueError("Missing huggingface.repo_id in config")
    
    if 'configs' not in config['upload']:
        raise ValueError("Missing upload.configs in config")
    
    logger.info("Config validation passed")


def validate_local_files(config: Dict) -> Dict[str, Dict]:
    """
    Validate local files exist and build inventory.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        File inventory dictionary
    """
    logger.info("Building file inventory...")
    
    source_base = config['upload']['source_base']
    configs = config['upload']['configs']
    
    inventory = build_file_inventory(source_base, configs)
    stats = compute_inventory_statistics(inventory)
    
    # Log statistics
    logger.info(f"File inventory built:")
    for config_name, config_stats in stats.items():
        logger.info(f"  {config_name}:")
        for split_name, split_stats in config_stats['splits'].items():
            logger.info(
                f"    {split_name}: {split_stats['num_files']} files, "
                f"{split_stats['num_events']} events, "
                f"{split_stats['size_mb']:.1f} MB"
            )
    
    return inventory


def copy_to_nersc_www(
    config: Dict,
    inventory: Dict[str, Dict],
    dry_run: bool = False
) -> None:
    """
    Copy Parquet files to NERSC www directory for HTTP access.
    
    Args:
        config: Configuration dictionary
        inventory: File inventory from validate_local_files()
        dry_run: If True, only print what would be done
    """
    logger.info("Copying files to NERSC www directory...")
    
    www_path = Path(config['nersc']['www_path'])
    upload_configs = config['upload']['configs']
    
    for upload_config in upload_configs:
        config_name = upload_config['name']
        split_name = upload_config['split']
        nersc_subpath = upload_config['nersc_subpath']
        
        # Get files for this config/split
        if config_name not in inventory or split_name not in inventory[config_name]:
            logger.warning(f"No files found for {config_name}/{split_name}")
            continue
        
        files = inventory[config_name][split_name]
        
        # Create destination directory
        dest_dir = www_path / nersc_subpath
        
        if dry_run:
            logger.info(f"[DRY RUN] Would create directory: {dest_dir}")
        else:
            dest_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dest_dir}")
        
        # Copy files
        for file_info in files:
            src_path = Path(file_info['path'])
            dest_path = dest_dir / file_info['filename']
            
            if dest_path.exists():
                logger.info(f"  Skipping (exists): {file_info['filename']}")
                continue
            
            if dry_run:
                logger.info(f"  [DRY RUN] Would copy: {file_info['filename']} ({file_info['size_mb']:.1f} MB)")
            else:
                shutil.copy2(src_path, dest_path)
                logger.info(f"  Copied: {file_info['filename']} ({file_info['size_mb']:.1f} MB)")
    
    logger.info("File copy complete")


def fetch_existing_configs(repo_id: str, token: str) -> Optional[List[Dict]]:
    """
    Fetch existing configs from HuggingFace dataset README.
    
    Args:
        repo_id: HuggingFace repository ID
        token: HuggingFace API token
        
    Returns:
        List of existing configs or None if repo doesn't exist
    """
    if not HF_AVAILABLE:
        return None
    
    try:
        api = HfApi(token=token)
        # Try to download README
        readme_path = hf_hub_download(
            repo_id=repo_id,
            filename="README.md",
            repo_type="dataset",
            token=token
        )
        
        # Parse configs from YAML front matter
        with open(readme_path, 'r') as f:
            content = f.read()
        
        # Simple YAML front matter parsing
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                yaml_content = parts[1]
                metadata = yaml.safe_load(yaml_content)
                if 'configs' in metadata:
                    logger.info(f"Found {len(metadata['configs'])} existing configs in repo")
                    return metadata['configs']
        
        return None
    
    except Exception as e:
        logger.info(f"Could not fetch existing configs (repo may not exist): {e}")
        return None


def update_huggingface_repo(
    config: Dict,
    dataset_card: str,
    dry_run: bool = False
) -> None:
    """
    Update HuggingFace repository with dataset card.
    
    Args:
        config: Configuration dictionary
        dataset_card: Generated README.md content
        dry_run: If True, only print what would be done
    """
    if not HF_AVAILABLE:
        logger.error("huggingface_hub not installed. Cannot update repository.")
        return
    
    repo_id = config['huggingface']['repo_id']
    token_env = config['huggingface'].get('token_env', 'HF_TOKEN')
    create_if_missing = config['huggingface'].get('create_if_missing', True)
    
    # Get token
    token = os.environ.get(token_env)
    if not token:
        logger.error(f"HuggingFace token not found in environment variable: {token_env}")
        logger.error("Set token with: export HF_TOKEN=<your_token>")
        logger.error("Get token from: https://huggingface.co/settings/tokens")
        return
    
    logger.info(f"Updating HuggingFace repository: {repo_id}")
    
    if dry_run:
        logger.info("[DRY RUN] Would update README.md")
        logger.info("=" * 80)
        logger.info("README.md content:")
        logger.info("=" * 80)
        print(dataset_card)
        logger.info("=" * 80)
        return
    
    # Initialize API
    api = HfApi(token=token)
    
    # Check if repo exists, create if needed
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        logger.info(f"Repository exists: {repo_id}")
    except Exception:
        if create_if_missing:
            logger.info(f"Creating repository: {repo_id}")
            api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=False
            )
        else:
            logger.error(f"Repository does not exist and create_if_missing=False")
            return
    
    # Upload README
    logger.info("Uploading README.md...")
    api.upload_file(
        path_or_fileobj=dataset_card.encode('utf-8'),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Update dataset card via ColliderML upload tool"
    )
    
    logger.info(f"✅ Repository updated successfully!")
    logger.info(f"View at: https://huggingface.co/datasets/{repo_id}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Upload ColliderML datasets to HuggingFace with NERSC hosting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload with config file
  python upload_to_huggingface.py --config configs/hf_upload_ttbar.yaml
  
  # Dry run to see what would happen
  python upload_to_huggingface.py --config configs/hf_upload_ttbar.yaml --dry-run
  
  # Skip file copying (files already on NERSC)
  python upload_to_huggingface.py --config configs/hf_upload_ttbar.yaml --skip-copy
        """
    )
    
    parser.add_argument(
        '--config',
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what would be done without making changes'
    )
    parser.add_argument(
        '--skip-copy',
        action='store_true',
        help='Skip copying files to NERSC (assume already there)'
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
        
        # Validate local files and build inventory
        inventory = validate_local_files(config)
        
        if not inventory:
            logger.error("No files found in inventory. Check your configuration.")
            sys.exit(1)
        
        # Copy files to NERSC www (unless skipped)
        if not args.skip_copy:
            copy_to_nersc_www(config, inventory, dry_run=args.dry_run)
        else:
            logger.info("Skipping file copy (--skip-copy)")
        
        # Fetch existing configs for progressive update
        repo_id = config['huggingface']['repo_id']
        token_env = config['huggingface'].get('token_env', 'HF_TOKEN')
        token = os.environ.get(token_env)
        
        existing_configs = None
        if token:
            existing_configs = fetch_existing_configs(repo_id, token)
        
        # Generate dataset card
        logger.info("Generating dataset card...")
        dataset_card = generate_dataset_card(
            repo_id=repo_id,
            base_url=config['nersc']['base_url'],
            inventory=inventory,
            configs=config['upload']['configs'],
            template_path=config.get('readme', {}).get('template_path'),
            dataset_description=config.get('readme', {}).get('description'),
            existing_configs=existing_configs
        )
        
        # Update HuggingFace repository
        update_huggingface_repo(config, dataset_card, dry_run=args.dry_run)
        
        if args.dry_run:
            logger.info("\n✅ Dry run complete. No changes made.")
        else:
            logger.info("\n✅ Upload complete!")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()

