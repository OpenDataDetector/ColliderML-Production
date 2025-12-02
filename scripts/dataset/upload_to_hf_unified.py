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
    
    # Direct upload (faster, skips re-sharding)
    python scripts/dataset/upload_to_hf_unified.py scripts/dataset/unified_dataset_config.yaml --direct-upload

Features:
    - Progressive uploads: add configs one at a time without affecting existing data
    - Uses datasets library for automatic Parquet handling
    - Automatic README.md generation and updates
    - Skip existing configs to avoid re-uploading
    - Direct upload mode for faster uploads (experimental)
"""

import argparse
import logging
import os
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set HuggingFace cache to project directory to avoid home directory quota issues
# This must be done BEFORE importing datasets/huggingface_hub
HF_CACHE_DIR = "/global/cfs/cdirs/m4958/data/ColliderML/.hf_cache"
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_DIR, "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")

try:
    from datasets import load_dataset, get_dataset_config_names
    from huggingface_hub import HfApi, upload_folder
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


def update_readme_configs(
    repo_id: str,
    configs_to_add: List[Dict[str, str]],
    token: str,
    use_direct_paths: bool = False,
    dry_run: bool = False
) -> bool:
    """
    Update the README.md YAML frontmatter to include config definitions.
    
    This is required for HuggingFace to auto-detect multiple configs from
    parquet files. Without this, only a single 'default' config is detected.
    
    Args:
        repo_id: HuggingFace repository ID
        configs_to_add: List of dicts with 'config_name' keys
        token: HuggingFace API token
        use_direct_paths: If True, use data/<config>/ paths (direct upload)
                         If False, use <config>/ paths (standard push_to_hub)
        dry_run: If True, don't actually update
    
    Returns True if successful.
    """
    import re
    import tempfile
    
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Updating README.md with {len(configs_to_add)} config(s)")
    
    if dry_run:
        for cfg in configs_to_add:
            config_name = cfg['config_name']
            if use_direct_paths:
                logger.info(f"  Would add config: {config_name} -> data/{config_name}/*.parquet")
            else:
                logger.info(f"  Would add config: {config_name} -> {config_name}/*.parquet")
        return True
    
    try:
        api = HfApi(token=token)
        
        # Download current README
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                readme_path = api.hf_hub_download(
                    repo_id=repo_id,
                    filename="README.md",
                    repo_type="dataset",
                    local_dir=tmpdir
                )
                with open(readme_path, 'r') as f:
                    readme_content = f.read()
            except Exception as e:
                logger.warning(f"Could not download README.md: {e}")
                logger.warning("Creating new README with configs")
                readme_content = ""
        
        # Parse existing YAML frontmatter
        yaml_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', readme_content, re.DOTALL)
        if yaml_match:
            yaml_content = yaml_match.group(1)
            markdown_content = readme_content[yaml_match.end():]
            try:
                yaml_data = yaml.safe_load(yaml_content) or {}
            except:
                yaml_data = {}
        else:
            yaml_data = {}
            markdown_content = readme_content
        
        # Get or create configs list
        existing_configs = yaml_data.get('configs', [])
        existing_config_names = {c.get('config_name') for c in existing_configs if c}
        
        # Add new configs
        configs_added = 0
        for cfg in configs_to_add:
            config_name = cfg['config_name']
            if config_name not in existing_config_names:
                if use_direct_paths:
                    # Direct upload: files in data/<config>/
                    path_pattern = f"data/{config_name}/*.parquet"
                else:
                    # Standard push_to_hub: files in <config>/
                    path_pattern = f"{config_name}/*.parquet"
                
                new_config = {
                    'config_name': config_name,
                    'data_files': [{
                        'split': 'train',
                        'path': path_pattern
                    }]
                }
                existing_configs.append(new_config)
                configs_added += 1
                logger.info(f"  Adding config: {config_name} -> {path_pattern}")
        
        if configs_added == 0:
            logger.info("  No new configs to add (all already exist)")
            return True
        
        yaml_data['configs'] = existing_configs
        
        # Rebuild README
        new_yaml = yaml.dump(yaml_data, default_flow_style=False, sort_keys=False, allow_unicode=True)
        new_readme = f"---\n{new_yaml}---\n{markdown_content}"
        
        # Upload updated README
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(new_readme)
            tmp_readme_path = f.name
        
        try:
            api.upload_file(
                path_or_fileobj=tmp_readme_path,
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Update configs: add {configs_added} new config(s)"
            )
            logger.info(f"✓ Updated README.md with {configs_added} new config(s)")
        finally:
            os.unlink(tmp_readme_path)
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to update README configs: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


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


def sort_parquet_files_numerically(parquet_files: List[Path]) -> List[str]:
    """
    Sort parquet files numerically by event range.
    
    Files are named like: prefix.events0-999.parquet, prefix.events1000-1999.parquet
    Alphabetical sort would put events10000 before events1000, so we need numeric sort.
    """
    import re
    
    def extract_event_start(filepath: Path) -> int:
        """Extract the starting event number from filename."""
        match = re.search(r'events(\d+)-', filepath.name)
        if match:
            return int(match.group(1))
        return 0  # Default if pattern not found
    
    sorted_files = sorted(parquet_files, key=extract_event_start)
    return [str(f) for f in sorted_files]


def validate_event_ids(dataset, config_name: str) -> bool:
    """
    Validate that event IDs are continuous and unique from 0 to N-1.
    
    Returns True if valid, False otherwise.
    """
    logger.info(f"  Validating event IDs for {config_name}...")
    
    try:
        # Get all event IDs
        event_ids = dataset['event_id']
        n_events = len(event_ids)
        
        # Check uniqueness
        unique_ids = set(event_ids)
        if len(unique_ids) != n_events:
            logger.error(f"  ✗ Found {n_events - len(unique_ids)} duplicate event IDs!")
            return False
        
        # Check range (should be 0 to N-1)
        min_id = min(event_ids)
        max_id = max(event_ids)
        
        if min_id != 0:
            logger.warning(f"  ⚠ Event IDs start at {min_id}, not 0")
        
        if max_id != n_events - 1:
            logger.warning(f"  ⚠ Event IDs end at {max_id}, expected {n_events - 1}")
        
        # Check continuity (no gaps)
        expected_ids = set(range(min_id, max_id + 1))
        missing_ids = expected_ids - unique_ids
        
        if missing_ids:
            # Show sample of missing IDs
            missing_sample = sorted(missing_ids)[:10]
            logger.error(f"  ✗ Found {len(missing_ids)} missing event IDs!")
            logger.error(f"    Sample missing: {missing_sample}...")
            return False
        
        logger.info(f"  ✓ Event IDs validated: {n_events} unique, continuous from {min_id} to {max_id}")
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Failed to validate event IDs: {e}")
        return False


def upload_config(
    config_info: Dict,
    repo_id: str,
    max_shard_size: str,
    private: bool,
    token: str,
    num_workers: int = 1,
    dry_run: bool = False
) -> Tuple[bool, float]:
    """
    Upload a single config to HuggingFace using datasets library (standard method).

    Args:
        config_info: Dict with config metadata
        repo_id: HuggingFace repo ID
        max_shard_size: Max size per shard (e.g., "500MB")
        private: Whether repo is private
        token: HuggingFace token
        num_workers: Number of parallel processes for upload (default: 1)
        dry_run: If True, don't actually upload

    Returns (success: bool, elapsed_time: float).
    """
    config_name = config_info['config_name']
    data_path = config_info['path']

    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Uploading config: {config_name}")
    logger.info(f"  Data path: {data_path}")
    logger.info(f"  Method: datasets push_to_hub (loads, validates, re-shards, {num_workers} workers)")

    if dry_run:
        logger.info(f"  Would upload {len(list(data_path.glob('*.parquet')))} parquet files")
        return True, 0.0

    start_time = time.time()
    
    try:
        # Get parquet files and sort them numerically (not alphabetically!)
        parquet_files = list(data_path.glob("*.parquet"))
        sorted_files = sort_parquet_files_numerically(parquet_files)
        logger.info(f"  Found {len(sorted_files)} parquet files (sorted numerically)")
        
        # Load dataset from sorted parquet files
        logger.info(f"  Loading parquet files...")
        load_start = time.time()
        dataset = load_dataset(
            "parquet",
            data_files=sorted_files,
            split="train"
        )
        load_time = time.time() - load_start
        logger.info(f"  Loaded {len(dataset)} events in {load_time:.1f}s")
        
        # Validate event IDs are continuous and unique
        if not validate_event_ids(dataset, config_name):
            logger.error(f"  ✗ Event ID validation failed - aborting upload")
            return False, time.time() - start_time

        # Push to hub with num_proc for parallel upload
        logger.info(f"  Pushing to HuggingFace ({num_workers} workers)...")
        push_start = time.time()
        dataset.push_to_hub(
            repo_id,
            config_name=config_name,
            max_shard_size=max_shard_size,
            private=private,
            token=token,
            num_proc=num_workers if num_workers > 1 else None
        )
        push_time = time.time() - push_start
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Successfully uploaded: {config_name}")
        logger.info(f"  Load time: {load_time:.1f}s, Push time: {push_time:.1f}s, Total: {elapsed:.1f}s")
        return True, elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"✗ Failed to upload {config_name}: {e}")
        logger.error(f"  Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"  Full traceback:\n{traceback.format_exc()}")
        return False, elapsed


def upload_config_direct(
    config_info: Dict,
    repo_id: str,
    token: str,
    num_workers: int = 5,
    dry_run: bool = False
) -> Tuple[bool, float]:
    """
    Upload a single config directly using upload_folder (faster, no re-sharding).
    
    This uploads parquet files directly without loading into datasets library.
    Files are uploaded to: data/{config_name}/ folder in the repo.
    
    NOTE: HuggingFace auto-detection won't work with our naming scheme.
    We need to create/update the dataset README with YAML config to define
    how to load each config.
    
    Args:
        config_info: Dict with config metadata
        repo_id: HuggingFace repo ID
        token: HuggingFace token
        num_workers: Number of concurrent upload threads (default: 5)
        dry_run: If True, don't actually upload
    
    Returns (success: bool, elapsed_time: float).
    """
    import tempfile
    import shutil
    
    config_name = config_info['config_name']
    data_path = config_info['path']

    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Direct uploading config: {config_name}")
    logger.info(f"  Data path: {data_path}")
    logger.info(f"  Method: direct upload_folder (no re-sharding, {num_workers} workers)")

    # Get parquet files and calculate total size
    parquet_files = list(data_path.glob("*.parquet"))
    total_size = sum(f.stat().st_size for f in parquet_files)
    logger.info(f"  Found {len(parquet_files)} parquet files ({total_size / 1e9:.2f} GB)")

    if dry_run:
        logger.info(f"  Would upload to: data/{config_name}/")
        return True, 0.0

    start_time = time.time()
    
    try:
        # Sort files numerically for validation
        sorted_files = sort_parquet_files_numerically(parquet_files)
        
        # Quick validation: load just to check event IDs
        logger.info(f"  Validating event IDs (loading for check only)...")
        validate_start = time.time()
        dataset = load_dataset(
            "parquet",
            data_files=sorted_files,
            split="train"
        )
        if not validate_event_ids(dataset, config_name):
            logger.error(f"  ✗ Event ID validation failed - aborting upload")
            return False, time.time() - start_time
        # Free memory
        del dataset
        validate_time = time.time() - validate_start
        logger.info(f"  Validation passed in {validate_time:.1f}s")
        
        # Create temporary directory with correct structure
        # Files need to be: data/{config_name}/train-XXXXX-of-XXXXX.parquet
        # to be auto-detected as train split
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "data" / config_name
            config_dir.mkdir(parents=True)
            
            # Copy and rename files to HuggingFace naming convention
            n_files = len(sorted_files)
            logger.info(f"  Preparing {n_files} files with HF naming convention...")
            prep_start = time.time()
            
            for i, src_file in enumerate(sorted_files):
                # Use train-00000-of-00100.parquet format
                dst_name = f"train-{i:05d}-of-{n_files:05d}.parquet"
                dst_file = config_dir / dst_name
                # Create symlink instead of copy to save time
                os.symlink(src_file, dst_file)
            
            prep_time = time.time() - prep_start
            logger.info(f"  Files prepared in {prep_time:.1f}s (using symlinks)")
            
            # Upload the folder with multi-threaded upload
            logger.info(f"  Uploading to HuggingFace ({num_workers} concurrent threads)...")
            upload_start = time.time()
            
            api = HfApi(token=token)
            # Note: upload_folder uses create_commit internally which has num_threads parameter
            # But upload_folder doesn't expose it directly - we use create_commit for more control
            from huggingface_hub import CommitOperationAdd
            
            # Build list of commit operations
            operations = []
            for parquet_file in config_dir.glob("*.parquet"):
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=f"data/{config_name}/{parquet_file.name}",
                        path_or_fileobj=str(parquet_file)
                    )
                )
            
            # Use create_commit with num_threads for parallel uploads
            api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message=f"Add config: {config_name}",
                num_threads=num_workers
            )
            
            upload_time = time.time() - upload_start
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Successfully uploaded: {config_name}")
        logger.info(f"  Validate: {validate_time:.1f}s, Prep: {prep_time:.1f}s, Upload: {upload_time:.1f}s, Total: {elapsed:.1f}s")
        return True, elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"✗ Failed to upload {config_name}: {e}")
        logger.error(f"  Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"  Full traceback:\n{traceback.format_exc()}")
        return False, elapsed


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


def clean_repo(repo_id: str, token: str, dry_run: bool = False) -> bool:
    """
    Delete all parquet files from the repository.
    
    Removes files from:
    - Root-level config folders (e.g., ttbar_pu0_particles/)
    - data/ folder (e.g., data/ttbar_pu0_particles/)
    
    Keeps README.md and .gitattributes.
    
    Returns True if successful, False otherwise.
    """
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Cleaning repository: {repo_id}")
    
    try:
        api = HfApi(token=token)
        
        # List all files in the repo
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        
        # Find parquet files and config folders to delete
        files_to_delete = []
        for f in repo_files:
            # Skip essential files
            if f in ['README.md', '.gitattributes']:
                continue
            # Delete parquet files
            if f.endswith('.parquet'):
                files_to_delete.append(f)
        
        if not files_to_delete:
            logger.info("  No parquet files found to delete")
            return True
        
        logger.info(f"  Found {len(files_to_delete)} parquet files to delete")
        
        # Group by folder for cleaner logging
        folders = set()
        for f in files_to_delete:
            parts = f.split('/')
            if len(parts) > 1:
                folders.add('/'.join(parts[:-1]))
        
        for folder in sorted(folders):
            count = sum(1 for f in files_to_delete if f.startswith(folder + '/'))
            logger.info(f"    {folder}/: {count} files")
        
        if dry_run:
            logger.info("  [DRY RUN] Would delete these files")
            return True
        
        # Delete files using commit operations
        from huggingface_hub import CommitOperationDelete
        
        operations = [
            CommitOperationDelete(path_in_repo=f) for f in files_to_delete
        ]
        
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


def upload_readme(repo_id: str, readme_path: Path, token: str, dry_run: bool = False) -> bool:
    """
    Upload README.md to the HuggingFace repository.

    Returns True if successful, False otherwise.
    """
    if not readme_path.exists():
        logger.warning(f"README file not found: {readme_path}")
        return False

    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Uploading README from: {readme_path}")

    if dry_run:
        logger.info(f"  Would upload README.md to {repo_id}")
        return True

    try:
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token
        )
        logger.info(f"✓ Successfully uploaded README.md")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to upload README: {e}")
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
  
  # Direct upload (faster, no re-sharding) - EXPERIMENTAL
  python upload_to_hf_unified.py unified_dataset_config.yaml --direct-upload
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

    parser.add_argument(
        '--no-readme',
        action='store_true',
        help='Skip uploading the README.md file'
    )

    parser.add_argument(
        '--direct-upload',
        action='store_true',
        help='Use direct folder upload (faster, no re-sharding). EXPERIMENTAL: '
             'Files are renamed to HF convention and uploaded directly.'
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Number of concurrent upload threads for direct upload (default: half of CPU cores). '
             'More workers = faster upload but more bandwidth usage.'
    )

    parser.add_argument(
        '--update-configs-only',
        action='store_true',
        help='Only update README.md YAML configs without uploading data. '
             'Use this to fix HuggingFace dataset viewer not showing configs.'
    )

    parser.add_argument(
        '--skip-readme-config-update',
        action='store_true',
        help='Skip updating README.md YAML configs after upload. '
             'Useful for timing studies or when managing configs manually.'
    )

    parser.add_argument(
        '--clean-repo',
        action='store_true',
        help='Delete all parquet files from the repo before uploading. '
             'Removes files from both root config folders and data/ folder. '
             'Useful for timing studies or starting fresh.'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Set default num_workers to half of CPU cores (minimum 2)
    if args.num_workers is None:
        import multiprocessing
        args.num_workers = max(2, multiprocessing.cpu_count() // 2)
        logger.info(f"Using {args.num_workers} upload workers (half of {multiprocessing.cpu_count()} CPUs)")

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

        # Clean repository if requested
        if args.clean_repo:
            logger.info("=" * 80)
            logger.info("Cleaning repository (--clean-repo)")
            logger.info("=" * 80)
            if not clean_repo(repo_id, token, args.dry_run):
                logger.error("Failed to clean repository. Exiting.")
                sys.exit(1)

        # Upload README.md if configured and not skipped
        if not args.no_readme:
            readme_path_str = config['huggingface'].get('readme_path')
            if readme_path_str:
                # Resolve path relative to config file
                config_dir = Path(args.config).parent
                readme_path = config_dir / readme_path_str
                upload_readme(repo_id, readme_path, token, args.dry_run)
            else:
                logger.debug("No readme_path configured, skipping README upload")

        # Get existing configs if skip_existing is enabled
        existing_configs = []
        if skip_existing and not args.dry_run:
            existing_configs = get_existing_configs(repo_id, token)

        # Build list of configs to upload
        all_configs = build_config_list(config)

        if not all_configs:
            logger.warning("No configs found to upload. Check your configuration.")
            sys.exit(0)

        # Handle --update-configs-only: just update README YAML configs and exit
        if args.update_configs_only:
            logger.info("=" * 80)
            logger.info("Updating README.md YAML configs only (no data upload)")
            logger.info("=" * 80)
            
            # Determine which configs to add based on filtering
            configs_to_add = all_configs
            if args.configs:
                configs_to_add = [c for c in configs_to_add if c['config_name'] in args.configs]
            
            if not configs_to_add:
                logger.warning("No configs specified. Use --configs or provide all configs.")
                sys.exit(1)
            
            logger.info(f"Adding {len(configs_to_add)} config(s) to README.md:")
            for c in configs_to_add:
                logger.info(f"  - {c['config_name']}")
            
            success = update_readme_configs(
                repo_id,
                configs_to_add,
                token,
                use_direct_paths=args.direct_upload,
                dry_run=args.dry_run
            )
            
            if success:
                logger.info("\n" + "=" * 80)
                logger.info("✓ README configs updated successfully!")
                logger.info(f"View dataset at: https://huggingface.co/datasets/{repo_id}")
                logger.info("=" * 80)
            else:
                logger.error("✗ Failed to update README configs")
                sys.exit(1)
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
        logger.info(f"Upload method: {'direct (fast)' if args.direct_upload else 'standard (push_to_hub)'}")
        for c in configs_to_upload:
            logger.info(f"  - {c['config_name']}")
        logger.info("=" * 80)

        if args.dry_run:
            logger.info("[DRY RUN] No changes will be made")

        # Upload each config
        success_count = 0
        failed_count = 0
        total_upload_time = 0.0
        upload_times = []

        for i, config_info in enumerate(configs_to_upload, 1):
            logger.info(f"\n[{i}/{len(configs_to_upload)}] Processing {config_info['config_name']}...")

            if args.direct_upload:
                success, elapsed = upload_config_direct(
                    config_info,
                    repo_id,
                    token,
                    num_workers=args.num_workers,
                    dry_run=args.dry_run
                )
            else:
                success, elapsed = upload_config(
                    config_info,
                    repo_id,
                    max_shard_size,
                    private,
                    token,
                    num_workers=args.num_workers,
                    dry_run=args.dry_run
                )

            if success:
                success_count += 1
                total_upload_time += elapsed
                upload_times.append((config_info['config_name'], elapsed))
            else:
                failed_count += 1

        # Update README.md YAML configs for auto-detection
        # This is critical for HuggingFace to show multiple configs in the viewer
        if success_count > 0 and not args.skip_readme_config_update:
            # Collect successfully uploaded configs
            successful_configs = [
                {'config_name': name} for name, _ in upload_times
            ]
            logger.info(f"\nUpdating README.md with {len(successful_configs)} config(s)...")
            update_readme_configs(
                repo_id,
                successful_configs,
                token,
                use_direct_paths=args.direct_upload,  # data/<config>/ vs <config>/
                dry_run=args.dry_run
            )
        elif args.skip_readme_config_update and success_count > 0:
            logger.info(f"\nSkipping README config update (--skip-readme-config-update)")

        # Final summary
        logger.info("\n" + "=" * 80)
        if args.dry_run:
            logger.info("✓ Dry run complete")
        else:
            logger.info(f"✓ Upload complete!")
            logger.info(f"  Successfully uploaded: {success_count}")
            logger.info(f"  Failed: {failed_count}")
            if upload_times:
                logger.info(f"\n  Timing Summary:")
                for name, elapsed in upload_times:
                    logger.info(f"    {name}: {elapsed:.1f}s")
                logger.info(f"    ────────────────────────────────")
                logger.info(f"    Total upload time: {total_upload_time:.1f}s ({total_upload_time/60:.1f} min)")
                if len(upload_times) > 1:
                    avg_time = total_upload_time / len(upload_times)
                    logger.info(f"    Average per config: {avg_time:.1f}s")
            logger.info(f"\nView dataset at: https://huggingface.co/datasets/{repo_id}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
