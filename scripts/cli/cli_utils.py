#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common utilities for ColliderML CLI tools.
Contains functionality shared between run_stage.py and job_submission.py.
"""

import os
import yaml
import logging
import subprocess
import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Constants
CONFIG_FILE_NAME = "expanded_config.yaml"
GIT_COMMIT_SUCCESS_FILE = ".git_commit_success"

# Stage to script mappings
STAGE_SCRIPT_MAP = {
    # Simulation scripts
    "madgraph_generation": "simulation/madgraph_gen.py",
    "pythia_generation": "simulation/pythia_gen.py",
    "merge_smear": "simulation/merge_and_smear.py",
    "simulation": "simulation/ddsim_run.py",
    "digitization": "simulation/digi_and_reco.py",
    
    # Postprocessing scripts
    "build_tracks": "postprocessing/convert_tracks.py",
    "build_hits": "postprocessing/convert_hits.py",
    "build_particles": "postprocessing/convert_particles.py"
}

def get_stage_script_path(config, software_repo_path=None):
    """
    Determines the appropriate script path for a stage based on config.
    
    Args:
        config: The configuration dictionary
        software_repo_path: Optional path to the software repository root
                           (defaults to using scripts directory relative to this file)
    
    Returns:
        Path object pointing to the stage script
    """
    # First check if config has its own stage_script_map
    stage = config["stage"]
    stage_script_rel_path = config.get("stage_script_map", {}).get(stage)
    
    # If not in config, use default mapping
    if not stage_script_rel_path:
        stage_script_rel_path = STAGE_SCRIPT_MAP.get(stage)
        if not stage_script_rel_path:
            raise ValueError(f"No script defined for stage: {stage}")
    
    # Determine base scripts directory
    if not software_repo_path:
        scripts_dir = Path(__file__).resolve().parent.parent  # From cli/ up to scripts/
    else:
        scripts_dir = software_repo_path / "scripts"
    
    # Construct and validate absolute path
    script_path = scripts_dir / stage_script_rel_path
    if not script_path.is_file():
        raise FileNotFoundError(f"Stage script not found at {script_path}")
        
    return script_path

def get_version_directory(config):
    """
    Gets the version directory path from a config dictionary.
    Path: output_base_dir/campaign/dataset/version/
    
    Args:
        config: The configuration dictionary
        
    Returns:
        Path to the version directory
    """
    if "campaign" not in config:
        raise ValueError("Configuration missing 'campaign' field for version directory construction.")
    if "dataset" not in config:
        raise ValueError("Configuration missing 'dataset' field for version directory construction.")
    if "version" not in config:
        raise ValueError("Configuration missing 'version' field for version directory construction.")
    
    common_config = config.get("common")
    if not isinstance(common_config, dict):
        raise ValueError("Configuration missing 'common' section or 'common' is not a dictionary; required for 'output_base_dir'.")
    
    if "output_base_dir" not in common_config:
        raise ValueError("Configuration missing 'output_base_dir' in 'common' section.")
        
    base_dir = Path(common_config["output_base_dir"])
    version_dir = base_dir / config["campaign"] / config["dataset"] / config["version"]
    return version_dir

def get_run_directory(config):
    """Get the runs directory path from config."""
    version_dir = get_version_directory(config)
    return version_dir / "runs"

def get_git_root(start_path):
    """
    Traverse up from start_path to find the .git directory.
    
    Args:
        start_path: Path to start searching from
        
    Returns:
        Path to git repository root or None if not found
    """
    current_path = Path(start_path).resolve()
    while current_path != current_path.parent:
        if (current_path / ".git").is_dir():
            return current_path
        current_path = current_path.parent
    
    # Check root directory as final attempt
    if (current_path / ".git").is_dir():
        return current_path
    
    return None

def git_commit_and_log_config(config, config_path, software_repo_path, force_commit=False):
    """
    Performs Git branch check, commits changes, tags the version, and logs the config.
    
    Args:
        config: The configuration dictionary
        config_path: Path to the original config file
        software_repo_path: Path to the software repository root
        force_commit: Whether to force a commit and tag even if no changes or success marker exists.
        
    Returns:
        bool: True on success, False on failure
    """
    try:
        if "campaign" not in config:
            logger.error("Configuration missing 'campaign' field. Cannot proceed with git operations.")
            return False
        if "dataset" not in config:
            logger.error("Configuration missing 'dataset' field. Cannot proceed with git operations.")
            return False
        if "version" not in config:
            logger.error("Configuration missing 'version' field. Cannot proceed with git operations.")
            return False

        # --- 1. Branch Check ---
        expected_branch_name = f"campaign/{config['campaign']}/dataset/{config['dataset']}"
        try:
            current_branch_cmd = ["git", "-C", str(software_repo_path), "rev-parse", "--abbrev-ref", "HEAD"]
            current_branch_name = subprocess.check_output(current_branch_cmd, text=True, cwd=software_repo_path).strip()
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get current git branch in {software_repo_path}: {e.stderr}")
            return False

        if current_branch_name != expected_branch_name:
            logger.error(f"Incorrect Git branch. Expected: '{expected_branch_name}', but currently on: '{current_branch_name}'.")
            logger.error(f"Please switch to branch '{expected_branch_name}' or create it (e.g., git checkout -b {expected_branch_name}) and push your changes there before running.")
            return False
        logger.info(f"Git branch check passed. Currently on expected branch: {current_branch_name}")

        # --- 2. Determine output directory & skip if already processed (unless forced) ---
        output_version_dir = get_version_directory(config) # Will now include campaign
        commit_success_file = output_version_dir / GIT_COMMIT_SUCCESS_FILE
        
        output_version_dir.mkdir(parents=True, exist_ok=True) # Ensure it exists early

        if not force_commit and commit_success_file.exists():
            logger.info(f"Success marker file {commit_success_file} exists. Skipping git commit and tag for this version.")
            logged_config_path = output_version_dir / CONFIG_FILE_NAME
            if not logged_config_path.exists():
                 logger.warning(f"Git commit marker exists, but config snapshot {logged_config_path} not found. Re-logging config.")
                 with open(logged_config_path, 'w') as f_out:
                    yaml.dump(config, f_out, default_flow_style=False, sort_keys=False)
                 logger.info(f"Config snapshot saved to {logged_config_path}")
            return True

        # --- 3. Git Commit Logic ---
        logger.info(f"Attempting to git commit changes in {software_repo_path} on branch {current_branch_name}")
        
        status_cmd = ["git", "-C", str(software_repo_path), "status", "--porcelain"]
        process = subprocess.run(status_cmd, capture_output=True, text=True)
        if process.returncode != 0:
            logger.error(f"Git status check failed in {software_repo_path}: {process.stderr}")
            return False
        
        committed_this_run = False
        if not process.stdout.strip() and not force_commit: # No changes and not forcing a commit
            logger.info(f"No new changes to commit in {software_repo_path}.")
        else: # There are changes, or force_commit is true
            add_cmd = ["git", "-C", str(software_repo_path), "add", "."]
            subprocess.run(add_cmd, check=True, capture_output=True) # Use check=True for auto error on fail
            
            process_after_add = subprocess.run(status_cmd, capture_output=True, text=True)
            if process_after_add.stdout.strip(): # If there are still changes staged
                commit_message = (f"Auto-commit for campaign '{config['campaign']}', "
                                  f"dataset '{config['dataset']}', version '{config['version']}'")
                commit_cmd = ["git", "-C", str(software_repo_path), "commit", "-m", commit_message]
                subprocess.run(commit_cmd, check=True, capture_output=True)
                logger.info(f"Git commit successful in {software_repo_path}.")
                committed_this_run = True
            elif process.stdout.strip(): 
                 logger.info("No effective changes to commit after 'git add'.")
            else: 
                 logger.info("No changes to commit, and force_commit did not find new changes to force through.")


        git_hash_cmd = ["git", "-C", str(software_repo_path), "rev-parse", "HEAD"]
        current_git_hash = subprocess.check_output(git_hash_cmd, text=True, cwd=software_repo_path).strip()
        logger.info(f"Current Git HEAD for {software_repo_path}: {current_git_hash}")

        # --- 4. Git Tagging Logic ---
        tag_name = f"version:{config['version']}" # Colons are generally fine in tag names
        tag_message = (f"Tag for campaign: {config['campaign']}, dataset: {config['dataset']}, "
                       f"version: {config['version']}")
        
        tag_exists_cmd = ["git", "-C", str(software_repo_path), "rev-parse", "-q", "--verify", f"refs/tags/{tag_name}"]
        tag_check_process = subprocess.run(tag_exists_cmd, capture_output=True, text=True, cwd=software_repo_path)
        
        tag_exists = tag_check_process.returncode == 0
        existing_tag_hash = tag_check_process.stdout.strip() if tag_exists else None

        should_create_tag = True
        if tag_exists:
            if existing_tag_hash == current_git_hash and not force_commit:
                logger.info(f"Tag '{tag_name}' already exists and points to the current commit ({current_git_hash}). Skipping tag creation.")
                should_create_tag = False
            elif force_commit:
                logger.info(f"Tag '{tag_name}' already exists. Deleting and re-creating due to force_commit=True.")
                delete_tag_cmd = ["git", "-C", str(software_repo_path), "tag", "-d", tag_name]
                subprocess.run(delete_tag_cmd, check=True, capture_output=True)
            else: 
                logger.error(f"Tag '{tag_name}' exists on a different commit ({existing_tag_hash} vs HEAD {current_git_hash}). Use --force-commit to retag, or resolve manually.")
                return False


        if should_create_tag:
            logger.info(f"Attempting to create tag '{tag_name}' pointing to commit {current_git_hash}.")
            create_tag_cmd = ["git", "-C", str(software_repo_path), "tag", "-a", tag_name, "-m", tag_message, current_git_hash]
            try:
                subprocess.run(create_tag_cmd, check=True, capture_output=True)
                logger.info(f"Successfully created/updated tag '{tag_name}'.")
            except subprocess.CalledProcessError as e_tag:
                logger.error(f"Failed to create tag '{tag_name}': {e_tag.stderr.decode() if isinstance(e_tag.stderr, bytes) else e_tag.stderr}")
                return False


        # --- 5. Save Config Snapshot and Success Marker ---
        logged_config_path = output_version_dir / CONFIG_FILE_NAME
        with open(logged_config_path, 'w') as f_out:
            yaml.dump(config, f_out, default_flow_style=False, sort_keys=False)
        logger.info(f"Full configuration snapshot saved to {logged_config_path}")
        
        with open(commit_success_file, 'w') as f_marker:
            f_marker.write(f"Commit successful at {datetime.datetime.now()}\n"
                           f"Git Branch: {current_branch_name}\n"
                           f"Git Hash: {current_git_hash}\n"
                           f"Git Tag: {tag_name if should_create_tag or (tag_exists and existing_tag_hash == current_git_hash) else 'skipped or failed'}\n"
                           f"Config: {logged_config_path.name}\n")
        logger.info(f"Git commit and tag success marker created at {commit_success_file}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Git operation failed during script execution: {e.cmd}")
        stderr_output = e.stderr.decode('utf-8').strip() if isinstance(e.stderr, bytes) else e.stderr.strip() if e.stderr else ""
        stdout_output = e.stdout.decode('utf-8').strip() if isinstance(e.stdout, bytes) else e.stdout.strip() if e.stdout else ""
        if stdout_output:
            logger.error(f"Stdout: {stdout_output}")
        if stderr_output:
            logger.error(f"Stderr: {stderr_output}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during git_commit_and_log_config: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def create_necessary_directories(config):
    """
    Create necessary directories for a run based on config.
    Path: output_base_dir/campaign/dataset/version/ + subdirs
    
    Args:
        config: The configuration dictionary
        
    Returns:
        dict: Dictionary of directory paths created
    """
    version_dir = get_version_directory(config) # This will now include campaign
    run_dir = version_dir / "runs"
    log_dir = version_dir / "logs" / f"stage_{config['stage']}"
    validation_dir = version_dir / "validation" / f"stage_{config['stage']}"
    
    dirs_to_create = [version_dir, run_dir, log_dir, validation_dir]
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)
    
    return {
        "version_dir": version_dir,
        "run_dir": run_dir,
        "log_dir": log_dir,
        "validation_dir": validation_dir
    } 