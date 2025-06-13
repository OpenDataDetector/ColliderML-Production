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

# Define stage categories
SIMULATION_STAGES = ["madgraph_generation", "pythia_generation", "merge_smear", "simulation", "digitization"]
POSTPROCESSING_STAGES = ["build_tracks", "build_hits", "build_particles"]
VALID_STAGES = SIMULATION_STAGES + POSTPROCESSING_STAGES

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

def get_env_setup_cmds(config):
    """
    Gets and processes environment setup commands from the configuration.

    It reads variable values from the `env_setup.env_variables` section
    and searches for commands using a hierarchical approach:
    1. Look for a key matching the specific stage name (e.g., 'madgraph_generation').
    2. If not found, fall back to the stage category (e.g., 'simulation').

    It then substitutes all {VARIABLE} occurrences in the found commands.

    Args:
        config (dict): The main configuration dictionary, which must include
                       the 'env_setup' section.

    Returns:
        list: A list of processed environment setup commands.
    """
    env_setup_config = config.get("env_setup", {})
    if not env_setup_config:
        logger.warning("`env_setup` section not found in configuration. Cannot process environment.")
        return []

    # Get the env_variables dictionary, with fallback to old "variables" name for compatibility
    env_variables_config = env_setup_config.get("env_variables", env_setup_config.get("variables", {}))

    # Dynamically create a dictionary for formatting variables.
    # Keys are uppercased to match conventions like {SOFTWARE_DIR}.
    format_dict = {key.upper(): value for key, value in env_variables_config.items() if isinstance(value, str)}
    
    # Resolve nested variable references (e.g., variables that reference other variables)
    # We need multiple passes to handle chained references
    max_iterations = 5  # Prevent infinite loops
    for iteration in range(max_iterations):
        substitutions_made = False
        for key, value in format_dict.items():
            try:
                new_value = value.format(**format_dict)
                if new_value != value:
                    format_dict[key] = new_value
                    substitutions_made = True
            except KeyError:
                # Some variables may reference others that haven't been resolved yet
                pass
        
        if not substitutions_made:
            break
    else:
        logger.warning("Maximum iterations reached while resolving nested variables. Some may be unresolved.")

    # Ensure SOFTWARE_DIR is present, as it's fundamental.
    if "SOFTWARE_DIR" not in format_dict or not format_dict["SOFTWARE_DIR"]:
        logger.error("`software_dir` not defined or is empty in the `env_setup.env_variables` section of your config.")
        raise ValueError("Missing essential configuration: software_dir")

    stage = config.get("stage")
    if not stage:
        logger.warning("Cannot determine environment setup: 'stage' not in config.")
        return []

    # Hierarchical search for command list: stage-specific first, then category.
    stage_cmds = []
    if stage in env_setup_config:
        logger.info(f"Using specific environment setup for stage '{stage}'.")
        stage_cmds = env_setup_config.get(stage, [])
    else:
        # Determine category and fall back to it
        category = None
        if stage in SIMULATION_STAGES:
            category = "simulation"
        elif stage in POSTPROCESSING_STAGES:
            category = "postprocessing"

        if category and category in env_setup_config:
            logger.info(f"No specific setup for '{stage}', falling back to category '{category}'.")
            stage_cmds = env_setup_config.get(category, [])
        else:
            logger.warning(f"No environment setup found for stage '{stage}' or category '{category}'.")

    if not stage_cmds:
        return []

    # Substitute variables
    processed_cmds = []
    for cmd in stage_cmds:
        try:
            processed_cmds.append(cmd.format(**format_dict))
        except KeyError as e:
            logger.error(f"Variable {e} is used in a command but not defined in env_setup.env_variables.")
            raise ValueError(f"Undefined variable in command: {cmd}")

    return processed_cmds

def apply_config_defaults(config):
    """
    Apply config defaults from env_setup.config_defaults to the main config.
    
    This function recursively merges defaults into the config, but only for
    keys that are not already present (config values take precedence).
    
    Args:
        config (dict): The main configuration dictionary
        
    Returns:
        dict: The updated configuration with defaults applied
    """
    env_setup_config = config.get("env_setup", {})
    config_defaults = env_setup_config.get("config_defaults", {})
    
    if not config_defaults:
        logger.debug("No config_defaults found in env_setup. Skipping default application.")
        return config
    
    # Create a copy to avoid modifying the original
    updated_config = config.copy()
    
    # Apply defaults recursively
    def merge_defaults(target_dict, defaults_dict):
        """Recursively merge defaults into target, with target taking precedence."""
        for key, default_value in defaults_dict.items():
            if key not in target_dict:
                # Key doesn't exist in target, use default
                target_dict[key] = default_value
                logger.debug(f"Applied default for '{key}': {default_value}")
            elif isinstance(default_value, dict) and isinstance(target_dict[key], dict):
                # Both are dicts, recurse
                merge_defaults(target_dict[key], default_value)
            # If key exists and is not a dict, keep the existing value (target precedence)
    
    merge_defaults(updated_config, config_defaults)
    
    return updated_config

def substitute_config_variables(config):
    """
    Substitute variables in config values using patterns like {VAR_NAME}.
    
    This function looks for patterns like {directories.software_dir} or {common.account}
    and substitutes them with values from the same config. It avoids interfering with:
    - Double-brace templates like {{XQCUT}} (left untouched)
    - JSON structure braces
    
    Args:
        config (dict): The configuration dictionary
        
    Returns:
        dict: The configuration with variables substituted
    """
    import re
    import copy
    
    # Work with a deep copy to avoid modifying the original
    result_config = copy.deepcopy(config)
    
    # Pattern to match single-brace variables like {section.key} but not {{template}}
    # This uses negative lookbehind and lookahead to avoid double braces
    var_pattern = re.compile(r'(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_.]*)\}(?!\})')
    
    def substitute_in_value(value, path=""):
        """Recursively substitute variables in a value."""
        if isinstance(value, str):
            # Only process strings that contain single-brace patterns
            if '{' in value and not value.startswith('{{'):
                def replace_variable(match):
                    var_path = match.group(1)
                    
                    # Split the path (e.g., "directories.software_dir" -> ["directories", "software_dir"])
                    path_parts = var_path.split('.')
                    
                    # Navigate through the config to find the value
                    current = result_config
                    try:
                        for part in path_parts:
                            current = current[part]
                        return str(current)
                    except (KeyError, TypeError):
                        logger.debug(f"Could not resolve variable reference: {{{var_path}}} at {path}")
                        return match.group(0)  # Return the original if we can't resolve it
                
                return var_pattern.sub(replace_variable, value)
            else:
                return value
        elif isinstance(value, dict):
            return {k: substitute_in_value(v, f"{path}.{k}" if path else k) for k, v in value.items()}
        elif isinstance(value, list):
            return [substitute_in_value(item, f"{path}[{i}]") for i, item in enumerate(value)]
        else:
            return value
    
    # Apply substitutions with multiple passes to handle chained references
    max_iterations = 3  # Reduced iterations since we're being more targeted
    for iteration in range(max_iterations):
        old_result = copy.deepcopy(result_config)
        result_config = substitute_in_value(result_config)
        
        if result_config == old_result:
            # No more substitutions made
            break
    else:
        logger.warning("Maximum iterations reached during variable substitution. Some references may be unresolved.")
    
    return result_config

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
    Ensures git working directory is clean, commits changes if needed, tags the version, and logs the config.
    
    Args:
        config: The configuration dictionary
        config_path: Path to the original config file
        software_repo_path: Path to the software repository root
        force_commit: Whether to force a commit and tag even if no changes or success marker exists.
        
    Returns:
        tuple: (success: bool, processed_config_path: Path or None) 
               - success: True on success, False on failure
               - processed_config_path: Path to the saved processed config file
    """
    try:
        if "campaign" not in config:
            logger.error("Configuration missing 'campaign' field. Cannot proceed with git operations.")
            return False, None
        if "dataset" not in config:
            logger.error("Configuration missing 'dataset' field. Cannot proceed with git operations.")
            return False, None
        if "version" not in config:
            logger.error("Configuration missing 'version' field. Cannot proceed with git operations.")
            return False, None

        # --- 1. Get current branch name (for logging, not enforcement) ---
        try:
            current_branch_cmd = ["git", "-C", str(software_repo_path), "rev-parse", "--abbrev-ref", "HEAD"]
            current_branch_name = subprocess.check_output(current_branch_cmd, text=True, cwd=software_repo_path).strip()
            logger.info(f"Currently on git branch: {current_branch_name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get current git branch in {software_repo_path}: {e.stderr}")
            return False, None

        # --- 2. Determine output directory paths ---
        output_version_dir = get_version_directory(config) # Will now include campaign
        commit_success_file = output_version_dir / GIT_COMMIT_SUCCESS_FILE
        
        output_version_dir.mkdir(parents=True, exist_ok=True) # Ensure it exists early

        # Determine original config filename and new paths
        original_config_filename = Path(config_path).name
        config_snapshot_dir = output_version_dir / "configs"
        logged_config_path = config_snapshot_dir / original_config_filename

        # --- 3. Git Working Directory Check and Commit Logic (ALWAYS check) ---
        logger.info(f"Checking git working directory status in {software_repo_path}")
        
        status_cmd = ["git", "-C", str(software_repo_path), "status", "--porcelain"]
        process = subprocess.run(status_cmd, capture_output=True, text=True)
        if process.returncode != 0:
            logger.error(f"Git status check failed in {software_repo_path}: {process.stderr}")
            return False, None
        
        committed_this_run = False
        if not process.stdout.strip() and not force_commit: # Working directory is clean
            logger.info(f"Git working directory is clean. Proceeding with stage execution.")
        else: # There are uncommitted changes, or force_commit is true
            if process.stdout.strip():
                # Show the user what changes exist
                logger.info("Uncommitted changes detected:")
                status_verbose_cmd = ["git", "-C", str(software_repo_path), "status", "--short"]
                status_output = subprocess.run(status_verbose_cmd, capture_output=True, text=True)
                for line in status_output.stdout.strip().split('\n'):
                    if line.strip():
                        logger.info(f"  {line}")
                
                # Prompt for commit message
                try:
                    commit_message = input("\nEnter commit message (or press Enter for default): ").strip()
                    if not commit_message:
                        commit_message = (f"Auto-commit for campaign '{config['campaign']}', "
                                          f"dataset '{config['dataset']}', version '{config['version']}'")
                except (EOFError, KeyboardInterrupt):
                    logger.error("User cancelled commit. Cannot proceed with uncommitted changes.")
                    return False, None
                except Exception:
                    # Handle non-interactive environments (e.g., SLURM jobs)
                    logger.warning("Non-interactive environment detected. Using default commit message.")
                    commit_message = (f"Auto-commit for campaign '{config['campaign']}', "
                                      f"dataset '{config['dataset']}', version '{config['version']}'")
                
                # Add and commit changes
                add_cmd = ["git", "-C", str(software_repo_path), "add", "."]
                subprocess.run(add_cmd, check=True, capture_output=True)
                
                commit_cmd = ["git", "-C", str(software_repo_path), "commit", "-m", commit_message]
                subprocess.run(commit_cmd, check=True, capture_output=True)
                logger.info(f"Git commit successful: '{commit_message}'")
                committed_this_run = True
            elif force_commit:
                # Force commit case - create an empty commit if needed
                commit_message = (f"Forced commit for campaign '{config['campaign']}', "
                                  f"dataset '{config['dataset']}', version '{config['version']}'")
                commit_cmd = ["git", "-C", str(software_repo_path), "commit", "--allow-empty", "-m", commit_message]
                subprocess.run(commit_cmd, check=True, capture_output=True)
                logger.info(f"Forced empty commit successful: '{commit_message}'")
                committed_this_run = True

        # --- 4. Check if version already processed (after git commit) ---
        if not force_commit and commit_success_file.exists() and not committed_this_run:
            logger.info(f"Success marker file {commit_success_file} exists and no new commits made. Skipping git tag and config re-save for this version.")
            # Check for the config in the new location
            if not logged_config_path.exists():
                 logger.warning(f"Git commit marker exists, but config snapshot {logged_config_path} not found. Re-logging config.")
                 config_snapshot_dir.mkdir(parents=True, exist_ok=True) # Ensure 'configs' subdir exists
                 with open(logged_config_path, 'w') as f_out:
                    yaml.dump(config, f_out, default_flow_style=False, sort_keys=False)
                 logger.info(f"Config snapshot saved to {logged_config_path}")
            return True, logged_config_path

        git_hash_cmd = ["git", "-C", str(software_repo_path), "rev-parse", "HEAD"]
        current_git_hash = subprocess.check_output(git_hash_cmd, text=True, cwd=software_repo_path).strip()
        logger.info(f"Current Git HEAD for {software_repo_path}: {current_git_hash}")

        # --- 5. Save Config Snapshot and Success Marker ---
        config_snapshot_dir.mkdir(parents=True, exist_ok=True) # Ensure 'configs' subdir exists before writing
        
        # Create a clean copy of the config without internal env_setup data
        clean_config = {k: v for k, v in config.items() if k != 'env_setup'}
        
        with open(logged_config_path, 'w') as f_out:
            yaml.dump(clean_config, f_out, default_flow_style=False, sort_keys=False)
        logger.info(f"Full configuration snapshot saved to {logged_config_path}")
        
        relative_config_path_for_marker = Path("configs") / original_config_filename
        with open(commit_success_file, 'w') as f_marker:
            f_marker.write(f"Commit successful at {datetime.datetime.now()}\n"
                           f"Git Branch: {current_branch_name}\n"
                           f"Git Hash: {current_git_hash}\n"
                           f"Config: {relative_config_path_for_marker}\n") # Use relative path here
        logger.info(f"Git commit success marker created at {commit_success_file}")
        return True, logged_config_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Git operation failed during script execution: {e.cmd}")
        stderr_output = e.stderr.decode('utf-8').strip() if isinstance(e.stderr, bytes) else e.stderr.strip() if e.stderr else ""
        stdout_output = e.stdout.decode('utf-8').strip() if isinstance(e.stdout, bytes) else e.stdout.strip() if e.stdout else ""
        if stdout_output:
            logger.error(f"Stdout: {stdout_output}")
        if stderr_output:
            logger.error(f"Stderr: {stderr_output}")
        return False, None
    except Exception as e:
        logger.error(f"An unexpected error occurred during git_commit_and_log_config: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, None

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