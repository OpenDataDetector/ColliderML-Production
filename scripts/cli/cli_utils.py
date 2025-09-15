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
MADGRAPH_STAGES = ["madgraph_init", "madgraph_generation"]
SIMULATION_STAGES = MADGRAPH_STAGES + ["pythia_generation", "merge_smear", "simulation", "digitization"]
POSTPROCESSING_STAGES = [
    "build_tracks",
    "build_tracker_hits",
    "build_particles",
    "build_manifest",
    "convert_all",
]
VALID_STAGES = SIMULATION_STAGES + POSTPROCESSING_STAGES

# Define which stages need shifter container (subset of simulation stages)
# madgraph_init and madgraph_generation run on host environment and don't need shifter
SHIFTER_STAGES = ["pythia_generation", "merge_smear", "simulation", "digitization"]

# Stage to script mappings
STAGE_SCRIPT_MAP = {
    # Simulation scripts
    "madgraph_init": "simulation/madgraph_init.py",
    "madgraph_generation": "simulation/madgraph_gen.py",
    "pythia_generation": "simulation/pythia_gen.py",
    "merge_smear": "simulation/merge_and_smear.py",
    "simulation": "simulation/ddsim_run.py",
    "digitization": "simulation/digi_and_reco.py",
    
    # Postprocessing scripts
    "build_tracks": "postprocessing/convert_tracks.py",
    "build_tracker_hits": "postprocessing/convert_digihits.py",
    "build_particles": "postprocessing/convert_particles.py",
    "build_manifest": "postprocessing/build_manifest.py",
    "convert_all": "postprocessing/convert_all.py",
}

def get_env_setup_cmds(config):
    """
    Gets and processes environment setup commands from the configuration.
    Simple version: just substitute variables using string.format()
    
    Args:
        config (dict): The main configuration dictionary with 'env_setup' section.

    Returns:
        list: A list of processed environment setup commands.
    """
    env_setup_config = config.get("env_setup", {})
    if not env_setup_config:
        logger.warning("No env_setup section found in configuration.")
        return []

    # Get environment variables (preserve original case for formatting)
    env_variables = env_setup_config.get("env_variables", {})
    format_dict = {key: value for key, value in env_variables.items() if isinstance(value, str)}

    stage = config.get("stage")
    if not stage:
        logger.warning("Cannot determine environment setup: 'stage' not in config.")
        return []

    # Find commands: stage-specific first, then category fallback
    stage_cmds = []
    if stage in env_setup_config:
        stage_cmds = env_setup_config.get(stage, [])
    else:
        # Determine category and fall back to it
        category = None
        if stage in MADGRAPH_STAGES:
            category = "madgraph"
        elif stage in SIMULATION_STAGES:
            category = "simulation"
        elif stage in POSTPROCESSING_STAGES:
            category = "postprocessing"

        if category and category in env_setup_config:
            stage_cmds = env_setup_config.get(category, [])

    if not stage_cmds:
        logger.warning(f"No environment setup found for stage '{stage}'.")
        return []

    # Simple variable substitution
    processed_cmds = []
    for cmd in stage_cmds:
        try:
            processed_cmds.append(cmd.format(**format_dict))
        except KeyError as e:
            logger.error(f"Variable {e} used in command but not defined in env_variables: {cmd}")
            raise ValueError(f"Undefined variable in command: {cmd}")

    return processed_cmds

def build_stage_command(config, config_path, stage_script_path, output_dir, output_subdir="0", 
                       execution_mode="interactive", slurm_procid_offset=0, run_id_expr=None):
    """
    Build the complete command setup for running a stage, handling both simulation and postprocessing stages.
    This centralizes environment setup logic to ensure consistency between interactive and batch modes.
    
    Shifter containers are only used for stages that require them (defined in SHIFTER_STAGES).
    madgraph_generation runs in the host environment without shifter.
    
    Args:
        config (dict): The configuration dictionary
        config_path (str/Path): Path to the config file
        stage_script_path (str/Path): Path to the stage script
        output_dir (str/Path): Output directory path
        output_subdir (str): Output subdirectory (or "all" for interactive)
        execution_mode (str): "interactive", "distributed_slurm", or "monolithic_slurm"
        slurm_procid_offset (int): Offset for SLURM process ID calculation
    
    Returns:
        dict: Dictionary containing:
            - "use_shifter": bool - Whether to use shifter container
            - "shifter_command": str - Shifter command prefix (if applicable)
            - "env_setup_commands": list - Environment setup commands
            - "python_command": str - The main Python command
            - "full_command": str - Complete command ready to execute
    """
    stage = config["stage"]
    is_simulation = stage in SIMULATION_STAGES
    is_postprocessing = stage in POSTPROCESSING_STAGES
    
    if not (is_simulation or is_postprocessing):
        raise ValueError(f"Unknown stage category for stage: {stage}")
    
    # Get environment setup commands
    env_setup_cmds = get_env_setup_cmds(config)
    
    # Determine if we need shifter (only specific stages need it)
    use_shifter = stage in SHIFTER_STAGES
    
    # Build the main Python command
    python_cmd_parts = [
        "python",
        str(stage_script_path),
        "--config", str(config_path)
    ]
    
    # Add stage-specific arguments
    if execution_mode == "monolithic_slurm":
        # For monolithic mode, pass the output directory (same as other modes)
        python_cmd_parts.extend(["--output", str(output_dir)])
    elif is_simulation:
        # For simulation stages in distributed mode
        if execution_mode == "interactive":
            python_cmd_parts.extend(["--output", str(output_dir)])
            if output_subdir is not None:
                python_cmd_parts.extend(["--output-subdir", str(output_subdir)])
                if output_subdir != "all":
                    # Add seed for specific run
                    dataset = config.get("dataset", "unknown")
                    version = config.get("version", "unknown")
                    python_cmd_parts.extend([
                        "--seed", f"{dataset}_{version}_run{output_subdir}"
                    ])
        else:  # distributed_slurm
            # Allow custom run id expression (e.g., mapping from a provided run list)
            run_idx_expr = run_id_expr if run_id_expr is not None else f"\$(({slurm_procid_offset} + SLURM_PROCID))"
            python_cmd_parts.extend([
                "--output", str(output_dir),
                "--output-subdir", run_idx_expr
            ])
            dataset = config.get("dataset", "unknown")
            version = config.get("version", "unknown")
            python_cmd_parts.extend([
                "--seed", f"{dataset}_{version}_run{run_idx_expr}"
            ])
    else:  # postprocessing stages
        if execution_mode == "interactive":
            if output_subdir is not None and output_subdir != "all":
                python_cmd_parts.extend(["--chunk-index", str(output_subdir)])
            # For "all" case in postprocessing, we might need special handling
        else:  # distributed_slurm
            # Allow custom run id expression (e.g., mapping from a provided run list)
            run_idx_expr = run_id_expr if run_id_expr is not None else f"\$(({slurm_procid_offset} + SLURM_PROCID))"
            python_cmd_parts.extend([
                "--chunk-index", run_idx_expr
            ])
    
    python_command = " ".join(python_cmd_parts)
    
    # If G4 warning filtering is enabled, wrap the command to filter stdout.
    # This is the only robust way to catch the G4Exception warnings, which
    # are unexpectedly routed to stdout in the production environment.
    filter_g4_warnings = config.get("filter_g4_warnings", False)
    if filter_g4_warnings and stage == "simulation":
        logger.info("Applying shell-level stdout filtering for Geant4 warnings.")
        filter_pattern = "G4Exception|deltaMass|Primary particle PDG"
        # 'set -o pipefail' ensures that if the python script fails, the job fails.
        # The grep command filters out the unwanted Geant4 warning lines from stdout.
        python_command = f"set -o pipefail; {python_command} | grep -v -E '{filter_pattern}'"

    # Build the complete command based on execution mode and shifter usage
    if execution_mode == "interactive":
        if use_shifter:
            # Interactive mode with shifter
            common_cfg = config.get("common", {})
            container = common_cfg.get("container")
            if not container:
                raise ValueError(f"Stage '{stage}' requires shifter container but 'common.container' not found in config")
            
            shifter_cmd = f"shifter --image={container} --module=cvmfs bash -c \""
            
            # Combine env setup and python command inside shifter
            inner_commands = env_setup_cmds + [python_command]
            inner_command_str = " && ".join(inner_commands)
            
            full_command = shifter_cmd + inner_command_str + "\""
            
            return {
                "use_shifter": True,
                "shifter_command": shifter_cmd,
                "env_setup_commands": env_setup_cmds,
                "python_command": python_command,
                "full_command": full_command
            }
        else:
            # Interactive mode without shifter
            full_command_list = env_setup_cmds + [python_command]
            full_command = " && ".join(full_command_list)
            
            return {
                "use_shifter": False,
                "shifter_command": None,
                "env_setup_commands": env_setup_cmds,
                "python_command": python_command,
                "full_command": full_command
            }
    
    else:  # SLURM modes
        if use_shifter:
            # SLURM with shifter (stages that need containers)
            common_cfg = config.get("common", {})
            container = common_cfg.get("container")
            if not container:
                raise ValueError(f"Stage '{stage}' requires shifter container but 'common.container' not found in config")
            
            srun_options = "--exact --kill-on-bad-exit=0"
            shifter_cmd = f"srun {srun_options} -u shifter --image={container} --module=cvmfs bash -c \""
            
            # Environment setup commands are added inside the shifter container
            # Python command is also inside
            return {
                "use_shifter": True,
                "shifter_command": shifter_cmd,
                "env_setup_commands": env_setup_cmds,
                "python_command": python_command,
                "full_command": None  # Will be built by the SLURM job submitter
            }
        else:
            # SLURM without shifter (postprocessing stages)
            srun_options = "--exact --kill-on-bad-exit=0"
            srun_cmd = f"srun {srun_options} bash -c \""
            
            return {
                "use_shifter": False,
                "shifter_command": srun_cmd,
                "env_setup_commands": env_setup_cmds,
                "python_command": python_command,
                "full_command": None  # Will be built by the SLURM job submitter
            }

def apply_config_defaults(config):
    """
    Apply config defaults from env_setup.config_defaults to the main config.
    Recursively merges defaults, with config values taking precedence.
    
    Args:
        config (dict): The main configuration dictionary
        
    Returns:
        dict: The updated configuration with defaults applied
    """
    env_setup_config = config.get("env_setup", {})
    config_defaults = env_setup_config.get("config_defaults", {})
    
    if not config_defaults:
        return config
    
    updated_config = config.copy()
    
    def merge_defaults(target_dict, defaults_dict):
        """Recursively merge defaults into target, with target taking precedence."""
        for key, default_value in defaults_dict.items():
            if key not in target_dict:
                target_dict[key] = default_value
            elif isinstance(default_value, dict) and isinstance(target_dict[key], dict):
                merge_defaults(target_dict[key], default_value)
    
    merge_defaults(updated_config, config_defaults)
    return updated_config

def substitute_config_variables(config):
    """
    Apply final variable substitution to config after defaults are merged.
    Only handles {section.key} references within the config itself.
    
    Args:
        config (dict): The configuration dictionary
        
    Returns:
        dict: The configuration with variables substituted
    """
    import re
    import copy
    
    result_config = copy.deepcopy(config)
    
    # Pattern to match {section.key} but not {{template}} 
    var_pattern = re.compile(r'(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_.]+)\}(?!\})')
    
    def substitute_in_value(value):
        if isinstance(value, str) and '{' in value and not value.startswith('{{'):
            def replace_variable(match):
                var_path = match.group(1)
                path_parts = var_path.split('.')
                current = result_config
                try:
                    for part in path_parts:
                        current = current[part]
                    return str(current)
                except (KeyError, TypeError):
                    logger.debug(f"Could not resolve config reference: {{{var_path}}}")
                    return match.group(0)
            return var_pattern.sub(replace_variable, value)
        elif isinstance(value, dict):
            return {k: substitute_in_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [substitute_in_value(item) for item in value]
        else:
            return value
    
    return substitute_in_value(result_config)

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