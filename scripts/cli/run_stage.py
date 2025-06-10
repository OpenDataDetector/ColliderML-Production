# usr/danieltm/ColliderML/software/colliderml_dev/scripts/cli/run_stage.py
import argparse
import yaml
import subprocess
import sys
import os
from pathlib import Path
import logging
import datetime

# Import shared utilities
import cli_utils

# Import JobSubmitter for SLURM job submission modes
from job_submission import JobSubmitter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG_FILE_NAME = "expanded_config.yaml"
GIT_COMMIT_SUCCESS_FILE = ".git_commit_success"

def get_git_root(start_path):
    """Traverse up to find the .git directory."""
    current_path = Path(start_path).resolve()
    while current_path != current_path.parent:
        if (current_path / ".git").is_dir():
            return current_path
        current_path = current_path.parent
    if (current_path / ".git").is_dir(): # Check root directory
        return current_path
    return None

def get_software_version_dir(config):
    """Constructs the path to the specific version directory for software snapshot."""
    base_dir = Path(config["common"]["output_base_dir"])
    # Use a dedicated subdirectory for software snapshots to avoid cluttering the version_dir
    # This keeps version_dir for actual data runs as defined in job_submission.py
    software_snapshot_base = base_dir / config["dataset"] / config["version"] / "software_snapshots"
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # version_specific_software_dir = software_snapshot_base / timestamp
    # For now, let's use a simpler, non-timestamped name. Can be refined.
    version_specific_software_dir = software_snapshot_base / "current"
    return version_specific_software_dir

def git_commit_and_log_config(config, config_path, software_repo_path, force_commit=False):
    """Commits the software state and logs the config. Returns True on success."""
    try:
        # Determine the output directory for this specific run based on config
        # This is where we'll save the config and the git commit marker
        # We need job_submission.py's setup_directories logic or similar here
        # For now, let's use a simplified version_dir from the config for placing commit log
        # This should ideally align with where job_submission.py sets up its main version_dir
        temp_submitter_for_paths = JobSubmitter(config_path=config_path, dry_run=True)
        output_version_dir = temp_submitter_for_paths.version_dir # This is dataset/version/
        
        commit_success_file = output_version_dir / GIT_COMMIT_SUCCESS_FILE

        if not force_commit and commit_success_file.exists():
            logger.info(f"Git commit for this version ({output_version_dir}) already performed. Skipping.")
            # Also ensure the config snapshot exists
            logged_config_path = output_version_dir / CONFIG_FILE_NAME
            if not logged_config_path.exists():
                 logger.warning(f"Git commit marker exists, but config snapshot {logged_config_path} not found. Re-logging config.")
                 with open(logged_config_path, 'w') as f_out:
                    yaml.dump(config, f_out, default_flow_style=False, sort_keys=False)
                 logger.info(f"Config snapshot saved to {logged_config_path}")
            return True

        logger.info(f"Attempting to git commit changes in {software_repo_path}")
        
        # Check for uncommitted changes
        status_cmd = ["git", "-C", str(software_repo_path), "status", "--porcelain"]
        process = subprocess.run(status_cmd, capture_output=True, text=True)
        if process.returncode != 0:
            logger.error(f"Git status check failed in {software_repo_path}: {process.stderr}")
            return False
        
        if not process.stdout.strip() and not force_commit:
            logger.info(f"No changes to commit in {software_repo_path}.")
        else:
            add_cmd = ["git", "-C", str(software_repo_path), "add", "."]
            subprocess.run(add_cmd, check=True)
            
            commit_message = f"Automatic commit for stage: {config['stage']}, dataset: {config['dataset']}, version: {config['version']}"
            commit_cmd = ["git", "-C", str(software_repo_path), "commit", "-m", commit_message]
            subprocess.run(commit_cmd, check=True)
            logger.info(f"Git commit successful in {software_repo_path}.")

        # Log the commit hash
        hash_cmd = ["git", "-C", str(software_repo_path), "rev-parse", "HEAD"]
        git_hash = subprocess.check_output(hash_cmd, text=True, cwd=software_repo_path).strip()
        logger.info(f"Current Git HEAD for {software_repo_path}: {git_hash}")

        # Save the expanded config to the run's output directory
        output_version_dir.mkdir(parents=True, exist_ok=True)
        logged_config_path = output_version_dir / CONFIG_FILE_NAME
        with open(logged_config_path, 'w') as f_out:
            yaml.dump(config, f_out, default_flow_style=False, sort_keys=False)
        logger.info(f"Full configuration snapshot saved to {logged_config_path}")
        
        # Create success marker file
        with open(commit_success_file, 'w') as f_marker:
            f_marker.write(f"Commit successful at {datetime.datetime.now()}\nGit Hash: {git_hash}\nConfig: {logged_config_path.name}\n")
        logger.info(f"Git commit success marker created at {commit_success_file}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Git operation failed: {e}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during git commit and log: {e}")
        return False

def get_stage_script_path(config, job_submitter_instance):
    """Gets the absolute path to the stage script."""
    # This uses the JobSubmitter's method, which should be robust
    relative_script_path = job_submitter_instance.get_stage_script() 
    # job_submitter.get_stage_script() now returns an absolute path
    return Path(relative_script_path)

def run_interactive(config, config_path_arg, stage_script_path):
    """Runs the stage script directly as a subprocess."""
    logger.info(f"Running stage '{config['stage']}' interactively.")
    logger.info(f"Using script: {stage_script_path}")

    # Basic command structure
    command = [
        sys.executable,  # Path to current python interpreter
        str(stage_script_path),
        "--config", str(config_path_arg)
    ]

    # Create necessary output directories first
    logger.info("Creating output directories for interactive run...")
    directories = cli_utils.create_necessary_directories(config)
    run_dir = directories["run_dir"]
    command.extend(["--output", str(run_dir)])
    command.extend(["--output-subdir", "all"])

    logger.info(f"Executing command: {' '.join(command)}")
    try:
        # Run the script without capturing output to show it in real-time
        process = subprocess.run(command, check=True)
        logger.info(f"Interactive stage '{config['stage']}' completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Interactive stage '{config['stage']}' failed with return code: {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        logger.error(f"Error: Stage script {stage_script_path} not found.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run a ColliderML data production stage.")
    parser.add_argument("config", help="Path to the YAML configuration file for the stage.")
    parser.add_argument("--execution-mode", choices=["interactive", "monolithic_slurm", "distributed_slurm"], 
                        default=None, help="Override execution mode (optional). If not set, derived from config or defaults to distributed_slurm if ambiguous.")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run. For SLURM modes, saves batch scripts instead of submitting.")
    parser.add_argument("--force-commit", action="store_true", help="Force git commit even if no changes are detected or if a previous commit marker exists for this version.")
    parser.add_argument("--run-range", type=int, nargs=2, metavar=('START', 'END'),
                       help="Range of runs to process for distributed modes (START inclusive, END exclusive). Overrides config if set.")
    parser.add_argument("--run-list", type=int, nargs='+', metavar='RUN_ID',
                       help="List of specific run IDs to process for distributed modes. Overrides config if set.")
    
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration from: {args.config}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file: {e}")
        sys.exit(1)

    # --- 1. Determine Software Repo Path and Perform Git Commit ---
    software_repo_path = cli_utils.get_git_root(Path(__file__).resolve())
    if not software_repo_path:
        logger.warning("Could not find .git directory. Skipping git commit and software snapshot.")
    else:
        logger.info(f"Identified software repository for commit: {software_repo_path}")
        if not cli_utils.git_commit_and_log_config(config, args.config, software_repo_path, args.force_commit):
            logger.error("Failed to perform git commit and log configuration. Exiting.")
            sys.exit(1)
        logger.info("Git commit and config logging successful.")

    # --- 2. Determine Execution Mode --- 
    # Priority: CLI arg > config file > default (e.g., distributed_slurm)
    execution_mode = args.execution_mode
    if not execution_mode:
        execution_mode = config.get("job_config", {}).get("execution_mode", "distributed_slurm")
    logger.info(f"Effective execution mode: {execution_mode}")

    # --- 3. Execute based on mode ---
    if execution_mode == "interactive":
        # For interactive mode, we don't need JobSubmitter at all
        try:
            stage_script_path = cli_utils.get_stage_script_path(config, software_repo_path)
            run_interactive(config, args.config, stage_script_path)
        except (ValueError, FileNotFoundError) as e:
            logger.error(f"Failed to locate script for interactive execution: {e}")
            sys.exit(1)

    elif execution_mode in ["monolithic_slurm", "distributed_slurm"]:
        # For SLURM modes, use JobSubmitter
        try:
            # Pass run_range and run_list from CLI args to JobSubmitter if provided
            submitter_run_range = args.run_range if args.run_range else None
            submitter_run_list = args.run_list if args.run_list else None
            
            # For monolithic_slurm mode, we ensure n_runs and runs_per_node are suitable
            effective_config = config.copy()
            if execution_mode == "monolithic_slurm":
                if "job_config" not in effective_config:
                    effective_config["job_config"] = {}
                if "n_runs" not in effective_config["job_config"] or effective_config["job_config"]["n_runs"] > 1:
                    logger.info("Setting n_runs=1 for monolithic_slurm mode (if not already 1)")
                    effective_config["job_config"]["n_runs"] = 1
                if "runs_per_node" not in effective_config["job_config"]:
                    logger.info("Setting runs_per_node=1 for monolithic_slurm mode")
                    effective_config["job_config"]["runs_per_node"] = 1
                
                # Write temporary config file if we made changes
                if effective_config != config:
                    temp_config_path = Path(args.config).with_suffix('.monolithic.yaml')
                    with open(temp_config_path, 'w') as f:
                        yaml.dump(effective_config, f, default_flow_style=False, sort_keys=False)
                    logger.info(f"Created temporary modified config for monolithic mode: {temp_config_path}")
                    config_path_for_submitter = str(temp_config_path)
                else:
                    config_path_for_submitter = args.config
            else:
                config_path_for_submitter = args.config

            # Initialize JobSubmitter
            job_submitter = JobSubmitter(
                config_path=config_path_for_submitter,
                dry_run=args.dry_run,
                run_range=submitter_run_range,
                run_list=submitter_run_list
            )
            logger.info("JobSubmitter initialized.")
            
            # Execute based on mode
            if execution_mode == "monolithic_slurm":
                logger.info(f"Preparing for monolithic SLURM submission for stage: {config['stage']}")
                # In future, we could add a dedicated submit_monolithic_job method to JobSubmitter
                if job_submitter.n_nodes == 1:
                    logger.info("Using standard submit_jobs for monolithic job submission (n_nodes=1).")
                    job_ids = job_submitter.submit_jobs()
                    if not args.dry_run and job_ids:
                        logger.info(f"Submitted monolithic job with ID: {job_ids[0]}")
                        # Validation jobs could still apply if defined
                        validation_ids = job_submitter.submit_validation_jobs(job_ids)
                        if validation_ids:
                            logger.info(f"Submitted validation job with ID: {validation_ids[0]}")
                    elif args.dry_run:
                        logger.info(f"Dry run for monolithic SLURM completed. Script saved in {job_submitter.dry_run_dir}")
                else:
                    logger.error("Configuration resulted in n_nodes > 1, which is inconsistent with monolithic_slurm mode.")
                    sys.exit(1)
            else:  # distributed_slurm
                logger.info(f"Preparing for distributed SLURM submission for stage: {config['stage']}")
                job_ids = job_submitter.submit_jobs()
                if not args.dry_run and job_ids:
                    validation_ids = job_submitter.submit_validation_jobs(job_ids)
                    logger.info(f"Submitted {len(job_ids)} distributed production jobs.")
                    if validation_ids:
                        logger.info(f"Submitted {len(validation_ids)} validation jobs.")
                elif args.dry_run:
                    logger.info(f"Dry run for distributed SLURM completed. Scripts saved in: {job_submitter.dry_run_dir}")
                    if job_submitter.config.get("validation_config"):
                        logger.info(f"Validation scripts saved in: {job_submitter.validation_dir}")
                        
            # Clean up temporary config file if created
            if execution_mode == "monolithic_slurm" and effective_config != config:
                try:
                    if os.path.exists(temp_config_path):
                        os.remove(temp_config_path)
                        logger.info(f"Removed temporary config file: {temp_config_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove temporary config file: {e}")
                    
        except Exception as e:
            logger.error(f"Error in SLURM job submission: {e}")
            sys.exit(1)
    else:
        logger.error(f"Unknown execution mode: {execution_mode}")
        sys.exit(1)

    logger.info(f"run_stage.py finished for stage '{config['stage']}'.")

if __name__ == "__main__":
    main() 