# usr/danieltm/ColliderML/software/colliderml_dev/scripts/cli/run_stage.py
import argparse
import yaml
import subprocess
import sys
import os
from pathlib import Path
import logging
import datetime
import json

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
    software_snapshot_base = base_dir / config["dataset"] / config["version"] / "software_snapshots"
    version_specific_software_dir = software_snapshot_base / "current"
    return version_specific_software_dir

def git_commit_and_log_config(config, config_path, git_repo_path, force_commit=False):
    """Commits the software state and logs the config. Returns True on success."""
    try:
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
            return True, logged_config_path

        logger.info(f"Attempting to git commit changes in {git_repo_path}")
        
        # Check for uncommitted changes
        status_cmd = ["git", "-C", str(git_repo_path), "status", "--porcelain"]
        process = subprocess.run(status_cmd, capture_output=True, text=True)
        if process.returncode != 0:
            logger.error(f"Git status check failed in {git_repo_path}: {process.stderr}")
            return False, None
        
        if not process.stdout.strip() and not force_commit:
            logger.info(f"No changes to commit in {git_repo_path}.")
        else:
            add_cmd = ["git", "-C", str(git_repo_path), "add", "."]
            subprocess.run(add_cmd, check=True)
            
            commit_message = f"Automatic commit for stage: {config['stage']}, dataset: {config['dataset']}, version: {config['version']}"
            commit_cmd = ["git", "-C", str(git_repo_path), "commit", "-m", commit_message]
            subprocess.run(commit_cmd, check=True)
            logger.info(f"Git commit successful in {git_repo_path}.")

        # Log the commit hash
        hash_cmd = ["git", "-C", str(git_repo_path), "rev-parse", "HEAD"]
        git_hash = subprocess.check_output(hash_cmd, text=True, cwd=git_repo_path).strip()
        logger.info(f"Current Git HEAD for {git_repo_path}: {git_hash}")

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
        return True, logged_config_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Git operation failed: {e}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        return False, None
    except Exception as e:
        logger.error(f"An unexpected error occurred during git commit and log: {e}")
        return False, None

def get_stage_script_path(config, git_repo_path):
    """Gets the absolute path to the stage script."""
    # This uses the JobSubmitter's method, which should be robust
    relative_script_path = JobSubmitter(config_path=config["config_path"], git_repo_path=git_repo_path).get_stage_script() 
    # job_submitter.get_stage_script() now returns an absolute path
    return Path(relative_script_path)

def run_validation(config, runs_dir):
    """Run validation and return results."""
    logger.info("Loading validation library...")
    validation_path = Path(__file__).parent.parent / 'simulation' / 'validation'
    sys.path.insert(0, str(validation_path))
    
    try:
        from validation_lib import validate_stage, load_validation_rules
        
        rules_path = validation_path / 'validation_rules.yaml'
        if not rules_path.exists():
            logger.error(f"Validation rules not found: {rules_path}")
            return None
            
        logger.info(f"Loading validation rules from: {rules_path}")
        validation_rules = load_validation_rules(rules_path)
        
        logger.info(f"Validating runs in: {runs_dir}")
        result = validate_stage(
            runs_dir=Path(runs_dir),
            stage=config['stage'],
            validation_rules=validation_rules
        )
        
        # Save report
        report_dir = Path(runs_dir).parent / 'validation_reports'
        report_dir.mkdir(exist_ok=True, parents=True)
        report_path = report_dir / f"validation_report_{config['stage']}.json"
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Validation report saved to: {report_path}")
        
        return result
        
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        logger.exception("Full traceback:")
        return None

def run_guardian(validation_result, config, runs_dir):
    """Run guardian decision logic and return decision."""
    if validation_result is None:
        logger.error("Cannot run guardian without validation results")
        return {'action': 'FAIL', 'exit_code': 1, 'reason': 'Validation failed to run'}
    
    logger.info("Loading error guardian...")
    validation_path = Path(__file__).parent.parent / 'simulation' / 'validation'
    
    try:
        from error_guardian import make_decision, load_guardian_policy
        
        policy_path = validation_path / 'guardian_policy.yaml'
        if not policy_path.exists():
            logger.error(f"Guardian policy not found: {policy_path}")
            return {'action': 'FAIL', 'exit_code': 1, 'reason': 'Policy file missing'}
            
        logger.info(f"Loading guardian policy from: {policy_path}")
        guardian_policy = load_guardian_policy(policy_path)
        
        retry_count = int(os.environ.get('SLURM_RESTART_COUNT', '0'))
        max_retries = guardian_policy.get('retry_policy', {}).get('max_retries', 3)
        
        logger.info(f"Making guardian decision (retry {retry_count}/{max_retries})...")
        decision = make_decision(
            validation_result=validation_result,
            runs_dir=Path(runs_dir),
            guardian_policy=guardian_policy,
            retry_count=retry_count,
            max_retries=max_retries
        )
        
        return decision
        
    except Exception as e:
        logger.error(f"Guardian decision failed with error: {e}")
        logger.exception("Full traceback:")
        return {'action': 'FAIL', 'exit_code': 1, 'reason': f'Guardian error: {e}'}

def run_interactive(config, config_path_arg, stage_script_path):
    """Runs the stage script interactively with integrated validation + guardian."""
    
    # Check if validation is enabled (default: true)
    validation_config = config.get('validation_config') or {}
    validation_enabled = validation_config.get('enabled', True)
    
    logger.info("=" * 80)
    logger.info(f"STAGE: {config['stage']}")
    logger.info(f"VALIDATION: {'ENABLED' if validation_enabled else 'DISABLED'}")
    logger.info("=" * 80)

    # Create necessary output directories
    debug_output_dir = config.get("debug_output_dir")
    if debug_output_dir:
        run_dir = Path(debug_output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using custom debug output directory from config: {run_dir}")
        output_subdir = None
    else:
        logger.info("Creating output directories for interactive run...")
        directories = cli_utils.create_necessary_directories(config)
        run_dir = directories["run_dir"]
        output_subdir = "all"

    # ===== PHASE 1: Execute Stage =====
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"PHASE 1: Executing stage - {config['stage']}")
    logger.info("=" * 80)
    logger.info(f"Using script: {stage_script_path}")
    
    stage_success = False
    stage_exit_code = 1
    
    try:
        command_info = cli_utils.build_stage_command(
            config=config,
            config_path=config_path_arg,
            stage_script_path=stage_script_path,
            output_dir=run_dir,
            output_subdir=output_subdir,
            execution_mode="interactive"
        )
        
        final_command_str = command_info["full_command"]
        
        if command_info["use_shifter"]:
            logger.info(f"Running stage '{config['stage']}' with shifter container.")
        else:
            logger.info(f"Running stage '{config['stage']}' in regular environment.")
        
        if not command_info["env_setup_commands"]:
            logger.warning("No environment setup commands found. Running script in current environment.")
        
        logger.info(f"Executing command: {final_command_str}")
        
        # Run stage (don't exit on error if validation is enabled)
        process = subprocess.run(final_command_str, shell=True, check=False)
        stage_exit_code = process.returncode
        
        if stage_exit_code == 0:
            logger.info(f"✓ Stage '{config['stage']}' completed successfully.")
            stage_success = True
        else:
            logger.warning(f"✗ Stage '{config['stage']}' failed with exit code: {stage_exit_code}")
            stage_success = False
            
    except FileNotFoundError:
        logger.error(f"Error: Stage script {stage_script_path} not found.")
        stage_exit_code = 1
        stage_success = False
    except Exception as e:
        logger.error(f"Error building/executing command: {e}")
        logger.exception("Full traceback:")
        stage_exit_code = 1
        stage_success = False
    
    # If validation is disabled, exit with stage result
    if not validation_enabled:
        logger.info(f"Validation disabled. Exiting with stage exit code: {stage_exit_code}")
        sys.exit(stage_exit_code)
    
    # ===== PHASE 2: Validate Outputs =====
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 2: Validating outputs")
    logger.info("=" * 80)
    
    validation_result = run_validation(config, run_dir)
    
    if validation_result:
        logger.info(f"Validation status: {validation_result.get('status', 'UNKNOWN')}")
        logger.info(f"Total runs: {validation_result.get('total_runs', 0)}")
        logger.info(f"Successful: {validation_result.get('successful_runs', 0)}")
        logger.info(f"Failed: {validation_result.get('failed_runs', 0)}")
        if validation_result.get('failed_runs', 0) > 0:
            logger.info(f"Failure rate: {validation_result.get('failure_rate', 0):.1f}%")
    
    # ===== PHASE 3: Guardian Decision =====
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 3: Error guardian decision")
    logger.info("=" * 80)
    
    decision = run_guardian(validation_result, config, run_dir)
    
    logger.info(f"Guardian action: {decision.get('action', 'UNKNOWN')}")
    logger.info(f"Reason: {decision.get('reason', 'No reason provided')}")
    logger.info(f"Exit code: {decision.get('exit_code', 1)}")
    
    # Display any actions taken
    if 'actions_taken' in decision and decision['actions_taken']:
        logger.info("Actions taken:")
        for action in decision['actions_taken']:
            logger.info(f"  - {action}")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"INTERACTIVE RUN COMPLETE - Exiting with code {decision['exit_code']}")
    logger.info("=" * 80)
    
    # Exit with guardian's decision
    sys.exit(decision['exit_code'])

def main():
    parser = argparse.ArgumentParser(description="Run a ColliderML data production stage.")
    parser.add_argument("configs", nargs='+', help="Path(s) to YAML configuration file(s). Multiple configs will be combined into one SLURM job.")
    parser.add_argument("--execution-mode", choices=["interactive", "monolithic_slurm", "distributed_slurm", "multi_node_slurm"], 
                        default=None, help="Override execution mode (optional). If not set, derived from config or defaults to distributed_slurm if ambiguous.")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run. For SLURM modes, saves batch scripts instead of submitting.")
    parser.add_argument("--force-commit", action="store_true", help="Force git commit even if no changes are detected or if a previous commit marker exists for this version.")
    parser.add_argument("--run-range", type=int, nargs=2, metavar=('START', 'END'),
                       help="Range of runs to process for distributed modes (START inclusive, END exclusive). Overrides config if set.")
    parser.add_argument("--run-list", type=int, nargs='+', metavar='RUN_ID',
                       help="List of specific run IDs to process for distributed modes. Overrides config if set.")
    parser.add_argument("--allow-master", action="store_true", help="Allow running on master/main branch.")
    
    args = parser.parse_args()

    # --- 1. Load all configs and apply processing ---
    env_config_path = Path(__file__).resolve().parent / "env_setup.yaml"
    
    configs = []
    for config_path in args.configs:
        try:
            config = cli_utils.load_and_process_config(config_path, env_config_path)
            logger.info(f"Successfully loaded and processed configuration from: {config_path}")
            configs.append(config)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration file {config_path}: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error processing configuration file {config_path}: {e}")
            sys.exit(1)
    
    if not env_config_path.exists():
        logger.warning(f"{env_config_path} not found. Ensure environment is configured if needed.")
    
    # For backward compatibility, use first config for git root detection
    config = configs[0]

    # --- 2. Determine Software Repo Path and Perform Git Commit ---
    git_repo_path = cli_utils.get_git_root(Path(__file__).resolve())
    if git_repo_path:
        # Check if we're on master/main branch 
        try:
            current_branch_cmd = ["git", "-C", str(git_repo_path), "rev-parse", "--abbrev-ref", "HEAD"]
            current_branch_name = subprocess.check_output(current_branch_cmd, text=True, cwd=git_repo_path).strip()
            
            if current_branch_name in ["master", "main"] and not args.allow_master:
                logger.error(f"Cannot run jobs on '{current_branch_name}' branch. Use a config branch or --allow-master flag.")
                sys.exit(1)
                
        except subprocess.CalledProcessError:
            logger.warning("Could not determine git branch, continuing...")

    # --- 3. Perform Git Commit (if in git repo) ---
    # For multi-config, we commit once and save each config to its respective directory
    processed_config_paths = []
    if git_repo_path:
        # Perform git commit once using first config
        success, first_processed_config_path = cli_utils.git_commit_and_log_config(
            configs[0], args.configs[0], git_repo_path, args.force_commit
        )
        if not success:
            logger.error("Failed to perform git commit and log configuration. Exiting.")
            sys.exit(1)
        processed_config_paths.append(first_processed_config_path)
        logger.info("Git commit successful.")
        logger.info(f"First config saved to: {first_processed_config_path}")
        
        # For additional configs (if multi-config), save config snapshots to their directories
        if len(configs) > 1:
            for i in range(1, len(configs)):
                cfg = configs[i]
                cfg_path = args.configs[i]
                # Save config snapshot without git commit (already done)
                output_version_dir = cli_utils.get_version_directory(cfg)
                output_version_dir.mkdir(parents=True, exist_ok=True)
                config_snapshot_dir = output_version_dir / "configs"
                config_snapshot_dir.mkdir(parents=True, exist_ok=True)
                logged_config_path = config_snapshot_dir / Path(cfg_path).name
                
                # Create clean copy without env_setup
                clean_config = {k: v for k, v in cfg.items() if k != 'env_setup'}
                with open(logged_config_path, 'w') as f_out:
                    yaml.dump(clean_config, f_out, default_flow_style=False, sort_keys=False)
                processed_config_paths.append(logged_config_path)
                logger.info(f"Config {i+1} saved to: {logged_config_path}")
    else:
        # No git repo, use original config paths
        processed_config_paths = args.configs

    # --- 4. Determine Execution Mode --- 
    # Priority: CLI arg > config file > default (e.g., distributed_slurm)
    execution_mode = args.execution_mode
    if not execution_mode:
        execution_mode = config.get("job_config", {}).get("execution_mode", "distributed_slurm")
    
    # For multi-config, enforce multi_node_slurm mode
    if len(configs) > 1:
        if execution_mode != "multi_node_slurm":
            logger.error(
                f"Multi-config jobs only support multi_node_slurm mode. "
                f"Current mode: {execution_mode}. Use --execution-mode multi_node_slurm or set in config."
            )
            sys.exit(1)
        logger.info(f"Multi-config mode: combining {len(configs)} configs into single SLURM job")
    
    logger.info(f"Effective execution mode: {execution_mode}")

    # --- 5. Execute based on mode ---
    if len(configs) > 1:
        # Multi-config mode - combine multiple stages into one job
        from multi_config_job import MultiConfigJobSubmitter
        
        try:
            multi_submitter = MultiConfigJobSubmitter(
                config_paths=[str(p) for p in processed_config_paths],
                config_dicts=configs,
                git_repo_path=git_repo_path,
                dry_run=args.dry_run
            )
            
            job_ids = multi_submitter.submit()
            
            if not args.dry_run and job_ids:
                logger.info(f"Submitted combined multi-config job with ID: {job_ids[0]}")
                
            # Submit validation jobs for each stage
            validation_ids = multi_submitter.submit_validation_jobs(job_ids)
            if validation_ids and not args.dry_run:
                logger.info(f"Submitted {len(validation_ids)} validation jobs")
            elif args.dry_run:
                logger.info(f"Dry run completed. Scripts saved in: {multi_submitter.dry_run_dir}")
                
        except Exception as e:
            logger.error(f"Error in multi-config job submission: {e}")
            import traceback
            logger.error(traceback.format_exc())
            sys.exit(1)
            
    elif execution_mode == "interactive":
        # Single-config interactive mode
        try:
            stage_script_path = cli_utils.get_stage_script_path(config, git_repo_path)
            # Use processed config if available, otherwise fall back to original
            config_to_use = processed_config_paths[0] if processed_config_paths else args.configs[0]
            run_interactive(config, config_to_use, stage_script_path)
        except (ValueError, FileNotFoundError) as e:
            logger.error(f"Failed to locate script for interactive execution: {e}")
            sys.exit(1)

    elif execution_mode in ["monolithic_slurm", "distributed_slurm", "multi_node_slurm"]:
        # For SLURM modes, use JobSubmitter
        try:
            # Pass run_range and run_list from CLI args to JobSubmitter if provided
            submitter_run_range = args.run_range if args.run_range else None
            submitter_run_list = args.run_list if args.run_list else None
            
            # Start with the processed config (which has env_setup defaults applied)
            effective_config = config.copy()
            
            # For monolithic_slurm mode, we ensure n_runs and runs_per_node are suitable
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
                    temp_config_path = Path(args.configs[0]).with_suffix('.monolithic.yaml')
                    with open(temp_config_path, 'w') as f:
                        yaml.dump(effective_config, f, default_flow_style=False, sort_keys=False)
                    logger.info(f"Created temporary modified config for monolithic mode: {temp_config_path}")
                    config_path_for_submitter = str(temp_config_path)
                else:
                    # Use processed config if available, otherwise fall back to original
                    config_path_for_submitter = str(processed_config_paths[0]) if processed_config_paths else args.configs[0]
            else:
                # Use processed config if available, otherwise fall back to original
                config_path_for_submitter = str(processed_config_paths[0]) if processed_config_paths else args.configs[0]

            # Initialize JobSubmitter with processed config
            job_submitter = JobSubmitter(
                config_path=config_path_for_submitter,
                config_dict=effective_config,  # Pass the processed config with defaults applied
                git_repo_path=git_repo_path,
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
                        logger.info(f"Submitted monolithic job with ID: {job_ids[0]} (with integrated validation)")
                    elif args.dry_run:
                        logger.info(f"Dry run for monolithic SLURM completed. Script saved in {job_submitter.dry_run_dir}")
                else:
                    logger.error("Configuration resulted in n_nodes > 1, which is inconsistent with monolithic_slurm mode.")
                    sys.exit(1)
            elif execution_mode == "distributed_slurm":
                logger.info(f"Preparing for distributed SLURM submission for stage: {config['stage']}")
                job_ids = job_submitter.submit_jobs()
                if not args.dry_run and job_ids:
                    logger.info(f"Submitted {len(job_ids)} distributed production jobs with integrated validation.")
                elif args.dry_run:
                    logger.info(f"Dry run for distributed SLURM completed. Scripts saved in: {job_submitter.dry_run_dir}")
            else:  # multi_node_slurm
                logger.info(f"Preparing for single multinode SLURM submission for stage: {config['stage']}")
                job_ids = job_submitter.submit_multi_node_job()
                if not args.dry_run and job_ids:
                    logger.info(f"Submitted multinode production job with integrated validation.")
                elif args.dry_run:
                    logger.info(f"Dry run for multinode SLURM completed. Script saved in: {job_submitter.dry_run_dir}")
                        
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
            import traceback
            logger.error(traceback.format_exc())
            sys.exit(1)
    else:
        logger.error(f"Unknown execution mode: {execution_mode}")
        sys.exit(1)

    logger.info(f"run_stage.py finished for stage '{config['stage']}'.")

if __name__ == "__main__":
    main() 