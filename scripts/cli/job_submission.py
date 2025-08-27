# colliderml_dev/scripts/cli/job_submission.py
import os
import math
from pathlib import Path
import yaml
from simple_slurm import Slurm
import datetime
import logging
import sys

# Import common utilities
import cli_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobSubmitter:
    """Handles SLURM job submission for ColliderML pipeline stages"""
    
    # Separate stage categories
    SIMULATION_STAGES = ["madgraph_generation", "pythia_generation", "merge_smear", "simulation", "digitization"]
    POSTPROCESSING_STAGES = ["build_tracks", "build_hits", "build_particles"]
    VALID_STAGES = SIMULATION_STAGES + POSTPROCESSING_STAGES
    
    def __init__(self, config_path=None, config_dict=None, git_repo_path=None, dry_run=False, run_range=None, run_list=None):
        """Initialize with YAML config file or pre-processed config dict"""
        self.dry_run = dry_run
        self.git_repo_path = git_repo_path
        self.run_range = run_range
        self.run_list = run_list
        
        # Both run_range and run_list can't be specified simultaneously
        if run_range and run_list:
            raise ValueError("Cannot specify both run_range and run_list")
        
        # Accept either config_path OR config_dict
        if config_dict is not None:
            self.config = config_dict
            self.config_path = config_path or "processed_config"  # Use provided path or placeholder
        elif config_path is not None:
            self.config_path = config_path
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            raise ValueError("Must provide either config_path or config_dict")
            
        self.validate_config()
        self.calculate_job_distribution()
        self.setup_directories()
        
    def validate_config(self):
        """Validate configuration"""
        if self.config["stage"] not in cli_utils.VALID_STAGES:
            raise ValueError(f"Invalid stage. Must be one of {cli_utils.VALID_STAGES}")
            
        required_fields = ["job_config", "common", "version", "dataset"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required config section: {field}")
                
        job_required = ["n_runs", "runs_per_node", "time_limit", "qos"]
        for field in job_required:
            if field not in self.config["job_config"]:
                raise ValueError(f"Missing required job_config field: {field}")
    
    def calculate_job_distribution(self):
        """Calculate number of nodes needed based on runs"""
        runs_per_node = self.config["job_config"]["runs_per_node"]
        
        if self.run_list:
            # Using a specific list of runs
            self.run_ids = sorted(self.run_list)
            n_runs = len(self.run_ids)
            self.n_nodes = math.ceil(n_runs / runs_per_node)
            self.start_run = 0  # Not used with run_list, but set for completeness
            
        elif self.run_range:
            # Using a range of runs
            start_run, end_run = self.run_range
            n_runs = end_run - start_run
            self.n_nodes = math.ceil(n_runs / runs_per_node)
            self.start_run = start_run
            self.run_ids = None  # Not using specific run IDs
            
        else:
            # Default: process all runs from 0 to n_runs-1
            n_runs = self.config["job_config"]["n_runs"]
            self.n_nodes = math.ceil(n_runs / runs_per_node)
            self.start_run = 0
            self.run_ids = None  # Not using specific run IDs
    
    def setup_directories(self):
        """Create necessary directories with new structure"""
        # Use cli_utils to get and create directories
        directories = cli_utils.create_necessary_directories(self.config)
        
        # Store directories for later use
        self.version_dir = directories["version_dir"]
        self.run_dir = directories["run_dir"]
        self.log_dir = directories["log_dir"]
        self.validation_dir = directories["validation_dir"]
        
        # Setup dry run directory if needed
        if self.dry_run:
            self.dry_run_dir = self.version_dir / "dry_run"
            self.dry_run_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.dry_run_dir = None
    
    def get_run_id(self, node_idx, process_idx):
        """Calculate run ID from node and process indices"""
        if self.run_list:
            # Using specific run IDs from list
            list_idx = (node_idx * self.config["job_config"]["runs_per_node"]) + process_idx
            if list_idx < len(self.run_ids):
                return self.run_ids[list_idx]
            return None
        elif self.run_range:
            # Using range of runs
            start_run, end_run = self.run_range
            run_id = start_run + (node_idx * self.config["job_config"]["runs_per_node"]) + process_idx
            if run_id >= end_run:
                return None
        else:
            # Default: sequential runs from 0
            run_id = (node_idx * self.config["job_config"]["runs_per_node"]) + process_idx
            if run_id >= self.config["job_config"]["n_runs"]:
                return None
        return run_id
        
    def get_run_dir(self, run_id):
        """Get directory for specific run"""
        return self.run_dir / f"{run_id}"
    
    def is_simulation_stage(self):
        """Check if current stage is a simulation stage"""
        return self.config["stage"] in cli_utils.SIMULATION_STAGES
    
    def is_postprocessing_stage(self):
        """Check if current stage is a postprocessing stage"""
        return self.config["stage"] in cli_utils.POSTPROCESSING_STAGES
    
    def get_stage_script(self):
        """Get the appropriate script for current stage using cli_utils"""
        try:
            return str(cli_utils.get_stage_script_path(self.config, self.git_repo_path))
        except (ValueError, FileNotFoundError) as e:
            logger.error(f"Error finding script path for stage {self.config['stage']}: {e}")
            raise
    
    def get_env_setup(self):
        """Get the environment setup commands for the current stage from config."""
        env_setup = self.config.get("env_setup", [])
        if isinstance(env_setup, list):
            return env_setup
        elif isinstance(env_setup, dict):
            return env_setup.get(self.config["stage"], env_setup.get("default", []))
        else:
            return []
    
    def create_slurm_job(self, node_idx):
        """Create Slurm job object for given node"""
        job_cfg = self.config["job_config"]
        common_cfg = self.config["common"]
        
        # Determine if this is a monolithic job from execution_mode
        is_monolithic = job_cfg.get("execution_mode") == "monolithic_slurm"
        
        # Create basic SLURM job configuration
        # Optional dependency: allow chaining on a prior SLURM job id
        dependency_kw = None
        depends_on = self.config["job_config"].get("depends_on")
        if depends_on:
            # Supports a single job id or a list of job ids. Enforce 'afterok'.
            try:
                job_ids = []
                if isinstance(depends_on, (list, tuple, set)):
                    for jid in depends_on:
                        s = str(jid).strip()
                        if s:
                            job_ids.append(s)
                else:
                    s = str(depends_on).strip()
                    if s:
                        job_ids.append(s)
                if job_ids:
                    dependency_kw = {"afterok": job_ids}
                    logger.info(f"Applying SLURM dependency afterok on: {job_ids}")
            except Exception:
                logger.warning(f"Invalid depends_on value in job_config: {depends_on}")

        slurm_kwargs = dict(
            job_name=f"colliderML_{self.config['stage']}_{node_idx}",
            account=common_cfg["account"],
            qos=job_cfg["qos"],
            time=job_cfg["time_limit"],
            nodes=1,
            ntasks_per_node=1 if is_monolithic else job_cfg["runs_per_node"],
            cpus_per_task=job_cfg.get("max_cores", 256) if is_monolithic else job_cfg.get("max_cores", 256)//job_cfg["runs_per_node"],
            constraint="cpu",
            output=str(self.log_dir / f"job_{node_idx}_%j.out"),
            error=str(self.log_dir / f"job_{node_idx}_%j.err")
        )
        if dependency_kw is not None:
            slurm_kwargs["dependency"] = dependency_kw
        slurm = Slurm(**slurm_kwargs)
        
        # Calculate run offset based on run range or normal distribution
        if self.run_range:
            previous_runs = self.run_range[0] + (node_idx * self.config["job_config"]["runs_per_node"])
        else:
            previous_runs = node_idx * self.config["job_config"]["runs_per_node"]
        
        # Different setup for simulation vs postprocessing stages
        if self.is_simulation_stage():
            self._add_simulation_commands(slurm, previous_runs, is_monolithic)
        else:
            self._add_postprocessing_commands(slurm, previous_runs, is_monolithic)
        
        return slurm
    
    def _add_simulation_commands(self, slurm, previous_runs, is_monolithic=False):
        """Add commands for simulation stages using shared command builder for consistency with interactive mode."""
        # Add basic SLURM environment setup
        slurm.add_cmd(r"cd $HOME")
        slurm.add_cmd("export SLURM_CPU_BIND=\"cores\"")
        
        # Use shared command builder for consistency
        execution_mode = "monolithic_slurm" if is_monolithic else "distributed_slurm"
        output_dir = cli_utils.get_version_directory(self.config) if is_monolithic else self.run_dir
        
        try:
            command_info = cli_utils.build_stage_command(
                config=self.config,
                config_path=self.config_path,
                stage_script_path=self.get_stage_script(),
                output_dir=output_dir,
                execution_mode=execution_mode,
                slurm_procid_offset=previous_runs
            )
            
            # Add the shifter/srun command prefix
            slurm.add_cmd(command_info["shifter_command"])
            
            # Add environment setup commands
            for cmd in command_info["env_setup_commands"]:
                slurm.add_cmd(cmd + " && \\")
            
            # Add the python command and close the quoted section
            slurm.add_cmd(command_info["python_command"] + "\"")
            
        except Exception as e:
            logger.error(f"Error building simulation command: {e}")
            raise
    
    def _add_postprocessing_commands(self, slurm, previous_runs, is_monolithic=False):
        """Add commands for postprocessing stages using shared command builder for consistency with interactive mode."""
        # Use shared command builder for consistency
        execution_mode = "monolithic_slurm" if is_monolithic else "distributed_slurm"
        output_dir = cli_utils.get_version_directory(self.config) if is_monolithic else self.run_dir
        
        try:
            command_info = cli_utils.build_stage_command(
                config=self.config,
                config_path=self.config_path,
                stage_script_path=self.get_stage_script(),
                output_dir=output_dir,
                execution_mode=execution_mode,
                slurm_procid_offset=previous_runs
            )
            
            # Add the srun command prefix (no shifter for postprocessing)
            slurm.add_cmd(command_info["shifter_command"])
            
            # Add environment setup commands
            for cmd in command_info["env_setup_commands"]:
                slurm.add_cmd(cmd + " && \\")
            
            # Add the python command and close the quoted section
            slurm.add_cmd(command_info["python_command"] + "\"")
            
        except Exception as e:
            logger.error(f"Error building postprocessing command: {e}")
            raise
    
    def submit_jobs(self):
        """Submit all jobs for the stage"""
        job_ids = []
        
        for node_idx in range(self.n_nodes):
            slurm = self.create_slurm_job(node_idx)
            
            if self.dry_run:
                script_path = self.save_batch_script(
                    slurm, f"job_{node_idx}.sh"
                )
                logger.info(f"Saved batch script for node {node_idx}/{self.n_nodes} to {script_path}")
                job_ids.append(f"DRY_RUN_JOB_{node_idx}")
            else:
                job_id = slurm.sbatch(
                    shell="/bin/bash", 
                    job_file=f"{self.log_dir}/job_{node_idx}.sh",
                    convert=False)
                job_ids.append(job_id)
                logger.info(f"Submitted job for node {node_idx}/{self.n_nodes} with ID {job_id}")
        
        return job_ids
    
    def submit_validation_jobs(self, job_ids):        
        """Submit validation jobs dependent on stage jobs"""

        if self.config.get("validation_config", None) is None:
            logger.info("No validation config found, skipping validation jobs")
            return []

        validation_ids = []
        
        for node_idx, job_id in enumerate(job_ids):
            slurm = Slurm(
                job_name=f"validate_{self.config['stage']}_{node_idx}",
                account=self.config["common"]["account"],
                qos=self.config["validation_config"]["qos"],
                time=self.config["validation_config"]["time_limit"],
                dependency={"afterany": [job_id]} if not self.dry_run else None,
                output=str(self.validation_dir / f"validation_{node_idx}.out"),
                error=str(self.validation_dir / f"validation_{node_idx}.err"),
                constraint="cpu",
                nodes=1,
                ntasks=1
            )
            
            slurm.add_cmd("cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software")
            slurm.add_cmd("eval \"$(conda shell.bash hook)\"")
            slurm.add_cmd("conda activate collider-env")

            # Get validation script path
            try:
                # Check if validation script might be in a standard location
                base_script_path = Path(__file__).parent.parent
                validation_script_name = f"validation/validate_{self.config['stage']}.py"
                validation_script_full_path = str(base_script_path / validation_script_name)
                
                # Verify the script exists
                if not Path(validation_script_full_path).is_file():
                    raise FileNotFoundError(f"Validation script not found at {validation_script_full_path}")
            except Exception as e:
                logger.warning(f"Failed to locate validation script automatically: {e}")
                # Fall back to assuming standard location
                validation_script_full_path = f"colliderml_dev/scripts/validation/validate_{self.config['stage']}.py"

            cmd = (f"python {validation_script_full_path} "
                  f"--stage {self.config['stage']} "
                  f"--run-dir {self.run_dir} "
                  f"--node-idx {node_idx} "
                  f"--runs-per-node {self.config['job_config']['runs_per_node']}")
            
            slurm.add_cmd(cmd)
            
            if self.dry_run:
                script_path = self.save_batch_script(
                    slurm, f"validation_{node_idx}.sh"
                )
                logger.info(f"Saved validation batch script to {script_path}")
                validation_ids.append(f"DRY_RUN_VALIDATION_{node_idx}")
            else:
                validation_id = slurm.sbatch(
                    shell="/bin/bash", 
                    job_file=f"{self.validation_dir}/validation_{node_idx}.sh",
                    convert=False)
                validation_ids.append(validation_id)
                logger.info(f"Submitted validation job {validation_id} for production job {job_id}")
            
        return validation_ids

    def save_batch_script(self, slurm, script_name):
        """Save the batch script that would be submitted"""
        script_path = self.dry_run_dir / script_name
        
        # Get the complete script content
        script_content = slurm.script(shell="/bin/bash", convert=False)
            
        # Save to file
        with open(script_path, 'w') as f:
            f.write(script_content)
            
        return script_path

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Submit ColliderML pipeline jobs to SLURM.")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Don't submit jobs, just save batch scripts")
    parser.add_argument("--run-range", type=int, nargs=2, metavar=('START', 'END'),
                       help="Range of runs to process (START inclusive, END exclusive)")
    parser.add_argument("--run-list", type=int, nargs='+', metavar='RUN_ID',
                       help="List of specific run IDs to process")
    args = parser.parse_args()
    
    logger.info("Note: job_submission.py is being used directly. For most cases, consider using run_stage.py instead.")
    
    # When run directly, we must find the git repo and load all configs
    # to pass a complete config object to the submitter.
    git_repo_path = cli_utils.get_git_root(Path(__file__).resolve())
    if not git_repo_path:
        logger.error("Could not find git repository root. Exiting.")
        sys.exit(1)

    with open(args.config, 'r') as f_main:
        config = yaml.safe_load(f_main)

    env_config_path = Path(__file__).resolve().parent / "env_setup.yaml"
    if env_config_path.exists():
        with open(env_config_path, 'r') as f_env:
            config["env_setup"] = yaml.safe_load(f_env)

    # Write a temporary merged config to pass to the submitter.
    # This is necessary because JobSubmitter reads its config from a file path.
    temp_config_path = Path(args.config).with_suffix('.temp_merged_for_job_submission.yaml')
    with open(temp_config_path, 'w') as f_temp:
        yaml.dump(config, f_temp)
    
    submitter = JobSubmitter(str(temp_config_path), 
                             git_repo_path=git_repo_path,
                             dry_run=args.dry_run, run_range=args.run_range, 
                             run_list=args.run_list)
    job_ids = submitter.submit_jobs()
    validation_ids = submitter.submit_validation_jobs(job_ids)

    # Clean up the temporary file
    os.remove(temp_config_path)
    
    # Clean up temporary file
    os.remove(temp_config_path)

    if args.dry_run:
        logger.info(f"Dry run completed. Batch scripts saved in: {submitter.dry_run_dir}")
    else:
        logger.info(f"Submitted {len(job_ids)} production jobs with IDs: {job_ids}")
        logger.info(f"Submitted {len(validation_ids)} validation jobs with IDs: {validation_ids}")

if __name__ == "__main__":
    main() 