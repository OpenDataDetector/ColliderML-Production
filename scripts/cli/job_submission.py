# colliderml_dev/scripts/cli/job_submission.py
import os
import math
from pathlib import Path
import yaml
from simple_slurm import Slurm
import datetime
import logging

# Import common utilities
import cli_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobSubmitter:
    """Handles SLURM job submission for ColliderML pipeline stages"""
    
    # Separate stage categories
    SIMULATION_STAGES = ["new_generation", "generation", "merge_smear", "simulation", "digitization"]
    POSTPROCESSING_STAGES = ["build_tracks", "build_hits", "build_particles"]
    VALID_STAGES = SIMULATION_STAGES + POSTPROCESSING_STAGES
    
    def __init__(self, config_path, dry_run=False, run_range=None, run_list=None):
        """Initialize with YAML config"""
        self.dry_run = dry_run
        self.config_path = config_path
        self.run_range = run_range
        self.run_list = run_list
        
        # Both run_range and run_list can't be specified simultaneously
        if run_range and run_list:
            raise ValueError("Cannot specify both run_range and run_list")
        
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        self.validate_config()
        self.calculate_job_distribution()
        self.setup_directories()
        
    def validate_config(self):
        """Validate configuration"""
        if self.config["stage"] not in self.VALID_STAGES:
            raise ValueError(f"Invalid stage. Must be one of {self.VALID_STAGES}")
            
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
        return self.config["stage"] in self.SIMULATION_STAGES
    
    def is_postprocessing_stage(self):
        """Check if current stage is a postprocessing stage"""
        return self.config["stage"] in self.POSTPROCESSING_STAGES
    
    def get_stage_script(self):
        """Get the appropriate script for current stage using cli_utils"""
        try:
            return str(cli_utils.get_stage_script_path(self.config))
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
        slurm = Slurm(
            job_name=f"colliderML_{self.config['stage']}_{node_idx}",
            account=common_cfg["account"],
            qos=job_cfg["qos"],
            time=job_cfg["time_limit"],
            nodes=1,
            ntasks_per_node=1 if is_monolithic else job_cfg["runs_per_node"],
            cpus_per_task=job_cfg.get("max_cores", 128) if is_monolithic else job_cfg.get("max_cores", 128)//job_cfg["runs_per_node"],
            constraint="cpu",
            output=str(self.log_dir / f"job_{node_idx}_%j.out"),
            error=str(self.log_dir / f"job_{node_idx}_%j.err")
        )
        
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
        """Add commands for simulation stages, using env_setup from config if present."""
        common_cfg = self.config["common"]
        
        # Add environment setup with container
        slurm.add_cmd(r"cd $HOME")
        slurm.add_cmd("export SLURM_CPU_BIND=\"cores\"")
        
        # Start srun command with container
        srun_options = "--exact"
        if not is_monolithic:
            # If distributed, we need to preserve SLURM_PROCID for run calculation
            pass
        
        slurm.add_cmd(f"srun {srun_options} -u shifter --image={common_cfg['container']} --module=cvmfs bash -c \"")
        
        # Use env_setup from config, or fallback to legacy hardcoded setup if not present
        env_setup_cmds = self.get_env_setup()
        if env_setup_cmds:
            for cmd in env_setup_cmds:
                slurm.add_cmd(cmd + " && \\")
        else:
            # Fallback: legacy hardcoded setup (for backward compatibility)
            slurm.add_cmd("cd /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase && \\")
            slurm.add_cmd("export ATLAS_LOCAL_ROOT_BASE=\$PWD && \\")
            slurm.add_cmd("source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh && \\")
            slurm.add_cmd("cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software && \\")
            slurm.add_cmd(f"source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_107 x86_64-el9-gcc13-opt && \\")
            slurm.add_cmd("source OtherLibraries/dd4hep-custom/bin/thisdd4hep.sh && \\")
            slurm.add_cmd(f"source acts/build/python/setup.sh && \\")
            slurm.add_cmd("source colliderml_env/bin/activate && \\")
        
        # Prepare stage-specific command 
        if is_monolithic:
            # For monolithic mode, don't use SLURM_PROCID, provide version_dir instead
            version_dir = cli_utils.get_version_directory(self.config)
            cmd = (f"python {self.get_stage_script()} "
                  f"--config {self.config_path} "
                  f"--output-base-dir {version_dir}")
            
            # If stage requires specific monolithic arguments, add them
            if self.config["stage"] == "new_generation":
                # For madgraph generation, we might pass additional parameters 
                # specific to that monolithic process
                pass
        else:
            # For distributed mode, use SLURM_PROCID
            cmd = (f"python {self.get_stage_script()} "
                  f"--config {self.config_path} "
                  f"--output {self.run_dir} "
                  f"--output-subdir \$(({previous_runs} + SLURM_PROCID)) "
                  f"--seed {self.config['dataset']}_{self.config['version']}_run\$(({previous_runs} + SLURM_PROCID))")
        
        # Close container command
        cmd += "\""
        slurm.add_cmd(cmd)
    
    def _add_postprocessing_commands(self, slurm, previous_runs, is_monolithic=False):
        """Add commands for postprocessing stages, using env_setup from config if present."""
        # Use srun to properly parallelize tasks with all environment setup inside
        srun_options = "--exact"
        slurm.add_cmd(f"srun {srun_options} bash -c \"")
        
        # Use env_setup from config, or fallback to legacy hardcoded setup if not present
        env_setup_cmds = self.get_env_setup()
        if env_setup_cmds:
            for cmd in env_setup_cmds:
                slurm.add_cmd(cmd + " && \\")
        else:
            # Fallback: legacy hardcoded setup (for backward compatibility)
            slurm.add_cmd("cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software && \\")
            slurm.add_cmd("eval \\\"\\$(conda shell.bash hook)\\\" && \\")
            slurm.add_cmd("conda activate collider-env && \\")
        
        # Add the Python command based on mode
        if is_monolithic:
            # For monolithic mode, we pass the whole version directory
            version_dir = cli_utils.get_version_directory(self.config)
            cmd = (f"python {self.get_stage_script()} "
                  f"--config {self.config_path} "
                  f"--output-base-dir {version_dir}")
        else:
            # For distributed mode, use SLURM_PROCID
            cmd = (f"python {self.get_stage_script()} "
                  f"--config {self.config_path} "
                  f"--chunk-index \$(({previous_runs} + SLURM_PROCID))")
        
        # Close the quotation for srun
        cmd += "\""
        slurm.add_cmd(cmd)
    
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
    
    submitter = JobSubmitter(args.config, dry_run=args.dry_run, 
                             run_range=args.run_range, run_list=args.run_list)
    job_ids = submitter.submit_jobs()
    validation_ids = submitter.submit_validation_jobs(job_ids)
    
    if args.dry_run:
        logger.info(f"Dry run completed. Batch scripts saved in: {submitter.dry_run_dir}")
    else:
        logger.info(f"Submitted {len(job_ids)} production jobs with IDs: {job_ids}")
        logger.info(f"Submitted {len(validation_ids)} validation jobs with IDs: {validation_ids}")

if __name__ == "__main__":
    main() 