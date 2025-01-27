# colliderml_dev/batch/job_submission.py
import os
import math
from pathlib import Path
import yaml
from simple_slurm import Slurm
import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobSubmitter:
    """Handles SLURM job submission for ColliderML pipeline stages"""
    
    VALID_STAGES = ["generation", "merge_smear", "simulation", "digitization"]
    
    def __init__(self, config_path, dry_run=False, run_range=None):
        """Initialize with YAML config"""
        self.dry_run = dry_run
        self.config_path = config_path
        self.run_range = run_range
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
        if self.run_range:
            start_run, end_run = self.run_range
            n_runs = end_run - start_run
            runs_per_node = self.config["job_config"]["runs_per_node"]
            self.n_nodes = math.ceil(n_runs / runs_per_node)
            self.start_run = start_run
        else:
            n_runs = self.config["job_config"]["n_runs"]
            runs_per_node = self.config["job_config"]["runs_per_node"]
            self.n_nodes = math.ceil(n_runs / runs_per_node)
            self.start_run = 0
        
    def setup_directories(self):
        """Create necessary directories with new structure"""
        base_dir = Path(self.config["common"]["output_base_dir"])
        self.version_dir = base_dir / self.config["dataset"] / self.config["version"]
        self.run_dir = self.version_dir / "runs"
        self.log_dir = self.version_dir / "logs" / f"stage_{self.config['stage']}"
        self.validation_dir = self.version_dir / "validation" / f"stage_{self.config['stage']}"
        self.dry_run_dir = self.version_dir / "dry_run" if self.dry_run else None
        
        dirs_to_create = [self.version_dir, self.run_dir, self.log_dir, self.validation_dir]
        if self.dry_run:
            dirs_to_create.append(self.dry_run_dir)
            
        for d in dirs_to_create:
            d.mkdir(parents=True, exist_ok=True)
    
    def get_run_id(self, node_idx, process_idx):
        """Calculate run ID from node and process indices"""
        if self.run_range:
            start_run, end_run = self.run_range
            run_id = start_run + (node_idx * self.config["job_config"]["runs_per_node"]) + process_idx
            if run_id >= end_run:
                return None
        else:
            run_id = (node_idx * self.config["job_config"]["runs_per_node"]) + process_idx
            if run_id >= self.config["job_config"]["n_runs"]:
                return None
        return run_id
        
    def get_run_dir(self, run_id):
        """Get directory for specific run"""
        return self.run_dir / f"{run_id}"
    
    def create_slurm_job(self, node_idx):
        """Create Slurm job object for given node"""
        job_cfg = self.config["job_config"]
        common_cfg = self.config["common"]
        
        slurm = Slurm(
            job_name=f"colliderML_{self.config['stage']}_{node_idx}",
            account=common_cfg["account"],
            qos=job_cfg["qos"],
            time=job_cfg["time_limit"],
            nodes=1,
            ntasks_per_node=job_cfg["runs_per_node"],
            cpus_per_task=job_cfg.get("max_cores", 128)//job_cfg["runs_per_node"],
            constraint="cpu",
            output=str(self.log_dir / f"job_{node_idx}_%j.out"),
            error=str(self.log_dir / f"job_{node_idx}_%j.err")
        )
        
        # Add environment setup with container
        slurm.add_cmd(r"cd $HOME")
        slurm.add_cmd("export SLURM_CPU_BIND=\"cores\"")
        
        # Start srun command with container
        slurm.add_cmd(f"srun --exact -u shifter --image={common_cfg['container']} --module=cvmfs bash -c \"")
        
        # Environment setup inside container
        slurm.add_cmd("cd /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase && \\")
        slurm.add_cmd("export ATLAS_LOCAL_ROOT_BASE=\$PWD && \\")
        slurm.add_cmd("source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh && \\")
        slurm.add_cmd("cd /global/cfs/cdirs/m3443/usr/dtmurnane/Side_Work/ACTS && \\")
        slurm.add_cmd("source acts/CI/setup_cvmfs_lcg.sh && \\")
        slurm.add_cmd("source cml_env/bin/activate && \\")
        slurm.add_cmd("source build/python/setup.sh && \\")

        # Calculate run offset based on run range or normal distribution
        if self.run_range:
            previous_runs = self.run_range[0] + (node_idx * self.config["job_config"]["runs_per_node"])
        else:
            previous_runs = node_idx * self.config["job_config"]["runs_per_node"]
        
        # Prepare stage-specific command using bash arithmetic for run calculation
        cmd = (f"python {self.get_stage_script()} "
              f"--config {self.config_path} "
              f"--output {self.run_dir} "
              f"--output-subdir \$(({previous_runs} + SLURM_PROCID)) "  # Bash arithmetic for run calculation
              f"--seed {self.config['dataset']}_{self.config['version']}_run\$(({previous_runs} + SLURM_PROCID))")
        
        # Close container command
        cmd += "\""
        slurm.add_cmd(cmd)
        
        return slurm
    
    def get_stage_script(self):
        """Get the appropriate script for current stage"""
        stage_scripts = {
            "generation": "pythia_gen.py",
            "merge_smear": "merge_and_smear.py",
            "simulation": "ddsim_run.py",
            "digitization": "digi_and_reco.py"
        }
        return f"colliderml_dev/scripts/simulation/{stage_scripts[self.config['stage']]}"
    
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
            
            slurm.add_cmd("cd /global/cfs/cdirs/m3443/usr/dtmurnane/Side_Work/ACTS")
            slurm.add_cmd("eval \"$(conda shell.bash hook)\"")
            slurm.add_cmd("conda activate collider-env")

            # Add validation command with additional parameters
            cmd = (f"python colliderml_dev/scripts/validation/validate_{self.config['stage']}.py "
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
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Don't submit jobs, just save batch scripts")
    parser.add_argument("--run-range", type=int, nargs=2, metavar=('START', 'END'),
                       help="Range of runs to process (START inclusive, END exclusive)")
    args = parser.parse_args()
    
    submitter = JobSubmitter(args.config, dry_run=args.dry_run, run_range=args.run_range)
    job_ids = submitter.submit_jobs()
    validation_ids = submitter.submit_validation_jobs(job_ids)
    
    if args.dry_run:
        logger.info(f"Dry run completed. Batch scripts saved in: {submitter.dry_run_dir}")
    else:
        logger.info(f"Submitted {len(job_ids)} production jobs with IDs: {job_ids}")
        logger.info(f"Submitted {len(validation_ids)} validation jobs with IDs: {validation_ids}")

if __name__ == "__main__":
    main()