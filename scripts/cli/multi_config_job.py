#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-config SLURM job submission for ColliderML.
Allows combining multiple stage configs into a single large SLURM job for >256 node discounts.
"""

import logging
import math
from pathlib import Path
from simple_slurm import Slurm

# Import utilities and JobSubmitter
import cli_utils
from job_submission import JobSubmitter

logger = logging.getLogger(__name__)


def validate_multi_config_compatibility(configs):
    """
    Validate that configs can be combined into one SLURM job.
    
    Args:
        configs (list): List of configuration dictionaries
        
    Raises:
        ValueError: If configs are incompatible
    """
    if len(configs) < 2:
        raise ValueError("Multi-config mode requires at least 2 configurations")
    
    # Check all stages are simulation stages (require shifter)
    stages = [config.get("stage") for config in configs]
    for stage in stages:
        if stage not in cli_utils.SHIFTER_STAGES:
            raise ValueError(
                f"Multi-config jobs only support simulation stages that use shifter. "
                f"Stage '{stage}' is not in SHIFTER_STAGES: {cli_utils.SHIFTER_STAGES}"
            )
    
    # Check all configs use same container
    containers = [config.get("common", {}).get("container") for config in configs]
    if len(set(containers)) > 1:
        raise ValueError(
            f"All configs must use the same container. Found: {set(containers)}"
        )
    
    if not containers[0]:
        raise ValueError("Container not specified in common.container for configs")
    
    # Warn about different time limits
    time_limits = [config.get("job_config", {}).get("time_limit") for config in configs]
    if len(set(time_limits)) > 1:
        logger.warning(
            f"Configs have different time limits: {set(time_limits)}. "
            f"Will use maximum: {max(time_limits)}"
        )
    
    # Warn about different QOS
    qos_values = [config.get("job_config", {}).get("qos") for config in configs]
    if len(set(qos_values)) > 1:
        logger.warning(
            f"Configs have different QOS values: {set(qos_values)}. "
            f"Will use first: {qos_values[0]}"
        )
    
    logger.info(f"✓ Validated {len(configs)} configs for multi-config job compatibility")


def calculate_task_ranges(submitters):
    """
    Calculate PROCID range for each stage.
    
    Args:
        submitters (list): List of JobSubmitter instances
        
    Returns:
        list: List of tuples (start_procid, end_procid) for each stage
    """
    ranges = []
    offset = 0
    for submitter in submitters:
        total_tasks = submitter.compute_total_tasks()
        ranges.append((offset, offset + total_tasks))
        offset += total_tasks
    return ranges


def calculate_procid_offset_expr(offset):
    """
    Generate bash expression for PROCID remapping.
    
    Args:
        offset (int): The offset to subtract from SLURM_PROCID
        
    Returns:
        str: Bash expression like "$((SLURM_PROCID - 100))"
    """
    if offset == 0:
        return "$SLURM_PROCID"
    return f"$((SLURM_PROCID - {offset}))"


class MultiConfigJobSubmitter:
    """
    Handles submission of multiple stage configs as a single SLURM job.
    
    Stages run in parallel with separate srun commands, each with PROCID remapping
    to ensure each stage sees local PROCID values (0-based).
    """
    
    def __init__(self, config_paths, config_dicts, git_repo_path=None, dry_run=False):
        """
        Initialize multi-config job submitter.
        
        Args:
            config_paths (list): List of paths to config files
            config_dicts (list): List of processed config dictionaries
            git_repo_path (Path): Path to git repository root
            dry_run (bool): If True, save scripts instead of submitting
        """
        self.config_paths = config_paths
        self.configs = config_dicts
        self.git_repo_path = git_repo_path
        self.dry_run = dry_run
        
        # Validate compatibility
        validate_multi_config_compatibility(self.configs)
        
        # Create individual JobSubmitters for each config
        # These handle directory setup, validation, and provide utilities
        logger.info(f"Creating JobSubmitters for {len(self.configs)} configs...")
        self.submitters = []
        for i, (path, config) in enumerate(zip(config_paths, config_dicts)):
            submitter = JobSubmitter(
                config_path=path,
                config_dict=config,
                git_repo_path=git_repo_path,
                dry_run=True  # Don't submit, just setup directories and metadata
            )
            self.submitters.append(submitter)
            logger.info(f"  Stage {i}: {config['stage']} ({submitter.compute_total_tasks()} tasks)")
        
        # Calculate combined resources
        self.calculate_combined_resources()
        
    def calculate_combined_resources(self):
        """Calculate total nodes, tasks, and PROCID ranges for all stages."""
        # Calculate PROCID ranges for each stage
        self.stage_ranges = calculate_task_ranges(self.submitters)
        
        # Calculate total tasks
        self.total_tasks = sum(submitter.compute_total_tasks() for submitter in self.submitters)
        
        # Calculate total nodes needed (sum across all stages)
        self.total_nodes = sum(submitter.n_nodes for submitter in self.submitters)
        
        # Get combined time limit (use maximum)
        time_limits = [config.get("job_config", {}).get("time_limit") for config in self.configs]
        self.time_limit = max(time_limits)
        
        # Get common settings from first config (already validated to be same)
        self.account = self.configs[0]["common"]["account"]
        self.container = self.configs[0]["common"]["container"]
        self.qos = self.configs[0]["job_config"]["qos"]
        
        # Use first config's version directory for job logs
        self.log_dir = self.submitters[0].log_dir.parent / "combined_job_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if self.dry_run:
            self.dry_run_dir = self.submitters[0].version_dir / "dry_run_combined"
            self.dry_run_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Combined resources: {self.total_nodes} nodes, {self.total_tasks} tasks")
        logger.info(f"PROCID ranges: {self.stage_ranges}")
    
    def _build_parallel_commands(self, slurm):
        """
        Generate parallel srun commands for all stages.
        
        Each stage gets its own srun command with PROCID remapping, running in background.
        All commands are followed by a wait to ensure all stages complete.
        
        Args:
            slurm (Slurm): The Slurm job object to add commands to
        """
        # Add basic environment setup
        slurm.add_cmd(r"cd $HOME")
        slurm.add_cmd("export SLURM_CPU_BIND=\"cores\"")
        
        logger.info("Building parallel srun commands for each stage...")
        
        # Build parallel srun command for each stage
        for stage_idx, (submitter, (start_procid, end_procid)) in enumerate(zip(self.submitters, self.stage_ranges)):
            config = submitter.config
            stage = config["stage"]
            num_tasks = end_procid - start_procid
            
            logger.info(f"  Stage {stage_idx} ({stage}): PROCID {start_procid}-{end_procid-1} ({num_tasks} tasks)")
            
            # Get run ID expression (supports run_list if specified)
            run_id_expr, run_ids_setup_cmd = submitter.get_run_id_expr_global()
            
            # Build the stage command using existing utilities
            command_info = cli_utils.build_stage_command(
                config=config,
                config_path=submitter.config_path,
                stage_script_path=submitter.get_stage_script(),
                output_dir=submitter.run_dir,
                execution_mode="distributed_slurm",
                slurm_procid_offset=start_procid,  # This will be used for offset calculation
                run_id_expr=run_id_expr
            )
            
            # Build srun command with PROCID filtering and remapping
            # We need to remap SLURM_PROCID to local stage PROCID
            offset_expr = calculate_procid_offset_expr(start_procid)
            
            # Build the conditional wrapper
            # Each task checks if it belongs to this stage's PROCID range
            condition = f"[ \\$SLURM_PROCID -ge {start_procid} ] && [ \\$SLURM_PROCID -lt {end_procid} ]"
            
            # Remap SLURM_PROCID to local stage PROCID
            if start_procid > 0:
                procid_remap = f"STAGE_PROCID={offset_expr}"
            else:
                procid_remap = "STAGE_PROCID=$SLURM_PROCID"
            
            # Get environment setup commands
            env_setup_cmds = command_info["env_setup_commands"]
            
            # Build the payload that each task will execute
            # Replace SLURM_PROCID references in python command with STAGE_PROCID
            python_cmd = command_info["python_command"].replace("SLURM_PROCID", "STAGE_PROCID")
            
            # Construct the complete command
            if run_ids_setup_cmd:
                # Handle run_list case: setup array, remap, then run
                setup_clean = run_ids_setup_cmd.replace(" && \\", "").strip()
                payload = f"{setup_clean} && {procid_remap} && {python_cmd}"
            else:
                payload = f"{procid_remap} && {python_cmd}"
            
            # Build the conditional + payload
            full_payload = f"if {condition}; then {payload}; fi"
            
            # Build srun command
            # Use --ntasks to specify how many tasks for this stage
            # Note: We run a single srun with all tasks, but each task conditionally executes
            srun_options = "--exact --kill-on-bad-exit=0 -u"
            
            # Add comment for clarity
            slurm.add_cmd(f"# Stage {stage_idx}: {stage} (PROCID {start_procid}-{end_procid-1})")
            
            # Build the srun command with shifter
            # Environment setup goes inside shifter
            env_setup_str = " && ".join(env_setup_cmds)
            if env_setup_str:
                shifter_payload = f"{env_setup_str} && {full_payload}"
            else:
                shifter_payload = full_payload
            
            # Escape quotes for the bash -c command
            shifter_payload_escaped = shifter_payload.replace('"', '\\"')
            
            # Build complete srun command with background execution
            srun_cmd = f'srun {srun_options} --ntasks={num_tasks} shifter bash -c "{shifter_payload_escaped}" &'
            slurm.add_cmd(srun_cmd)
        
        # Add wait to ensure all stages complete before job finishes
        slurm.add_cmd("")
        slurm.add_cmd("# Wait for all stages to complete")
        slurm.add_cmd("wait")
        slurm.add_cmd("echo 'All stages completed'")
    
    def submit(self):
        """
        Submit the combined multi-node job.
        
        Returns:
            list: List containing single job ID (or dry-run marker)
        """
        logger.info(f"Submitting combined multi-node job for {len(self.configs)} stages")
        
        # Parse dependencies from configs (use first config's dependency if any)
        dependency_kw = self.submitters[0].parse_dependency_kw()
        
        # Create single Slurm job object
        slurm_kwargs = dict(
            job_name=f"colliderML_combined_{len(self.configs)}stages",
            account=self.account,
            qos=self.qos,
            time=self.time_limit,
            nodes=self.total_nodes,
            ntasks=self.total_tasks,
            cpus_per_task=1,  # Will be determined by SLURM based on ntasks and nodes
            constraint="cpu",
            output=str(self.log_dir / "job_combined_%j.out"),
            error=str(self.log_dir / "job_combined_%j.err")
        )
        
        if dependency_kw is not None:
            slurm_kwargs["dependency"] = dependency_kw
            logger.info(f"Job dependency: {dependency_kw}")
        
        slurm = Slurm(**slurm_kwargs)
        
        # Add shifter image to SBATCH directives (required for simulation stages)
        setattr(slurm.namespace, "image", f"{self.container} --module=cvmfs")
        
        # Build parallel commands
        self._build_parallel_commands(slurm)
        
        # Submit or save dry-run
        if self.dry_run:
            script_path = self._save_batch_script(slurm, "job_combined_multiconfig.sh")
            logger.info(f"Dry-run: Saved combined batch script to {script_path}")
            return ["DRY_RUN_COMBINED_JOB"]
        else:
            job_id = slurm.sbatch(
                shell="/bin/bash",
                job_file=f"{self.log_dir}/job_combined_multiconfig.sh",
                convert=False
            )
            logger.info(
                f"Submitted combined multi-config job with ID {job_id} "
                f"spanning {self.total_nodes} nodes and {self.total_tasks} tasks"
            )
            return [job_id]
    
    def _save_batch_script(self, slurm, script_name):
        """Save the batch script for dry-run mode."""
        script_path = self.dry_run_dir / script_name
        script_content = slurm.script(shell="/bin/bash", convert=False)
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        return script_path
    
    def submit_validation_jobs(self, job_ids):
        """
        Submit separate validation job for each stage.
        
        Args:
            job_ids (list): List containing the combined job ID
            
        Returns:
            list: List of validation job IDs (or dry-run markers)
        """
        if not job_ids:
            logger.warning("No job IDs provided, skipping validation job submission")
            return []
        
        validation_ids = []
        
        logger.info(f"Submitting validation jobs for {len(self.submitters)} stages")
        
        # Submit validation job for each stage
        for stage_idx, submitter in enumerate(self.submitters):
            config = submitter.config
            stage = config.get("stage")
            
            # Skip if no validation config for this stage
            if not config.get("validation_config"):
                logger.info(f"  Stage {stage_idx} ({stage}): No validation config, skipping")
                continue
            
            logger.info(f"  Stage {stage_idx} ({stage}): Submitting validation job")
            
            # Call the submitter's validation job method with combined job dependency
            stage_validation_ids = submitter.submit_validation_jobs(job_ids)
            validation_ids.extend(stage_validation_ids)
        
        logger.info(f"Submitted {len(validation_ids)} validation jobs total")
        return validation_ids

