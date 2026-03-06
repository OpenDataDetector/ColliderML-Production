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

    def parse_dependency_kw(self):
        """Parse job_config.depends_on into Slurm dependency kw dict or None."""
        depends_on = self.config["job_config"].get("depends_on")
        if not depends_on:
            return None
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
                logger.info(f"Applying SLURM dependency afterok on: {job_ids}")
                return {"afterok": job_ids}
        except Exception:
            logger.warning(f"Invalid depends_on value in job_config: {depends_on}")
        return None

    def add_basic_slurm_env_setup(self, slurm):
        """Emit standard environment setup commands into the slurm script."""
        slurm.add_cmd(r"cd $HOME")
        slurm.add_cmd("export SLURM_CPU_BIND=\"cores\"")
    
    def add_validation_and_guardian_to_script(self, slurm, runs_dir, run_range=None, run_list=None):
        """
        Add validation + guardian phases to the batch script.
        Called after stage execution commands are added.
        
        Args:
            slurm: Slurm object to add commands to
            runs_dir: Path to the runs directory
            run_range: Tuple of (start, end) for run range validation (optional)
            run_list: List of specific run IDs to validate (optional)
        """
        validation_config = self.config.get('validation_config') or {}
        validation_enabled = validation_config.get('enabled', True)
        if not validation_enabled:
            logger.info("Validation disabled for this job")
            return
        
        validation_dir = Path(__file__).parent.parent / 'simulation' / 'validation'
        validation_script = validation_dir / 'run_validation.py'
        guardian_script = validation_dir / 'run_guardian.py'
        
        # Create validation reports directory
        report_dir = Path(runs_dir).parent / 'validation_reports'
        report_path = report_dir / f"validation_report_{self.config['stage']}.json"
        
        # Add phase separator
        slurm.add_cmd("")
        slurm.add_cmd("# ============================================================================")
        slurm.add_cmd("# PHASE 2: Validate Outputs")
        slurm.add_cmd("# ============================================================================")
        slurm.add_cmd(f"mkdir -p {report_dir}")
        
        stage_name = self.config["stage"]
        slurm.add_cmd(f"echo 'Validating outputs for stage {stage_name}...'")
        
        # Build validation command with run filtering
        validation_cmd = (
            f"python {validation_script} "
            f"--stage {stage_name} "
            f"--runs-dir {runs_dir} "
            f"--output {report_path} "
            f"--config {self.config_path}"
        )
        
        # Add run filtering arguments if provided
        if run_list is not None:
            run_list_str = " ".join(map(str, run_list))
            validation_cmd += f" --run-ids {run_list_str}"
        elif run_range is not None:
            start, end = run_range
            validation_cmd += f" --run-range {start} {end}"
        
        slurm.add_cmd(validation_cmd)
        
        # Add phase 3: Guardian decision
        slurm.add_cmd("")
        slurm.add_cmd("# ============================================================================")
        slurm.add_cmd("# PHASE 3: Error Guardian Decision")
        slurm.add_cmd("# ============================================================================")
        slurm.add_cmd(f"echo 'Running error guardian (retry $SLURM_RESTART_COUNT)...'")
        
        # Run guardian script
        guardian_cmd = (
            f"python {guardian_script} "
            f"--report {report_path} "
            f"--runs-dir {runs_dir}"
        )
        slurm.add_cmd(guardian_cmd)

    def compute_cpus_per_task(self, is_monolithic, tasks_for_node=None):
        job_cfg = self.config["job_config"]
        if is_monolithic:
            return job_cfg.get("max_cores", 256)
        runs_per_node = job_cfg["runs_per_node"]
        denom = tasks_for_node if tasks_for_node is not None else runs_per_node
        return max(1, job_cfg.get("max_cores", 256)//max(1, denom))

    def compute_tasks_for_node(self, node_idx):
        """Compute how many tasks (runs) should be launched on a given node."""
        runs_per_node = self.config["job_config"]["runs_per_node"]
        if self.run_list:
            total_runs = len(self.run_ids)
            started_before = node_idx * runs_per_node
            remaining = max(0, total_runs - started_before)
            return min(runs_per_node, remaining)
        if self.run_range:
            start_run, end_run = self.run_range
            total_runs = max(0, end_run - start_run)
            started_before = node_idx * runs_per_node
            remaining = max(0, total_runs - started_before)
            return min(runs_per_node, remaining)
        # Default: all runs from 0..n_runs-1
        total_runs = self.config["job_config"]["n_runs"]
        started_before = node_idx * runs_per_node
        remaining = max(0, total_runs - started_before)
        return min(runs_per_node, remaining)

    def compute_previous_runs(self, node_idx):
        """Compute the SLURM_PROCID offset for this node."""
        if self.run_range:
            return self.run_range[0] + (node_idx * self.config["job_config"]["runs_per_node"])
        return node_idx * self.config["job_config"]["runs_per_node"]

    def compute_total_tasks(self):
        """Compute total number of tasks to run across the job."""
        if self.run_list:
            return len(self.run_ids)
        if self.run_range:
            start_run, end_run = self.run_range
            return max(0, end_run - start_run)
        return int(self.config["job_config"]["n_runs"])

    def get_run_id_expr_for_node(self, node_idx, tasks_for_node, is_monolithic):
        """If using run_list in distributed mode, return (escaped expr, RUN_IDS setup cmd); else (None, None)."""
        if (not is_monolithic) and self.run_list:
            runs_per_node = self.config["job_config"]["runs_per_node"]
            start_index = node_idx * runs_per_node
            end_index = start_index + (tasks_for_node if tasks_for_node is not None else runs_per_node)
            node_run_ids = self.run_ids[start_index:end_index]
            run_ids_str = " ".join(str(r) for r in node_run_ids)
            run_ids_setup_cmd = f"RUN_IDS=({run_ids_str}) && \\"
            run_id_expr = r"\${RUN_IDS[\$SLURM_PROCID]}"
            if self.dry_run:
                logger.info(f"Node {node_idx} RUN_IDS: {node_run_ids}")
            return run_id_expr, run_ids_setup_cmd
        return None, None

    def get_run_id_expr_global(self):
        """For multi-node single job, return (escaped expr, RUN_IDS setup cmd) when using run_list; else (None, None)."""
        if self.run_list:
            run_ids_str = " ".join(str(r) for r in self.run_ids)
            run_ids_setup_cmd = f"RUN_IDS=({run_ids_str}) && \\"
            run_id_expr = r"\${RUN_IDS[\$SLURM_PROCID]}"
            if self.dry_run:
                logger.info(f"Global RUN_IDS: {self.run_ids}")
            return run_id_expr, run_ids_setup_cmd
        return None, None

    def _build_and_add_commands(self, slurm, is_monolithic, previous_runs, node_idx, tasks_for_node, is_postprocessing):
        """Common command assembly for both simulation and postprocessing stages."""
        execution_mode = "monolithic_slurm" if is_monolithic else "distributed_slurm"
        output_dir = (cli_utils.get_version_directory(self.config) if is_postprocessing and is_monolithic
                      else cli_utils.get_run_directory(self.config) if is_monolithic
                      else self.run_dir)
        run_id_expr, run_ids_setup_cmd = self.get_run_id_expr_for_node(node_idx, tasks_for_node, is_monolithic)

        command_info = cli_utils.build_stage_command(
            config=self.config,
            config_path=self.config_path,
            stage_script_path=self.get_stage_script(),
            output_dir=output_dir,
            execution_mode=execution_mode,
            slurm_procid_offset=previous_runs,
            run_id_expr=run_id_expr
        )
        use_shifter = command_info["use_shifter"]

        if use_shifter:
            # Shifter stages: put env setup and python inside shifter bash -c
            slurm.add_cmd(command_info["shifter_command"])
            if run_ids_setup_cmd:
                slurm.add_cmd(run_ids_setup_cmd)
            for cmd in command_info["env_setup_commands"]:
                slurm.add_cmd(cmd + " && \\")
            slurm.add_cmd(command_info["python_command"] + "\"")
        else:
            # Non-shifter (postprocessing): env setup OUTSIDE srun to avoid SLURM allocation issues
            for cmd in command_info["env_setup_commands"]:
                slurm.add_cmd(cmd)

            srun_options = "--exact --kill-on-bad-exit=0"
            # Always wrap payload in bash -c so shell features (e.g., $((...))) are evaluated per task
            if run_ids_setup_cmd:
                setup_clean = run_ids_setup_cmd.replace(" && \\", "").strip()
                payload = f"{setup_clean} && {command_info['python_command']}"
            else:
                payload = f"{command_info['python_command']}"
            # Quote payload, escaping any embedded quotes
            srun_cmd = (
                f"srun {srun_options} bash -c \"{payload}\""
            )
            slurm.add_cmd(srun_cmd)
    
    def get_runs_for_node(self, node_idx, tasks_for_node):
        """
        Get the list of actual run IDs that will be processed by this node.
        
        Args:
            node_idx: Node index
            tasks_for_node: Number of tasks on this node
            
        Returns:
            List of run IDs that this node will process
        """
        run_ids = []
        for process_idx in range(tasks_for_node):
            run_id = self.get_run_id(node_idx, process_idx)
            if run_id is not None:
                run_ids.append(run_id)
        return run_ids
    
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
    

    
    def create_slurm_job(self, node_idx):
        """Create Slurm job object for given node"""
        job_cfg = self.config["job_config"]
        common_cfg = self.config["common"]
        
        # Determine if this is a monolithic job from execution_mode
        is_monolithic = job_cfg.get("execution_mode") == "monolithic_slurm"
        runs_per_node = job_cfg["runs_per_node"]
        
        # Determine how many tasks should actually run on this node
        tasks_for_node = self.compute_tasks_for_node(node_idx)

        # Skip creating a job if there is nothing to run on this node
        if not is_monolithic and tasks_for_node <= 0:
            return None
        
        # Create basic SLURM job configuration
        # Optional dependency: allow chaining on a prior SLURM job id
        dependency_kw = self.parse_dependency_kw()
        
        slurm_kwargs = dict(
            job_name=f"colliderML_{self.config['stage']}_{node_idx}",
            account=common_cfg["account"],
            qos=job_cfg["qos"],
            time=job_cfg["time_limit"],
            nodes=1,
            ntasks_per_node=1 if is_monolithic else tasks_for_node,
            cpus_per_task=self.compute_cpus_per_task(is_monolithic, tasks_for_node),
            constraint="cpu",
            output=str(self.log_dir / f"job_{node_idx}_%j.out"),
            error=str(self.log_dir / f"job_{node_idx}_%j.err")
        )
        
        if dependency_kw is not None:
            slurm_kwargs["dependency"] = dependency_kw
        slurm = Slurm(**slurm_kwargs)
        
        # Add requeue and append flags for validation + guardian integration
        # For boolean flags, set to empty string to just add the flag without value
        # For flags with values, set the value directly
        setattr(slurm.namespace, "requeue", "")  # Boolean flag - no value
        setattr(slurm.namespace, "open-mode", "append")  # Flag with value
        
        # Add shifter image to SBATCH directives if needed (for performance)
        # Use setattr to add custom directive with multiple options on one line
        stage = self.config["stage"]
        if stage in cli_utils.SHIFTER_STAGES:
            container = common_cfg.get("container")
            if container:
                # Direct attribute injection: image value includes both --image and --module
                setattr(slurm.namespace, "image", f"{container} --module=cvmfs")
        
        # Calculate run offset based on run range or normal distribution
        previous_runs = self.compute_previous_runs(node_idx)
        
        # Different setup for simulation vs postprocessing stages
        if self.is_simulation_stage():
            self._add_simulation_commands(slurm, previous_runs, is_monolithic, node_idx, tasks_for_node)
        else:
            self._add_postprocessing_commands(slurm, previous_runs, is_monolithic, node_idx, tasks_for_node)
        
        return slurm
    
    def _add_simulation_commands(self, slurm, previous_runs, is_monolithic=False, node_idx=0, tasks_for_node=None):
        """Add commands for simulation stages using shared command builder for consistency with interactive mode."""
        self.add_basic_slurm_env_setup(slurm)
        
        # Add phase 1 header
        slurm.add_cmd("")
        slurm.add_cmd("# ============================================================================")
        slurm.add_cmd(f"# PHASE 1: Executing stage - {self.config['stage']}")
        slurm.add_cmd("# ============================================================================")
        slurm.add_cmd("echo \"SLURM_JOB_ID: $SLURM_JOB_ID\"")
        slurm.add_cmd("echo \"SLURM_RESTART_COUNT: $SLURM_RESTART_COUNT\"")
        slurm.add_cmd("")
        
        # Don't exit on stage failure - let validation/guardian handle it
        slurm.add_cmd("set +e  # Don't exit on error")
        
        try:
            self._build_and_add_commands(
                slurm=slurm,
                is_monolithic=is_monolithic,
                previous_runs=previous_runs,
                node_idx=node_idx,
                tasks_for_node=tasks_for_node,
                is_postprocessing=False
            )
        except Exception as e:
            logger.error(f"Error building simulation command: {e}")
            raise
        
        slurm.add_cmd("STAGE_EXIT_CODE=$?")
        slurm.add_cmd("set -e  # Re-enable exit on error")
        slurm.add_cmd("echo \"Stage completed with exit code: $STAGE_EXIT_CODE\"")
        
        # Add validation + guardian phases (validate only runs processed by this node)
        runs_for_this_node = self.get_runs_for_node(node_idx, tasks_for_node) if tasks_for_node else []
        self.add_validation_and_guardian_to_script(
            slurm, 
            self.run_dir,
            run_list=runs_for_this_node if runs_for_this_node else None
        )
    
    def _add_postprocessing_commands(self, slurm, previous_runs, is_monolithic=False, node_idx=0, tasks_for_node=None):
        """Add commands for postprocessing stages using shared command builder for consistency with interactive mode."""
        self.add_basic_slurm_env_setup(slurm)
        
        # Add phase 1 header
        slurm.add_cmd("")
        slurm.add_cmd("# ============================================================================")
        slurm.add_cmd(f"# PHASE 1: Executing stage - {self.config['stage']}")
        slurm.add_cmd("# ============================================================================")
        slurm.add_cmd("echo \"SLURM_JOB_ID: $SLURM_JOB_ID\"")
        slurm.add_cmd("echo \"SLURM_RESTART_COUNT: $SLURM_RESTART_COUNT\"")
        slurm.add_cmd("")
        
        # Don't exit on stage failure - let validation/guardian handle it
        slurm.add_cmd("set +e  # Don't exit on error")
        
        try:
            self._build_and_add_commands(
                slurm=slurm,
                is_monolithic=is_monolithic,
                previous_runs=previous_runs,
                node_idx=node_idx,
                tasks_for_node=tasks_for_node,
                is_postprocessing=True
            )
        except Exception as e:
            logger.error(f"Error building postprocessing command: {e}")
            raise
        
        slurm.add_cmd("STAGE_EXIT_CODE=$?")
        slurm.add_cmd("set -e  # Re-enable exit on error")
        slurm.add_cmd("echo \"Stage completed with exit code: $STAGE_EXIT_CODE\"")
        
        # Add validation + guardian phases (validate only runs processed by this node)
        # For postprocessing, use version_dir instead of run_dir for some stages
        output_dir = self.run_dir if is_monolithic else cli_utils.get_version_directory(self.config)
        runs_for_this_node = self.get_runs_for_node(node_idx, tasks_for_node) if tasks_for_node else []
        self.add_validation_and_guardian_to_script(
            slurm, 
            self.run_dir,
            run_list=runs_for_this_node if runs_for_this_node else None
        )
    
    def submit_jobs(self):
        """Submit all jobs for the stage"""
        job_ids = []
        
        for node_idx in range(self.n_nodes):
            slurm = self.create_slurm_job(node_idx)
            
            # Skip if no job was created (e.g., no runs for this node)
            if slurm is None:
                continue
            
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

    def _add_multinode_commands(self, slurm):
        """Add commands for a single multi-node job spanning all tasks."""
        self.add_basic_slurm_env_setup(slurm)
        
        # Add phase 1 header
        slurm.add_cmd("")
        slurm.add_cmd("# ============================================================================")
        slurm.add_cmd(f"# PHASE 1: Executing stage - {self.config['stage']}")
        slurm.add_cmd("# ============================================================================")
        slurm.add_cmd("echo \"SLURM_JOB_ID: $SLURM_JOB_ID\"")
        slurm.add_cmd("echo \"SLURM_RESTART_COUNT: $SLURM_RESTART_COUNT\"")
        slurm.add_cmd("")
        
        # Don't exit on stage failure - let validation/guardian handle it
        slurm.add_cmd("set +e  # Don't exit on error")

        # previous_runs is the range start if using ranges, else 0
        previous_runs = self.run_range[0] if self.run_range else 0
        # Global run id expr (for run_list)
        run_id_expr, run_ids_setup_cmd = self.get_run_id_expr_global()

        try:
            # Use shared builder; output_dir is runs dir for simulation, version dir for postprocessing in distributed modes
            is_post = self.is_postprocessing_stage()
            execution_mode = "distributed_slurm"  # reuse distributed semantics inside srun
            output_dir = self.run_dir if not is_post else cli_utils.get_version_directory(self.config)

            command_info = cli_utils.build_stage_command(
                config=self.config,
                config_path=self.config_path,
                stage_script_path=self.get_stage_script(),
                output_dir=output_dir,
                execution_mode=execution_mode,
                slurm_procid_offset=previous_runs,
                run_id_expr=run_id_expr
            )

            use_shifter = command_info["use_shifter"]
            if use_shifter:
                slurm.add_cmd(command_info["shifter_command"])
                if run_ids_setup_cmd:
                    slurm.add_cmd(run_ids_setup_cmd)
                for cmd in command_info["env_setup_commands"]:
                    slurm.add_cmd(cmd + " && \\")
                slurm.add_cmd(command_info["python_command"] + "\"")
            else:
                # Non-shifter (postprocessing): env setup OUTSIDE srun to avoid SLURM allocation issues
                for cmd in command_info["env_setup_commands"]:
                    slurm.add_cmd(cmd)
                srun_options = "--exact --kill-on-bad-exit=0"
                if run_ids_setup_cmd:
                    setup_clean = run_ids_setup_cmd.replace(" && \\", "").strip()
                    payload = f"{setup_clean} && {command_info['python_command']}"
                else:
                    payload = f"{command_info['python_command']}"
                srun_cmd = (
                    f"srun {srun_options} bash -c \"{payload}\""
                )
                slurm.add_cmd(srun_cmd)

        except Exception as e:
            logger.error(f"Error building multinode command: {e}")
            raise
        
        slurm.add_cmd("STAGE_EXIT_CODE=$?")
        slurm.add_cmd("set -e  # Re-enable exit on error")
        slurm.add_cmd("echo \"Stage completed with exit code: $STAGE_EXIT_CODE\"")
        
        # Add validation + guardian phases
        # Multi-node jobs: calculate which runs are processed based on total_tasks
        total_tasks = self.compute_total_tasks()
        if self.run_list:
            # Using specific run IDs from list
            runs_to_validate = self.run_ids[:total_tasks]
        elif self.run_range:
            # Using range of runs
            start_run, end_run = self.run_range
            runs_to_validate = list(range(start_run, min(start_run + total_tasks, end_run)))
        else:
            # Default: sequential runs from 0
            n_runs = self.config["job_config"]["n_runs"]
            runs_to_validate = list(range(min(total_tasks, n_runs)))
        
        self.add_validation_and_guardian_to_script(
            slurm, 
            self.run_dir,
            run_list=runs_to_validate
        )

    def submit_multi_node_job(self):
        """Submit a single SLURM job across multiple nodes with many tasks."""
        job_cfg = self.config["job_config"]
        common_cfg = self.config["common"]

        total_tasks = self.compute_total_tasks()
        # n_nodes already computed in calculate_job_distribution(); allow override via job_config.nodes
        n_nodes = int(job_cfg.get("nodes", self.n_nodes))

        # Compute cpus_per_task similar to distributed mode shape
        runs_per_node = job_cfg["runs_per_node"]
        cpus_per_task = max(1, job_cfg.get("max_cores", 256)//max(1, runs_per_node))

        # Optional dependency: allow chaining on a prior SLURM job id
        dependency_kw = self.parse_dependency_kw()

        slurm_kwargs = dict(
            job_name=f"colliderML_{self.config['stage']}_mn",
            account=common_cfg["account"],
            qos=job_cfg["qos"],
            time=job_cfg["time_limit"],
            nodes=n_nodes,
            ntasks=total_tasks,
            cpus_per_task=cpus_per_task,
            constraint="cpu",
            output=str(self.log_dir / f"job_multinode_%j.out"),
            error=str(self.log_dir / f"job_multinode_%j.err")
        )
        
        if dependency_kw is not None:
            slurm_kwargs["dependency"] = dependency_kw
        slurm = Slurm(**slurm_kwargs)
        
        # Add requeue and append flags for validation + guardian integration
        # For boolean flags, set to empty string to just add the flag without value
        setattr(slurm.namespace, "requeue", "")  # Boolean flag - no value
        setattr(slurm.namespace, "open-mode", "append")  # Flag with value
        
        # Add shifter image to SBATCH directives if needed (for performance)
        # Use setattr to add custom directive with multiple options on one line
        stage = self.config["stage"]
        if stage in cli_utils.SHIFTER_STAGES:
            container = common_cfg.get("container")
            if container:
                # Direct attribute injection: image value includes both --image and --module
                setattr(slurm.namespace, "image", f"{container} --module=cvmfs")

        # Add srun command invoking tasks across nodes
        self._add_multinode_commands(slurm)

        if self.dry_run:
            script_path = self.save_batch_script(slurm, "job_multinode.sh")
            logger.info(f"Saved multinode batch script to {script_path}")
            return ["DRY_RUN_JOB_MULTINODE"]
        else:
            job_id = slurm.sbatch(shell="/bin/bash", job_file=f"{self.log_dir}/job_multinode.sh", convert=False)
            logger.info(f"Submitted single multinode job with ID {job_id} spanning {n_nodes} nodes and {total_tasks} tasks")
            return [job_id]
    
    def submit_validation_jobs(self, job_ids):        
        """Submit a single-node validation job dependent on all stage jobs
        
        Uses 'afterany' dependency so validation always runs after the previous
        job completes, regardless of success or failure.
        """

        if self.config.get("validation_config", None) is None:
            logger.info("No validation config found, skipping validation jobs")
            return []

        # Defaults if not provided
        val_cfg = self.config.get("validation_config", {})
        val_qos = val_cfg.get("qos", self.config["job_config"].get("qos", "regular"))
        val_time = val_cfg.get("time_limit", "00:10:00")

        # Build single validation Slurm job with dependency on all production jobs
        # Use 'afterany' so validation runs even if the previous job failed
        slurm = Slurm(
            job_name=f"validate_{self.config['stage']}",
            account=self.config["common"]["account"],
            qos=val_qos,
            time=val_time,
            dependency={"afterany": job_ids} if (job_ids and not self.dry_run) else None,
            output=str(self.validation_dir / f"validation_%j.out"),
            error=str(self.validation_dir / f"validation_%j.err"),
            constraint="cpu",
            nodes=1,
            ntasks=1
        )

        # Basic env setup (reuse production env)
        slurm.add_cmd("cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software")
        slurm.add_cmd("eval \"$(conda shell.bash hook)\"")
        slurm.add_cmd("conda activate collider-env")

        # Locate validation script: prefer simulation/validation then fallback
        validation_script_full_path = None
        try:
            base_script_path = Path(__file__).parent.parent
            candidate1 = base_script_path / f"simulation/validation/validate_{self.config['stage']}.py"
            candidate2 = base_script_path / f"validation/validate_{self.config['stage']}.py"
            if candidate1.is_file():
                validation_script_full_path = str(candidate1)
            elif candidate2.is_file():
                validation_script_full_path = str(candidate2)
            else:
                raise FileNotFoundError(f"Validation script not found at {candidate1} or {candidate2}")
        except Exception as e:
            logger.warning(f"Failed to locate validation script automatically: {e}")
            # Fall back to a standard path; may fail at runtime if not present
            validation_script_full_path = f"colliderml_dev/scripts/simulation/validation/validate_{self.config['stage']}.py"

        # Validation CLI: stage and runs directory
        cmd = (f"python {validation_script_full_path} "
               f"--stage {self.config['stage']} "
               f"--runs-dir {self.run_dir}")

        slurm.add_cmd(cmd)

        if self.dry_run:
            script_path = self.save_batch_script(slurm, "validation.sh")
            logger.info(f"Saved validation batch script to {script_path}")
            return ["DRY_RUN_VALIDATION"]
        else:
            validation_id = slurm.sbatch(
                shell="/bin/bash", 
                job_file=f"{self.validation_dir}/validation.sh",
                convert=False)
            logger.info(f"Submitted single validation job {validation_id} for production jobs {job_ids}")
            return [validation_id]

    def save_batch_script(self, slurm, script_name):
        """Save the batch script that would be submitted"""
        script_path = self.dry_run_dir / script_name
        
        # Get the complete script content (includes all SBATCH directives added via add_arguments)
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

    # Clean up the temporary file (single removal)
    os.remove(temp_config_path)

    if args.dry_run:
        logger.info(f"Dry run completed. Batch scripts saved in: {submitter.dry_run_dir}")
    else:
        logger.info(f"Submitted {len(job_ids)} production jobs with IDs: {job_ids}")
        logger.info(f"Submitted {len(validation_ids)} validation jobs with IDs: {validation_ids}")

if __name__ == "__main__":
    main() 