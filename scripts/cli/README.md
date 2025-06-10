# ColliderML Stage Runner (`run_stage.py`)

This directory contains the main command-line interface for running stages of the ColliderML data production pipeline.

## Overview

The primary entry point is `run_stage.py`. It serves as the main front door for all data production tasks.

## Core Functionality of `run_stage.py`

1.  **Configuration-Driven**:
    *   It takes a single YAML configuration file as input (e.g., `generation_test.yaml`).
    *   This YAML file specifies the stage to be run (e.g., generation, simulation, build_tracks) and all other necessary parameters for that stage.

2.  **Configurable Environment Setup**:
    *   `run_stage.py` automatically loads environment setup commands from `env_setup.yaml`, located in the same directory.
    *   This file defines setup commands for different categories of stages (e.g., `simulation`, `postprocessing`).
    *   This allows for a clear separation of pipeline logic from environment setup and ensures consistency across all execution modes.
    *   You can use placeholders like `{SOFTWARE_DIR}` in the YAML file, which are automatically replaced with the path to your software repository.

3.  **Execution Mode Orchestration**:
    `run_stage.py` intelligently determines how to execute the specified stage based on the configuration. It supports three main execution modes:
    *   **Interactive**: For quick tasks or debugging, the stage's Python script is run directly as a subprocess. The environment is configured first, based on `env_setup.yaml`, before the script is executed.
    *   **Monolithic SLURM Job**: For stages that are best run as a single, cohesive job, `run_stage.py` submits one job to SLURM. The batch script will include the necessary environment setup commands.
    *   **Distributed SLURM Jobs**: For stages that can be parallelized, `run_stage.py` submits multiple SLURM jobs, each with the correct environment setup.

4.  **Reproducibility - Pre-run Steps**:
    *   Before any execution, `run_stage.py` performs crucial pre-run checks and actions:
        *   It validates the provided configuration file and the referenced stage-specific scripts and parameters.
        *   It **git commits** the current state of the repository (e.g., `ColliderML/software/`) to ensure that the exact code version used for the data production run is recorded. This commit will likely include the configuration file itself if it's part of the repository, or a copy of it.

5.  **Interface with `job_submission.py`**:
    *   The `job_submission.py` script (now also in this `cli` directory) has been refactored to act more like a library.
    *   For distributed and monolithic SLURM job submissions, `run_stage.py` will call functions within `job_submission.py` to prepare and submit the SLURM batch scripts.

## Workflow Example

1.  The user prepares a YAML configuration file (e.g., `my_config.yaml`) specifying the stage (e.g., `new_generation`), execution parameters, and output locations.
2.  The user ensures the environment setup is correctly specified in `scripts/cli/env_setup.yaml`.
3.  The user runs: `python run_stage.py --config my_config.yaml`
4.  `run_stage.py`:
    a.  Parses `my_config.yaml`.
    b.  Loads and merges `env_setup.yaml`.
    c.  Validates inputs.
    d.  Performs a git commit of the codebase.
    e.  Determines the execution mode (e.g., "monolithic SLURM" for `new_generation`).
    f.  If SLURM (monolithic or distributed):
        i.  Calls the relevant functions in `job_submission.py` to generate SLURM script(s) that include the environment setup.
        ii. `job_submission.py` submits the job(s) to SLURM.
    g.  If Interactive:
        i.  Executes the environment setup commands and then runs the target Python script for the stage in a subshell.
5.  The stage executes, producing data in the configured output directories, respecting the structure required for subsequent stages (e.g., a monolithic MadGraph stage might internally handle splitting its output into `runs/0`, `runs/1`, etc.). 