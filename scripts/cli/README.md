# ColliderML Stage Runner (`run_stage.py`)

This directory contains the main command-line interface for running stages of the ColliderML data production pipeline.

## Overview

The primary entry point is `run_stage.py`. It serves as the main front door for all data production tasks.

## Core Functionality of `run_stage.py`

1.  **Configuration-Driven**:
    *   It takes a single YAML configuration file as input (e.g., `generation_test.yaml`).
    *   This YAML file specifies the stage to be run (e.g., generation, simulation, build_tracks) and all other necessary parameters for that stage.

2.  **Execution Mode Orchestration**:
    `run_stage.py` intelligently determines how to execute the specified stage based on the configuration. It supports three main execution modes:
    *   **Interactive**: For quick tasks or debugging, the stage's Python script is imported and run directly as a subprocess within the `run_stage.py` execution.
    *   **Monolithic SLURM Job**: For stages that are best run as a single, cohesive job (e.g., MadGraph event generation which produces all events at once), `run_stage.py` submits one job to SLURM. This job will typically write its output to a single directory within the standard `dataset/version/` structure. Subsequent splitting or processing to fit the `runs/0`, `runs/1` convention for distributed stages will be handled either by this monolithic script itself or as a separate, small follow-up stage.
    *   **Distributed SLURM Jobs**: For stages that can be parallelized, `run_stage.py` utilizes the functionality (adapted from the original `job_submission.py`) to submit multiple SLURM jobs. Each job typically handles a subset of "runs" (e.g., `runs/0`, `runs/1`, ...), writing to its designated subdirectory.

3.  **Reproducibility - Pre-run Steps**:
    *   Before any execution, `run_stage.py` performs crucial pre-run checks and actions:
        *   It validates the provided configuration file and the referenced stage-specific scripts and parameters.
        *   It **git commits** the current state of the repository (e.g., `ColliderML/software/`) to ensure that the exact code version used for the data production run is recorded. This commit will likely include the configuration file itself if it's part of the repository, or a copy of it.

4.  **Interface with `job_submission.py`**:
    *   The `job_submission.py` script (now also in this `cli` directory) has been refactored to act more like a library.
    *   For distributed and monolithic SLURM job submissions, `run_stage.py` will call functions within `job_submission.py` to prepare and submit the SLURM batch scripts.

## Workflow Example

1.  The user prepares a YAML configuration file (e.g., `my_config.yaml`) specifying the stage (e.g., `new_generation`), execution parameters, output locations, and potentially the desired execution mode.
2.  The user runs: `python run_stage.py --config my_config.yaml`
3.  `run_stage.py`:
    a.  Parses `my_config.yaml`.
    b.  Validates inputs.
    c.  Performs a git commit of the codebase.
    d.  Determines the execution mode (e.g., "monolithic SLURM" for `new_generation`).
    e.  If SLURM (monolithic or distributed):
        i.  Calls the relevant functions in `job_submission.py` to generate SLURM script(s).
        ii. `job_submission.py` submits the job(s) to SLURM.
    f.  If Interactive:
        i.  Directly imports and runs the target Python script for the stage.
4.  The stage executes, producing data in the configured output directories, respecting the structure required for subsequent stages (e.g., a monolithic MadGraph stage might internally handle splitting its output into `runs/0`, `runs/1`, etc.). 