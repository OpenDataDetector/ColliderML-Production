#!/usr/bin/env python3
"""
MadGraph Process Initialization Script
=====================================

This script handles the expensive, one-time process generation step for MadGraph.
It compiles the matrix elements for a specific physics process and stores the
resulting process directory for later use by parallel event generation jobs.

This is the first stage of a two-step MadGraph workflow:
1. madgraph_init: Generate and compile the process (this script)
2. madgraph_generation: Parallel event generation using the compiled process

Usage:
    python madgraph_init.py --config config.yaml

The compiled process directory will be stored in:
    {dataset_version_dir}/madgraph_process/{process_name}/
"""

import os
import sys
import subprocess
import shutil
import argparse
import yaml
import logging
from pathlib import Path
import tarfile
from utils.config import create_base_parser, load_config
from utils.madgraph_utils import (
    run_command, 
    run_command_streaming,
    customize_card_with_regex, 
    get_version_directory_path,
)

logger = logging.getLogger(__name__)

# run_command is now imported from madgraph_utils

# customize_card_with_regex and detect_process_type_from_stdout are now imported from madgraph_utils

def generate_madgraph_process(config, scratch_dir, logger):
    """
    Generate and compile a MadGraph process.
    
    Args:
        config: Configuration object
        scratch_dir: Temporary directory for MadGraph operations
        logger: Logger instance
        
    Returns:
        Path to the generated process directory
    """
    process_name = f"{config.dataset}_{config.version}"
    mg_base_path = Path(config.mg_base_path)
    mg5_exe = mg_base_path / "bin" / "mg5_aMC"

    mg_model_cmd = f"import model {config.mg_model}"
    mg_define_cmds = config.mg_definitions
    mg_generate_cmd = config.mg_generate_command

    # Create temporary run directory
    temp_run_dir = scratch_dir / f"mg5_init_{process_name}"
    temp_run_dir.mkdir(parents=True, exist_ok=True)
    
    madgraph_proc_output_dirname = "proc_output_mg"
    
    logger.info(f"Generating MadGraph process: {process_name}")
    logger.info(f"Model: {config.mg_model}")
    logger.info(f"Process: {mg_generate_cmd}")
    
    # Create MadGraph script for process generation
    temp_proc_script_path = temp_run_dir / "process_script.mg5"
    with open(temp_proc_script_path, 'w') as f_out:
        f_out.write(f"{mg_model_cmd}\n")
        for define_cmd in mg_define_cmds:
            f_out.write(f"{define_cmd}\n")
        f_out.write(f"{mg_generate_cmd}\n")
        f_out.write(f"output {madgraph_proc_output_dirname} -f\n")
        f_out.write("exit\n")

    logger.info(f"Running MadGraph process generation (script: {temp_proc_script_path})")
    # Stream output in realtime while capturing combined text for later parsing
    stdout_proc, _ = run_command([str(mg5_exe), str(temp_proc_script_path)], cwd=temp_run_dir,
                                 stream=True, capture=True, merge_streams=True, logger=logger)
    
    # Verify process directory was created
    generated_process_dir = temp_run_dir / madgraph_proc_output_dirname
    cards_dir = generated_process_dir / "Cards"
    if not cards_dir.is_dir():
        logger.error(f"Cards directory not found at {cards_dir} after process generation.")
        sys.exit(1)

    logger.info("MadGraph process generation complete.")
    return generated_process_dir, stdout_proc

def customize_default_cards(process_dir, config, logger):
    """
    Customize the default cards in the process directory.
    This creates template cards suitable for later per-run customization.
    
    Args:
        process_dir: Path to the MadGraph process directory
        config: Configuration object
        process_generation_stdout: Output from process generation (for born/noborn detection)
        logger: Logger instance
    """
    cards_dir = process_dir / "Cards"
    
    # Customize run_card.dat (common to all process types)
    run_card_path = cards_dir / "run_card.dat"
    run_card_settings = config.card_customizations.get('run_card', {})
    
    if run_card_settings:
        # Exclude run-specific parameters (these will be set during event generation)
        default_run_card_settings = {
            k: v for k, v in run_card_settings.items() 
            if k not in ['nevents', 'iseed']
        }
        if default_run_card_settings:
            logger.info(f"Applying default run_card customizations: {list(default_run_card_settings.keys())}")
            customize_card_with_regex(run_card_path, default_run_card_settings)
    
    # Customize shower/pythia card based solely on run_mode (explicit)
    run_mode = getattr(config, 'run_mode', 'nlo_fxfx')
    if str(run_mode).lower() == 'lo_mlm':
        # For LO+MLM, always ensure pythia8_card.dat gets the JetMatching settings
        # Prefer pythia8_card.dat; fall back to pythia8_card_default.dat if needed
        pythia8_card_path = cards_dir / "pythia8_card.dat"
        if not pythia8_card_path.exists():
            alt_path = cards_dir / "pythia8_card_default.dat"
            if alt_path.exists():
                pythia8_card_path = alt_path
        pythia8_card_settings = config.card_customizations.get('pythia8_card', {})
        if pythia8_card_settings:
            logger.info(f"Applying pythia8_card customizations for run_mode=lo_mlm: {list(pythia8_card_settings.keys())}")
            customize_card_with_regex(pythia8_card_path, pythia8_card_settings)
        else:
            logger.info("No pythia8_card customizations specified for lo_mlm. Using defaults.")
        logger.info("Default card customization completed (forced pythia8 for lo_mlm).")
        return
    else:
        # NLO/FxFx path: customize shower_card.dat if provided
        shower_card_path = cards_dir / "shower_card.dat"
        shower_card_settings = config.card_customizations.get('shower_card', {})
        if shower_card_settings:
            logger.info(f"Applying shower_card customizations for run_mode={run_mode}: {list(shower_card_settings.keys())}")
            customize_card_with_regex(shower_card_path, shower_card_settings)
        else:
            logger.info("No shower_card customizations specified. Using MadGraph defaults.")
    
    logger.info("Default card customization completed.")


def compile_grids_and_envelopes(process_dir: Path, logger: logging.Logger):
    """
    Edit cards for a zero-event integration run and execute generate_events to
    build grids/envelopes inside the process directory.

    This uses nevents=0 and req_acc=0.001 to force grid construction without
    producing events. The process directory is modified in place.
    """
    cards_dir = process_dir / "Cards"
    run_card_path = cards_dir / "run_card.dat"

    logger.info("Preparing run_card.dat for grid/envelope compilation (nevents=0, req_acc=0.001)")
    try:
        compile_settings = {
            'nevents': 0,
            'req_acc': 0.001,
        }
        customize_card_with_regex(run_card_path, compile_settings)
    except Exception as e:
        logger.error(f"Failed to update run_card.dat for grid compilation: {e}")
        raise

    generate_events_exe = process_dir / "bin" / "generate_events"
    logger.info(f"Running grid/envelope compilation via {generate_events_exe} (-f --name run_build)")
    # Stream output in real-time for debuggability
    try:
        run_command([str(generate_events_exe), "-f", "--name", "run_build"], cwd=str(process_dir), stream=True, capture=False, merge_streams=True, logger=logger)
    except Exception as e:
        logger.error(f"Error running generate_events for grid build: {e}")
        raise

def store_process_directory(process_dir, config, logger):
    """
    Store the compiled process directory in the dataset version directory.
    
    Args:
        process_dir: Path to the compiled MadGraph process directory
        config: Configuration object
        logger: Logger instance
        
    Returns:
        Path to the stored process directory
    """
    # Determine storage location: dataset_version_dir/madgraph_process/
    version_dir = get_version_directory_path(config)
    final_process_dir = version_dir / "madgraph_process"
    
    # Remove existing directory if it exists, then create clean parent directories
    if final_process_dir.exists():
        logger.warning(f"Removing existing process directory: {final_process_dir}")
        shutil.rmtree(final_process_dir)
    
    # Ensure parent directory exists
    version_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy the process directory to final location
    logger.info(f"Storing compiled process directory: {process_dir} -> {final_process_dir}")
    # Preserve symlinks but ignore dangling ones that MadGraph sometimes creates
    shutil.copytree(process_dir, final_process_dir, symlinks=True, ignore_dangling_symlinks=True)
    
    # Log some statistics about the stored directory
    total_size = sum(f.stat().st_size for f in final_process_dir.rglob('*') if f.is_file())
    total_size_mb = total_size / (1024 * 1024)
    file_count = len(list(final_process_dir.rglob('*')))
    
    logger.info(f"Process directory stored successfully:")
    logger.info(f"  Location: {final_process_dir}")
    logger.info(f"  Size: {total_size_mb:.1f} MB")
    logger.info(f"  Files: {file_count}")
    
    return final_process_dir


def create_tarball_from_directory(source_dir: Path, tar_path: Path, logger: logging.Logger):
    """
    Create a gzipped tarball from source_dir at tar_path atomically.
    """
    try:
        tar_tmp_path = tar_path.with_suffix(tar_path.suffix + ".tmp")
        if tar_tmp_path.exists():
            try:
                tar_tmp_path.unlink()
            except Exception:
                pass
        logger.info(f"Creating tarball {tar_path.name} from {source_dir}")
        with tarfile.open(tar_tmp_path, "w:gz") as tar:
            # Store the directory with its basename as the top-level folder
            tar.add(source_dir, arcname=source_dir.name, recursive=True)
        # Atomic replace
        tar_tmp_path.replace(tar_path)
        size_mb = tar_path.stat().st_size / (1024 * 1024)
        logger.info(f"Tarball created: {tar_path} ({size_mb:.1f} MB)")
    except Exception as e:
        logger.error(f"Failed to create tarball {tar_path}: {e}")
        raise

def main():
    """Main entry point for madgraph_init script"""
    parser = create_base_parser("MadGraph process initialization for ColliderML")
    args = parser.parse_args()
    config = load_config(args)

    print(f"Config: {config}")

    # Set up logging
    log_level = getattr(config, 'log_level', 'INFO')
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger.info("Starting MadGraph process initialization")

    # Validate required configuration
    required_fields = ['mg_base_path', 'generation_scratch_dir', 'mg_model', 'mg_generate_command']
    for field in required_fields:
        if not hasattr(config, field) or getattr(config, field) is None:
            logger.error(f"Required configuration field missing: {field}")
            sys.exit(1)

    process_name = f"{config.dataset}_{config.version}"
    scratch_dir = Path(config.generation_scratch_dir)
    
    try:
        # Step 1: Generate the MadGraph process
        logger.info("=== Step 1: Generate MadGraph Process ===")
        process_dir, process_stdout = generate_madgraph_process(config, scratch_dir, logger)
        
        # Step 2: Customize default cards
        logger.info("=== Step 2: Customize Default Cards ===")
        customize_default_cards(process_dir, config, logger)

        # Step 2b: Build grids/envelopes in-place (0 events, req_acc=0.001)
        # For LO+MLM, compilation is cheap and grids are not needed. Keep DRY by
        # gating grid/envelope build behind run_mode.
        run_mode = getattr(config, 'run_mode', 'nlo_fxfx')
        if str(run_mode).lower() != 'lo_mlm':
            logger.info("=== Step 2b: Build Grids/Envelopes (No Events) ===")
            compile_grids_and_envelopes(process_dir, logger)
        else:
            logger.info("=== Step 2b: Skipping grid/envelope build for run_mode=lo_mlm ===")
        
        # Step 3: Store the process directory
        logger.info("=== Step 3: Store Process Directory ===")
        final_process_dir = store_process_directory(process_dir, config, logger)

        # Step 4: Create tarball artifact from stored directory
        logger.info("=== Step 4: Create Tarball Artifact ===")
        version_dir = get_version_directory_path(config)
        tarball_path = version_dir / "madgraph_process.tgz"
        create_tarball_from_directory(final_process_dir, tarball_path, logger)

        # Step 5: Cleanup temporary directory
        logger.info("=== Step 5: Cleanup ===")
        temp_parent = process_dir.parent
        logger.info(f"Cleaning up temporary directory: {temp_parent}")
        shutil.rmtree(temp_parent)
        
        logger.info("=== MadGraph Process Initialization Complete ===")
        logger.info(f"Process '{process_name}' ready for parallel event generation")
        logger.info(f"Stored directory: {final_process_dir}")
        logger.info(f"Stored tarball: {tarball_path}")
        
    except Exception as e:
        logger.error(f"MadGraph initialization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()