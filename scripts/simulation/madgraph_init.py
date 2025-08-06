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
from utils.config import create_base_parser, load_config

# Note: We avoid importing cli_utils to prevent anti-pattern of modifying shared utilities

logger = logging.getLogger(__name__)

def run_command(command, cwd=None, env=None, shell=False):
    """Execute a command and return stdout, stderr"""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        cwd=cwd,
        env=env,
        shell=shell
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error running command: {command}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
        sys.exit(1)
    return stdout, stderr

def customize_card_with_regex(card_path, card_settings):
    """
    Modifies a MadGraph card using regex for specific parameters.
    Works for both run_card.dat, shower_card.dat, and pythia8_card.dat.
    Updates existing parameters and adds new ones if they don't exist.
    """
    import re
    
    if not card_path.exists():
        logger.warning(f"Card file {card_path} does not exist. Skipping customization.")
        return
    
    with open(card_path, 'r') as f:
        content_lines = f.readlines()

    # Track which parameters were successfully updated
    updated_params = set()
    modified_lines = []
    
    for line in content_lines:
        modified_line = line
        for param_name, param_value in card_settings.items():
            # Skip if already updated (avoid double-updating)
            if param_name in updated_params:
                continue
                
            # Handle different card formats:
            # run_card/shower_card: '  10000 = nevents    ! Number of events'
            # pythia8_card: 'Main:numberOfEvents      = -1'
            
            # Pattern 1: MG format with = param_name
            mg_pattern = rf"^(\s*)(.+?)(\s*=\s*{re.escape(param_name)})(\s*[!#].*|\s*)$"
            mg_match = re.match(mg_pattern, line)
            
            # Pattern 2: Pythia8 format with param_name =
            pythia_pattern = rf"^(\s*{re.escape(param_name)}\s*=\s*)(.+?)(\s*[!#].*|\s*)$"
            pythia_match = re.match(pythia_pattern, line)
            
            if mg_match:
                # MadGraph format: value = param_name
                modified_line = f"{mg_match.group(1)}{str(param_value)}{mg_match.group(3)}{mg_match.group(4)}\n"
                updated_params.add(param_name)
                break
            elif pythia_match:
                # Pythia8 format: param_name = value
                modified_line = f"{pythia_match.group(1)}{str(param_value)}{pythia_match.group(3)}\n"
                updated_params.add(param_name)
                break
                
        modified_lines.append(modified_line)

    # Add any parameters that weren't found in the existing file
    missing_params = set(card_settings.keys()) - updated_params
    if missing_params:
        # Add a comment section for new parameters
        modified_lines.append("\n")
        modified_lines.append("!======================================================================\n")
        modified_lines.append("! Parameters added by ColliderML madgraph_init.py\n")
        modified_lines.append("!======================================================================\n")
        
        for param_name in sorted(missing_params):  # Sort for consistency
            param_value = card_settings[param_name]
            # Use Pythia8 format for new parameters (more common)
            new_line = f"{param_name} = {param_value}    ! Added by ColliderML script\n"
            modified_lines.append(new_line)

    with open(card_path, 'w') as f:
        f.writelines(modified_lines)
    
    # Log results
    updated_list = list(updated_params)
    added_list = list(missing_params) if missing_params else []
    logger.info(f"Updated {card_path.name}:")
    if updated_list:
        logger.info(f"  - Updated existing parameters: {updated_list}")
    if added_list:
        logger.info(f"  - Added new parameters: {added_list}")

def detect_process_type(process_generation_stdout):
    """
    Detect whether this is a born (NLO) or noborn (loop-induced) process.
    
    Args:
        process_generation_stdout: The stdout from MadGraph process generation
        
    Returns:
        str: "born" for NLO processes, "noborn" for loop-induced processes
    """
    if "noborn" in process_generation_stdout.lower():
        return "noborn"
    else:
        return "born"

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
    stdout_proc, stderr_proc = run_command([str(mg5_exe), str(temp_proc_script_path)], cwd=temp_run_dir)
    
    logger.info("MadGraph process generation STDOUT:")
    logger.info(stdout_proc)
    if stderr_proc:
        logger.warning("MadGraph process generation STDERR:")
        logger.warning(stderr_proc)
    
    # Verify process directory was created
    generated_process_dir = temp_run_dir / madgraph_proc_output_dirname
    cards_dir = generated_process_dir / "Cards"
    if not cards_dir.is_dir():
        logger.error(f"Cards directory not found at {cards_dir} after process generation.")
        sys.exit(1)

    logger.info("MadGraph process generation complete.")
    return generated_process_dir, stdout_proc

def customize_default_cards(process_dir, config, process_generation_stdout, logger):
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
    
    # Detect process type (born vs noborn)
    process_type = detect_process_type(process_generation_stdout)
    logger.info(f"Detected process type: {process_type}")
    
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
    
    # Customize shower/pythia card based on process type
    if process_type == "born":
        # NLO processes use shower_card.dat
        shower_card_path = cards_dir / "shower_card.dat"
        shower_card_settings = config.card_customizations.get('shower_card', {})
        
        if shower_card_settings:
            logger.info(f"Applying shower_card customizations: {list(shower_card_settings.keys())}")
            customize_card_with_regex(shower_card_path, shower_card_settings)
        else:
            logger.info("No shower_card customizations specified. Using MadGraph defaults.")
            
    elif process_type == "noborn":
        # Loop-induced processes use pythia8_card.dat
        pythia8_card_path = cards_dir / "pythia8_card.dat"
        pythia8_card_settings = config.card_customizations.get('pythia8_card', {})
        
        if pythia8_card_settings:
            logger.info(f"Applying pythia8_card customizations: {list(pythia8_card_settings.keys())}")
            customize_card_with_regex(pythia8_card_path, pythia8_card_settings)
        else:
            logger.info("No pythia8_card customizations specified. Using MadGraph defaults.")
    
    logger.info("Default card customization completed.")

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
    # Build version directory path directly (avoid modifying shared cli_utils)
    base_dir = Path(config.common["output_base_dir"])
    version_dir = base_dir / config.campaign / config.dataset / config.version
    final_process_dir = version_dir / "madgraph_process"
    
    # Remove existing directory if it exists, then create clean parent directories
    if final_process_dir.exists():
        logger.warning(f"Removing existing process directory: {final_process_dir}")
        shutil.rmtree(final_process_dir)
    
    # Ensure parent directory exists
    version_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy the process directory to final location
    logger.info(f"Storing compiled process directory: {process_dir} -> {final_process_dir}")
    # Use ignore_dangling_symlinks to handle broken symlinks that MadGraph sometimes creates
    shutil.copytree(process_dir, final_process_dir, ignore_dangling_symlinks=True)
    
    # Log some statistics about the stored directory
    total_size = sum(f.stat().st_size for f in final_process_dir.rglob('*') if f.is_file())
    total_size_mb = total_size / (1024 * 1024)
    file_count = len(list(final_process_dir.rglob('*')))
    
    logger.info(f"Process directory stored successfully:")
    logger.info(f"  Location: {final_process_dir}")
    logger.info(f"  Size: {total_size_mb:.1f} MB")
    logger.info(f"  Files: {file_count}")
    
    return final_process_dir

def main():
    """Main entry point for madgraph_init script"""
    parser = create_base_parser("MadGraph process initialization for ColliderML")
    args = parser.parse_args()
    config = load_config(args)

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
        customize_default_cards(process_dir, config, process_stdout, logger)
        
        # Step 3: Store the process directory
        logger.info("=== Step 3: Store Process Directory ===")
        final_process_dir = store_process_directory(process_dir, config, logger)
        
        # Step 4: Cleanup temporary directory
        logger.info("=== Step 4: Cleanup ===")
        temp_parent = process_dir.parent
        logger.info(f"Cleaning up temporary directory: {temp_parent}")
        shutil.rmtree(temp_parent)
        
        logger.info("=== MadGraph Process Initialization Complete ===")
        logger.info(f"Process '{process_name}' ready for parallel event generation")
        logger.info(f"Stored at: {final_process_dir}")
        
    except Exception as e:
        logger.error(f"MadGraph initialization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()