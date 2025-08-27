#!/usr/bin/env python3
"""
MadGraph Shared Utilities
=========================

Shared functions used by both madgraph_init.py and madgraph_gen.py
to maintain DRY principles and consistency.
"""

import os
import sys
import subprocess
import shutil
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def run_command(command,
                cwd=None,
                env=None,
                shell=False,
                stream=False,
                capture=True,
                merge_streams=False,
                raise_on_error=True,
                logger: logging.Logger = logger):
    """
    Execute a command.
    - When stream=False (default): buffer stdout/stderr and return (stdout, stderr).
    - When stream=True: stream lines to logger; if capture=True, return (combined_stdout, None) when merge_streams=True.
      Note: streaming currently merges stdout/stderr; merge_streams is forced True in streaming mode.
    """
    if stream:
        if not merge_streams:
            # Simpler, robust streaming path: merge to avoid deadlocks
            logger.debug("merge_streams=False requested with stream=True; forcing merge_streams=True to avoid deadlocks.")
            merge_streams = True
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if merge_streams else subprocess.PIPE,
            universal_newlines=True,
            cwd=cwd,
            env=env,
            shell=shell,
            bufsize=1,
        )
        assert process.stdout is not None
        combined_text = [] if capture else None
        for line in process.stdout:
            logger.info(line.rstrip())
            if capture:
                combined_text.append(line)
        ret = process.wait()
        if raise_on_error and ret != 0:
            raise RuntimeError(f"Command failed with exit code {ret}: {command}")
        return ("".join(combined_text) if capture else None, None)
    else:
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
        if raise_on_error and process.returncode != 0:
            raise RuntimeError(f"Error running command: {command}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
        return stdout, stderr

def run_command_streaming(command, cwd=None, env=None, shell=False, logger: logging.Logger = logger, raise_on_error=True):
    """Compatibility wrapper: prefer run_command(..., stream=True)."""
    stdout, _ = run_command(command, cwd=cwd, env=env, shell=shell, stream=True, capture=True, merge_streams=True, raise_on_error=raise_on_error, logger=logger)
    return 0

def customize_card_with_regex(card_path, card_settings):
    """
    Modifies a MadGraph card using regex for specific parameters.
    Works for both run_card.dat, shower_card.dat, and pythia8_card.dat.
    Updates existing parameters and adds new ones if they don't exist.
    """
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
        for param_name in sorted(missing_params):  # Sort for consistency
            param_value = card_settings[param_name]
            # Use Pythia8 format for new parameters (more common)
            new_line = f"{param_name} = {param_value} \n"
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

def detect_process_type_from_stdout(process_generation_stdout):
    """
    Detect whether this is a born (NLO) or noborn (loop-induced) process
    from MadGraph's process generation stdout.
    
    Args:
        process_generation_stdout: The stdout from MadGraph process generation
        
    Returns:
        str: "born" for NLO processes, "noborn" for loop-induced processes
    """
    if "noborn" in process_generation_stdout.lower():
        return "noborn"
    else:
        return "born"

def detect_process_type_from_files(process_dir):
    """
    Detect whether this is a born (NLO) or noborn (loop-induced) process
    by checking which auto-generated card files exist.
    
    Args:
        process_dir: Path to the MadGraph process directory
        
    Returns:
        str: "born" for NLO processes, "noborn" for loop-induced processes
    """
    cards_dir = process_dir / "Cards"
    
    shower_card = cards_dir / "shower_card.dat"
    pythia8_card = cards_dir / "pythia8_card.dat"
    
    if shower_card.exists():
        logger.info("Detected NLO (born) process - found shower_card.dat")
        return "born"
    elif pythia8_card.exists():
        logger.info("Detected loop-induced (noborn) process - found pythia8_card.dat")
        return "noborn"
    else:
        logger.warning("Could not detect process type - neither shower_card.dat nor pythia8_card.dat found")
        return "born"  # Default assumption

def get_version_directory_path(config):
    """
    Build the version directory path from config.
    
    Args:
        config: Configuration object
        
    Returns:
        Path: Version directory path
    """
    base_dir = Path(config.common["output_base_dir"])
    return base_dir / config.campaign / config.dataset / config.version

def customize_cards_for_process_type(process_dir, config, process_type, script_name="ColliderML"):
    """
    Customize the appropriate cards based on process type.
    
    Args:
        process_dir: Path to the MadGraph process directory
        config: Configuration object  
        process_type: "born" or "noborn"
        script_name: Name of the calling script for logging
    """
    cards_dir = process_dir / "Cards"
    
    # Always customize run_card.dat
    run_card_path = cards_dir / "run_card.dat"
    if hasattr(config, 'card_customizations') and 'run_card' in config.card_customizations:
        logger.info("Customizing run_card.dat...")
        customize_card_with_regex(run_card_path, config.card_customizations['run_card'])
    
    # Customize process-type specific cards
    if process_type == "born":
        # NLO process - customize shower_card.dat
        shower_card_path = cards_dir / "shower_card.dat"
        if hasattr(config, 'card_customizations') and 'shower_card' in config.card_customizations:
            logger.info("Customizing shower_card.dat for NLO process...")
            customize_card_with_regex(shower_card_path, config.card_customizations['shower_card'])
        else:
            logger.info("No shower_card customizations specified. Using MadGraph defaults.")
    
    elif process_type == "noborn":
        # Loop-induced process - customize pythia8_card.dat
        pythia8_card_path = cards_dir / "pythia8_card.dat"
        if hasattr(config, 'card_customizations') and 'pythia8_card' in config.card_customizations:
            logger.info("Customizing pythia8_card.dat for loop-induced process...")
            customize_card_with_regex(pythia8_card_path, config.card_customizations['pythia8_card'])
        else:
            logger.info("No pythia8_card customizations specified. Using MadGraph defaults.")
    
    logger.info(f"Card customization completed by {script_name}.")