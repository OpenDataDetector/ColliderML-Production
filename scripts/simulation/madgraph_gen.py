#!/usr/bin/env python3
"""
MadGraph Event Generation Script
===============================

This script handles parallel event generation using pre-compiled MadGraph processes.
It works with process directories created by madgraph_init.py.

This is the second stage of a two-step MadGraph workflow:
1. madgraph_init: Generate and compile the process (done once)
2. madgraph_generation: Parallel event generation using the compiled process (this script)

Usage:
    python madgraph_gen.py --config config.yaml --output /path/to/output --events 128 --seed 42

The script will:
1. Copy the pre-compiled process directory from madgraph_process/ to scratch
2. Customize cards for the specific run (events, seed, run naming)
3. Generate events using MadGraph launch
4. Process and move output files
"""

import os
import sys
import subprocess
import shutil
import argparse
import yaml
import re
import logging
from pathlib import Path
from utils.config import create_base_parser, load_config
from utils.madgraph_utils import (
    run_command,
    customize_card_with_regex,
    detect_process_type_from_files,
    get_version_directory_path
)

logger = logging.getLogger(__name__)

# run_command is imported from madgraph_utils

# customize_card_with_regex and detect_process_type_from_files are imported from madgraph_utils

def copy_process_directory(config):
    """
    Copy the pre-compiled process directory from storage to scratch space.
    
    Args:
        config: Configuration object
        
    Returns:
        tuple: (copied_process_dir_path, job_scratch_dir_path)
    """
    # Build path to stored process directory using shared utility
    version_dir = get_version_directory_path(config)
    stored_process_dir = version_dir / "madgraph_process"
    
    if not stored_process_dir.exists():
        raise FileNotFoundError(
            f"Pre-compiled process directory not found at {stored_process_dir}. "
            f"Please run madgraph_init stage first."
        )
    
    # Create scratch directory for this job
    scratch_dir = Path(config.generation_scratch_dir)
    process_name = f"{config.dataset}_{config.version}"
    job_scratch_dir = scratch_dir / f"mg5_gen_{process_name}_{os.getpid()}"
    job_scratch_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy process directory to scratch
    copied_process_dir = job_scratch_dir / "process"
    logger.info(f"Copying process directory: {stored_process_dir} -> {copied_process_dir}")
    
    # Preserve symlinks but ignore dangling ones that MadGraph sometimes creates  
    shutil.copytree(stored_process_dir, copied_process_dir, symlinks=True, ignore_dangling_symlinks=True)
    
    # Log copy statistics
    total_size = sum(f.stat().st_size for f in copied_process_dir.rglob('*') if f.is_file())
    total_size_mb = total_size / (1024 * 1024)
    file_count = len(list(copied_process_dir.rglob('*')))
    logger.info(f"Process directory copied: {total_size_mb:.1f} MB, {file_count} files")
    
    return copied_process_dir, job_scratch_dir

def customize_cards_for_run(process_dir, config, run_id=None):
    """
    Customize cards for a specific run with events, seed, and run naming.
    Follows the same pattern as pythia_gen.py and ddsim_run.py.
    
    Args:
        process_dir: Path to the copied MadGraph process directory
        config: Configuration object with events, seed, etc.
        run_id: Optional run ID for --name parameter (from SLURM_PROCID or similar)
    
    Returns:
        str: Process type ("born" or "noborn")
    """
    cards_dir = process_dir / "Cards"
    
    # Detect process type from existing files using shared utility
    process_type = detect_process_type_from_files(process_dir)
    
    # Set up run-specific parameters following pythia_gen.py pattern
    run_params = {}
    
    # Events and seed - following the exact pattern from other scripts
    if hasattr(config, 'events'):
        run_params['nevents'] = config.events
    if hasattr(config, 'seed'):
        run_params['iseed'] = config.seed
    
    # Always customize run_card.dat with run-specific parameters
    run_card_path = cards_dir / "run_card.dat"
    logger.info(f"Customizing run_card.dat for run (events={run_params.get('nevents', 'default')}, seed={run_params.get('iseed', 'default')})")
    
    # Get base run_card settings from config (guaranteed to exist and be a dict)
    base_run_card_settings = config.card_customizations['run_card']
    
    # Merge base settings with run-specific parameters
    final_run_card_settings = {**base_run_card_settings, **run_params}
    customize_card_with_regex(run_card_path, final_run_card_settings)
    
    # Customize process-type specific cards (shower_card.dat or pythia8_card.dat)
    if process_type == "born":
        # NLO process - customize shower_card.dat if it exists
        shower_card_path = cards_dir / "shower_card.dat"
        if shower_card_path.exists():
            shower_params = {}
            if hasattr(config, 'events'):
                shower_params['nevents'] = config.events
            if hasattr(config, 'seed'):
                shower_params['rnd_seed'] = config.seed
            
            # Get base shower_card settings from config (guaranteed to exist and be a dict)
            base_shower_settings = config.card_customizations['shower_card']
            final_shower_settings = {**base_shower_settings, **shower_params}
            
            if final_shower_settings:
                logger.info("Customizing shower_card.dat for NLO process")
                customize_card_with_regex(shower_card_path, final_shower_settings)
    
    elif process_type == "noborn":
        # Loop-induced process - customize pythia8_card.dat if it exists
        pythia8_card_path = cards_dir / "pythia8_card.dat"
        if pythia8_card_path.exists():
            pythia8_params = {}
            if hasattr(config, 'events'):
                pythia8_params['Main:numberOfEvents'] = config.events
            if hasattr(config, 'seed'):
                pythia8_params['Random:seed'] = config.seed
            
            # Get base pythia8_card settings from config (guaranteed to exist and be a dict)
            base_pythia8_settings = config.card_customizations['pythia8_card']
            final_pythia8_settings = {**base_pythia8_settings, **pythia8_params}
            
            if final_pythia8_settings:
                logger.info("Customizing pythia8_card.dat for loop-induced process")
                customize_card_with_regex(pythia8_card_path, final_pythia8_settings)
    
    return process_type


def split_hepmc_file(input_hepmc_path: Path,
                     final_output_base_dir: Path,
                     events_per_file: int,
                     output_filename: str = "events.hepmc"):
    """
    Splits a single (potentially gzipped) HEPMC file into multiple smaller HEPMC files,
    each in its own subdirectory (0, 1, 2, etc.) under final_output_base_dir.
    """
    try:
        import pyhepmc as hep
        from pyhepmc.io import WriterAscii
    except ImportError:
        print("Error: pyhepmc library not found. Please install it to use HEPMC splitting (e.g., pip install pyhepmc).")
        print(f"Skipping splitting of {input_hepmc_path}.")
        return [] # Indicate no files created

    try:
        from tqdm import tqdm
    except ImportError:
        print("Warning: tqdm library not found. Progress bar for splitting will not be shown (e.g., pip install tqdm).")
        def tqdm_dummy(iterable, *args, **kwargs): # Renamed to avoid conflict if tqdm is later imported globally
            return iterable
        tqdm_actual = tqdm_dummy # Use the dummy
    else:
        tqdm_actual = tqdm # Use the real tqdm

    print(f"--- Splitting HEPMC file: {input_hepmc_path} ---")
    print(f"--- Output base directory for splits: {final_output_base_dir} ---")
    print(f"--- Events per split file: {events_per_file} ---")

    current_f_out = None
    files_created = []
    event_count_total = 0
    processed_successfully = False

    try:
        with hep.open(str(input_hepmc_path)) as f_in:
            desc = f"Splitting {input_hepmc_path.name}"
            iterator = tqdm_actual(enumerate(f_in), desc=desc)

            for i, event in iterator:                
                if i % events_per_file == 0:
                    if current_f_out:
                        current_f_out.close()
                    
                    file_idx = i // events_per_file
                    # Use just the run number as the directory name
                    current_split_output_dir = final_output_base_dir / str(file_idx)
                    os.makedirs(current_split_output_dir, exist_ok=True)
                    
                    split_file_path = current_split_output_dir / output_filename
                    current_f_out = WriterAscii(str(split_file_path)) # Ensure path is string for WriterAscii
                    files_created.append(split_file_path)
                
                if current_f_out:
                    event.event_number = i % events_per_file
                    current_f_out.write_event(event)
                event_count_total = i + 1
        
        processed_successfully = True # If loop completes without error

    except Exception as e:
        print(f"Error during HEPMC splitting of {input_hepmc_path}: {e}")
    finally:
        if current_f_out:
            try:
                current_f_out.close()
            except Exception as e_close:
                print(f"Error closing output file during HEPMC splitting: {e_close}")
        
    if processed_successfully and event_count_total > 0:
        print(f"--- Splitting complete. Processed {event_count_total} events from {input_hepmc_path.name} into {len(files_created)} files. ---")
        return files_created
    elif processed_successfully and event_count_total == 0:
        print(f"--- No events found or processed in {input_hepmc_path.name}. ---")
        return []
    else: # Not processed_successfully
        print(f"--- Splitting failed for {input_hepmc_path.name}. ---")
        # Clean up any partially created files from this attempt if needed, though current logic doesn't require it.
        return []

def normalize_card_customizations(config):
    """Ensure card_customizations exists and all card types are dicts (not None)."""
    if not hasattr(config, 'card_customizations') or config.card_customizations is None:
        config.card_customizations = {}
    
    # Ensure each card type is a dict, not None
    for card_type in ['run_card', 'shower_card', 'pythia8_card']:
        if card_type not in config.card_customizations or config.card_customizations[card_type] is None:
            config.card_customizations[card_type] = {}

def setup_splitting_config(config, logger):
    """Set up HEPMC splitting configuration."""
    try:
        splitting_config = getattr(config, 'splitting_config', {})
        if isinstance(splitting_config, dict):
            splitting_enabled = splitting_config.get('enable', False)
        else:
            splitting_enabled = getattr(splitting_config, 'enable', False)
        
        split_events_per_file = splitting_config.get('events_per_file', 1000)
        split_output_filename = splitting_config.get('output_filename', 'events.hepmc')
        logger.info(f"Splitting config: enable={splitting_enabled}, events_per_file={split_events_per_file}")
        
        return splitting_enabled, split_events_per_file, split_output_filename
    except Exception as e:
        logger.warning(f"Error accessing splitting config: {e}. Using defaults")
        return False, 1000, 'events.hepmc'

def create_launch_script(temp_launch_script_path, copied_process_dir, process_type, run_name, config, logger):
    """Create the MadGraph launch script based on process type."""
    run_card_settings = config.card_customizations['run_card']
    parton_shower_mode = run_card_settings.get('parton_shower', 'OFF').upper()

    with open(temp_launch_script_path, 'w') as f:
        if process_type == "noborn" and parton_shower_mode == 'PYTHIA8':
            logger.info("Loop-induced process with parton showering detected. Generating multi-step launch script.")
            if run_name:
                f.write(f"launch {copied_process_dir} --name={run_name}\n")
            else:
                f.write(f"launch {copied_process_dir}\n")
            f.write("shower=Pythia8\n")
            f.write("0\n")  # Skips card editing prompt, starts event generation
        else:
            logger.info("Standard NLO process detected. Generating simple launch script.")
            if run_name:
                f.write(f"launch {copied_process_dir} -f --name={run_name}\n")
            else:
                f.write(f"launch {copied_process_dir} -f\n")
            f.write("set automatic_html_opening False\n")
            f.write("exit\n")

    logger.info(f"Process type: {process_type}, Run name: {run_name}")

def run_warmup_in_place(config, mg5_exe, logger):
    """Run a tiny in-place warmup (1 event, seed=42) in central madgraph_process, then exit.

    This builds/validates integration grids and compiled objects to be reused by
    subsequent distributed generation runs.
    """
    version_dir = get_version_directory_path(config)
    stored_process_dir = version_dir / "madgraph_process"
    if not stored_process_dir.exists():
        raise FileNotFoundError(
            f"madgraph_process not found at {stored_process_dir}. Run madgraph_init first."
        )

    # Temporarily override events/seed
    original_events = getattr(config, 'events', None)
    original_seed = getattr(config, 'seed', None)
    try:
        config.events = 1
        config.seed = 42

        process_type = customize_cards_for_run(stored_process_dir, config, run_id='warmup')

        temp_launch_script_path = version_dir / "warmup_launch.mg5"
        create_launch_script(temp_launch_script_path, stored_process_dir, process_type, 'warmup', config, logger)

        logger.info(f"Executing MadGraph warmup with script: {temp_launch_script_path}")
        stdout_event, stderr_event = run_command([str(mg5_exe), str(temp_launch_script_path)], cwd=stored_process_dir)
        logger.info("--- Warmup STDOUT: ---")
        logger.info(stdout_event)
        if stderr_event:
            logger.warning("--- Warmup STDERR: ---")
            logger.warning(stderr_event)
        logger.info("=== Warmup complete. Exiting without file moves/splitting. ===")
    finally:
        # Restore config
        if original_events is not None:
            config.events = original_events
        else:
            if hasattr(config, 'events'):
                delattr(config, 'events')
        if original_seed is not None:
            config.seed = original_seed
        else:
            if hasattr(config, 'seed'):
                delattr(config, 'seed')

def process_output_files(copied_process_dir, effective_output_dir, run_name, splitting_enabled, 
                        split_events_per_file, split_output_filename, logger):
    """Process and move/split output files from MadGraph."""
    events_dir_in_process = copied_process_dir / "Events"
    
    # Look for run-specific directories
    if run_name:
        actual_events_subdirs = list(events_dir_in_process.glob(run_name))
    else:
        actual_events_subdirs = list(events_dir_in_process.glob("run_*"))
        
    # Fallback to main Events directory if no run subdirs found  
    if not actual_events_subdirs:
        if events_dir_in_process.is_dir():
             actual_events_subdirs = [events_dir_in_process]
        else:
            logger.warning(f"Events directory {events_dir_in_process} not found.")
            actual_events_subdirs = []

    files_processed_count = 0
    logger.info(f"Processing events from {len(actual_events_subdirs)} directories")

    for events_subdir_path in actual_events_subdirs:
        if events_subdir_path.is_dir():
            # Process LHE files: move them directly
            for pattern in ["*.lhe", "*.lhe.gz"]:
                for event_file_path in events_subdir_path.glob(pattern):
                    try:
                        destination_path = effective_output_dir / event_file_path.name
                        shutil.move(str(event_file_path), str(destination_path))
                        logger.info(f"Moved LHE file {event_file_path.name} to {destination_path}")
                        files_processed_count += 1
                    except Exception as e:
                        logger.error(f"Error moving LHE file {event_file_path.name}: {e}")

            # Process HepMC files: split if enabled, otherwise move
            for pattern in ["*.hepmc", "*.hepmc.gz"]:
                for event_file_path in events_subdir_path.glob(pattern):
                    if splitting_enabled:
                        logger.info(f"Processing HEPMC file for splitting: {event_file_path}")
                        created_split_files = split_hepmc_file(
                            input_hepmc_path=event_file_path,
                            final_output_base_dir=effective_output_dir,
                            events_per_file=split_events_per_file,
                            output_filename=split_output_filename
                        )
                        if created_split_files:
                            files_processed_count += len(created_split_files)
                            try:
                                event_file_path.unlink()
                                logger.info(f"Removed original temporary HEPMC file {event_file_path} after successful splitting.")
                            except OSError as e:
                                logger.warning(f"Could not remove original temporary HEPMC file {event_file_path}: {e}")
                        else:
                            logger.warning(f"HEPMC splitting produced no files for {event_file_path} or was skipped. Attempting to move original.")
                            try:
                                destination_path = effective_output_dir / event_file_path.name
                                shutil.move(str(event_file_path), str(destination_path))
                                logger.info(f"Moved original HEPMC file {event_file_path.name} to {destination_path} (splitting failed/skipped).")
                                files_processed_count += 1
                            except Exception as e:
                                logger.error(f"Error moving original HEPMC file {event_file_path.name} after failed/skipped split: {e}")
                    else:
                        # Splitting not enabled, move the HEPMC file directly
                        try:
                            if event_file_path.name.endswith('.hepmc.gz'):
                                destination_path = effective_output_dir / "events.hepmc.gz"
                            else:
                                destination_path = effective_output_dir / "events.hepmc3"
                            shutil.move(str(event_file_path), str(destination_path))
                            logger.info(f"Moved HEPMC file {event_file_path.name} to {destination_path} (splitting disabled, ACTS-compatible name).")
                            files_processed_count += 1
                        except Exception as e:
                            logger.error(f"Error moving HEPMC file {event_file_path.name}: {e}")
    
    if files_processed_count == 0:
        logger.warning("No event files were found, moved, or split from the MadGraph run.")

def copy_final_cards(copied_process_dir, process_type, process_name, effective_output_dir, config, logger):
    """
    Copy final cards to both the per-run output directory and central version directory.
    """
    try:
        cards_dir = copied_process_dir / "Cards"
        final_run_card_path = cards_dir / "run_card.dat"
        
        # Determine card files to copy based on process type
        cards_to_copy = [("run_card.dat", "run_card")]
        
        if process_type == "born":
            cards_to_copy.append(("shower_card.dat", "shower_card"))
        elif process_type == "noborn":
            cards_to_copy.append(("pythia8_card.dat", "pythia8_card"))
        
        # Copy to per-run directory (runs/X/final_cards/)
        run_cards_dir = effective_output_dir / "final_cards"
        run_cards_dir.mkdir(exist_ok=True)
        
        # Copy to central version directory (version/final_cards/)
        version_dir = get_version_directory_path(config)
        central_cards_dir = version_dir / "final_cards"
        central_cards_dir.mkdir(exist_ok=True)
        
        for card_filename, card_type in cards_to_copy:
            card_path = cards_dir / card_filename
            
            if card_path.exists():
                # Copy to run-specific directory with simple name
                run_destination = run_cards_dir / card_filename
                shutil.copy(card_path, run_destination)
                logger.info(f"Copied final {card_type} to run directory: {run_destination}")
                
                # Copy to central directory with process name prefix
                central_destination = central_cards_dir / f"{process_name}_{card_filename}"
                shutil.copy(card_path, central_destination)
                logger.info(f"Copied final {card_type} to central directory: {central_destination}")
            else:
                logger.warning(f"Final {card_filename} not found at {card_path}. Cannot copy.")
                
    except Exception as e:
        logger.warning(f"Could not copy final MadGraph cards: {e}")

def main():
    """
    Main entry point for MadGraph event generation.
    This assumes that madgraph_init has already been run and the process directory exists.
    """
    # Setup and validation
    parser = create_base_parser("MadGraph event generation for ColliderML")
    args = parser.parse_args()
    config = load_config(args)

    # Set up logging
    log_level = getattr(config, 'log_level', 'INFO')
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger.info("Starting MadGraph event generation")

    # Validate required configuration
    required_fields = ['mg_base_path', 'generation_scratch_dir', 'events']
    for field in required_fields:
        if not hasattr(config, field) or getattr(config, field) is None:
            logger.error(f"Required configuration field missing: {field}")
            sys.exit(1)

    # Basic setup
    process_name = f"{config.dataset}_{config.version}"
    mg_base_path = Path(config.mg_base_path)
    mg5_exe = mg_base_path / "bin" / "mg5_aMC"
    
    # Fix card_customizations None issue
    normalize_card_customizations(config)

    # Setup output directory
    effective_output_dir = Path(args.output)
    if args.output_subdir:
        effective_output_dir = effective_output_dir / args.output_subdir
    effective_output_dir.mkdir(parents=True, exist_ok=True)

    # Setup splitting configuration
    splitting_enabled, split_events_per_file, split_output_filename = setup_splitting_config(config, logger)

    # Warmup toggle: support either a simple boolean `warmup: true` or legacy dict with `enable`
    warmup_attr = getattr(config, 'warmup', False)
    warmup_enabled = False
    if isinstance(warmup_attr, dict):
        warmup_enabled = bool(warmup_attr.get('enable', False))
    else:
        warmup_enabled = bool(warmup_attr)

    try:
        # Optional warmup path: run in-place on central process dir to build grids, then exit
        if warmup_enabled:
            logger.info("=== WARMUP: In-place grid build in central madgraph_process ===")
            run_warmup_in_place(config, mg5_exe, logger)
            return

        # STEP 1: Copy pre-compiled process directory
        logger.info("=== STEP 1: Copy Pre-compiled Process Directory ===")
        copied_process_dir, job_scratch_dir = copy_process_directory(config)
        
        # STEP 2: Customize cards for this specific run
        logger.info("=== STEP 2: Customize Cards for Run ===")
        run_id = os.environ.get('SLURM_PROCID') or os.environ.get('SLURM_ARRAY_TASK_ID') or '0'
        process_type = customize_cards_for_run(copied_process_dir, config, run_id)
        
        # STEP 3: Launch MadGraph event generation
        logger.info("=== STEP 3: Launch MadGraph Event Generation ===")
        temp_launch_script_path = job_scratch_dir / "launch_script.mg5"
        run_name = f"run_{run_id}" if run_id else None
        
        create_launch_script(temp_launch_script_path, copied_process_dir, process_type, run_name, config, logger)
        
        logger.info(f"Executing MadGraph with script: {temp_launch_script_path}")
        stdout_event, stderr_event = run_command([str(mg5_exe), str(temp_launch_script_path)], cwd=copied_process_dir)
        logger.info("--- MadGraph event generation STDOUT: ---")
        logger.info(stdout_event)
        if stderr_event:
            logger.warning("--- MadGraph event generation STDERR: ---")
            logger.warning(stderr_event)
        logger.info("--- MadGraph event generation complete. ---")

        # STEP 4: Process and move/split output files
        logger.info("=== STEP 4: Process and Move/Split Output Files ===")
        process_output_files(copied_process_dir, effective_output_dir, run_name, 
                           splitting_enabled, split_events_per_file, split_output_filename, logger)

        # STEP 5: Copy final cards
        logger.info("=== STEP 5: Copy Final Cards ===")
        copy_final_cards(copied_process_dir, process_type, process_name, effective_output_dir, config, logger)

        logger.info("=== MadGraph event generation completed successfully ===")
        
    except Exception as e:
        logger.error(f"Fatal error during MadGraph event generation: {e}")
        raise
    finally:
        # Cleanup
        if 'job_scratch_dir' in locals() and job_scratch_dir.exists():
            try:
                logger.info(f"Cleaning up temporary directory: {job_scratch_dir}")
                # shutil.rmtree(job_scratch_dir)
                logger.info("Cleanup completed successfully")
            except Exception as cleanup_error:
                logger.warning(f"Error during cleanup: {cleanup_error}")

    logger.info("--- MadGraph generation pipeline finished ---")

if __name__ == "__main__":
    main() 