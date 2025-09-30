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
from tqdm import tqdm 
from pathlib import Path
from utils.config import create_base_parser, load_config
from utils.madgraph_utils import (
    run_command,
    customize_card_with_regex,
    get_version_directory_path
)

logger = logging.getLogger(__name__)

# run_command is imported from madgraph_utils

# customize_card_with_regex and detect_process_type_from_files are imported from madgraph_utils

def stage_tarball_to_scratch(config):
    """
    Copy the tarball artifact to a unique scratch directory and extract it.

    Returns:
        tuple: (extracted_process_dir_path, job_scratch_dir_path)
    """
    version_dir = get_version_directory_path(config)
    tarball_path = version_dir / "madgraph_process.tgz"
    if not tarball_path.exists():
        raise FileNotFoundError(
            f"Tarball artifact not found at {tarball_path}. Run madgraph_init to create it."
        )

    scratch_dir = Path(config.generation_scratch_dir)
    process_name = f"{config.dataset}_{config.version}"
    # Make this unique per process/task to avoid collisions
    uniq = os.environ.get('SLURM_JOB_ID') or 'nojid'
    proc = os.environ.get('SLURM_PROCID') or os.getpid()
    job_scratch_dir = scratch_dir / f"mg5_gen_{process_name}_{uniq}_{proc}"
    job_scratch_dir.mkdir(parents=True, exist_ok=True)

    # Copy tarball locally per job for safe concurrent access
    local_tarball = job_scratch_dir / tarball_path.name
    logger.info(f"Copying tarball: {tarball_path} -> {local_tarball}")
    shutil.copy2(tarball_path, local_tarball)

    # Extract into 'process' subdir
    copied_process_dir = job_scratch_dir / "process"
    copied_process_dir.mkdir(exist_ok=True)
    logger.info(f"Extracting {local_tarball} to {copied_process_dir}")
    run_command(["tar", "-xzf", str(local_tarball), "-C", str(copied_process_dir)])

    # The tar contains a top-level directory named 'madgraph_process'
    # Normalize to point at that directory
    candidate = copied_process_dir / "madgraph_process"
    if candidate.is_dir():
        copied_process_dir = candidate
    else:
        # Fallback: try single subdir
        subdirs = [p for p in copied_process_dir.iterdir() if p.is_dir()]
        if len(subdirs) == 1:
            copied_process_dir = subdirs[0]

    # Log extracted statistics
    total_size = sum(f.stat().st_size for f in copied_process_dir.rglob('*') if f.is_file())
    total_size_mb = total_size / (1024 * 1024)
    file_count = len(list(copied_process_dir.rglob('*')))
    logger.info(f"Process directory extracted: {total_size_mb:.1f} MB, {file_count} files")

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
        None
    """
    cards_dir = process_dir / "Cards"
    
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
    
    # Customize shower/pythia cards routed by run_mode only
    run_mode = getattr(config, 'run_mode', None)
    if str(run_mode).lower() == 'lo_mlm':
        # Always apply pythia8_card customizations for LO+MLM (MLM JetMatching lives here)
        pythia8_card_path = cards_dir / "pythia8_card.dat"
        if pythia8_card_path.exists():
            p8_params = {}
            if hasattr(config, 'events'):
                p8_params['Main:numberOfEvents'] = config.events
            if hasattr(config, 'seed'):
                p8_params['Random:seed'] = config.seed
            base_p8 = config.card_customizations['pythia8_card']
            final_p8 = {**base_p8, **p8_params}
            if final_p8:
                logger.info("Customizing pythia8_card.dat for run_mode=lo_mlm")
                customize_card_with_regex(pythia8_card_path, final_p8)
    else:
        # NLO/FxFx path: customize shower_card if present, or pythia8_card if present
        shower_card_path = cards_dir / "shower_card.dat"
        if shower_card_path.exists():
            shower_params = {}
            if hasattr(config, 'events'):
                shower_params['nevents'] = config.events
            if hasattr(config, 'seed'):
                shower_params['rnd_seed'] = config.seed
            base_shower_settings = config.card_customizations['shower_card']
            final_shower_settings = {**base_shower_settings, **shower_params}
            if final_shower_settings:
                logger.info("Customizing shower_card.dat for NLO process")
                customize_card_with_regex(shower_card_path, final_shower_settings)
        else:
            pythia8_card_path = cards_dir / "pythia8_card.dat"
            if pythia8_card_path.exists():
                pythia8_params = {}
                if hasattr(config, 'events'):
                    pythia8_params['Main:numberOfEvents'] = config.events
                if hasattr(config, 'seed'):
                    pythia8_params['Random:seed'] = config.seed
                base_pythia8_settings = config.card_customizations['pythia8_card']
                final_pythia8_settings = {**base_pythia8_settings, **pythia8_params}
                if final_pythia8_settings:
                    logger.info("Customizing pythia8_card.dat for loop-induced/NLO fallback")
                    customize_card_with_regex(pythia8_card_path, final_pythia8_settings)

    # No return value needed any more


def split_hepmc_file(input_hepmc_path: Path,
                     final_output_base_dir: Path,
                     events_per_file: int,
                     output_filename: str = "events.hepmc",
                     global_run_offset: int = 0,
                     max_files_per_mg_run: int = None):
    """
    Splits a single (potentially gzipped) HEPMC file into multiple smaller HEPMC files,
    each in its own subdirectory (0, 1, 2, etc.) under final_output_base_dir.
    
    Args:
        global_run_offset: Offset to add to chunk indices for global run numbering (multi-node mode)
        max_files_per_mg_run: Maximum number of split files to keep per MadGraph run (caps output)
    """
    try:
        import pyhepmc as hep
        from pyhepmc.io import WriterAscii
    except ImportError:
        print("Error: pyhepmc library not found. Please install it (e.g., pip install pyhepmc). Skipping split.")
        return []

    if events_per_file <= 0:
        print("Warning: events_per_file must be > 0. Skipping split.")
        return []

    print(f"--- Splitting HEPMC file: {input_hepmc_path} ---")
    print(f"--- Output base directory for splits: {final_output_base_dir} ---")
    print(f"--- Events per split file: {events_per_file} ---")

    files_created = []
    current_writer = None
    total_events = 0

    try:
        with hep.open(str(input_hepmc_path)) as f_in:
            iterator = tqdm(enumerate(f_in), desc=f"Splitting {input_hepmc_path.name}")

            for event_index, event in iterator:
                if event_index % events_per_file == 0:
                    if current_writer is not None:
                        try:
                            current_writer.close()
                        except Exception:
                            pass
                    chunk_index = event_index // events_per_file
                    global_run_index = global_run_offset + chunk_index
                    split_dir = final_output_base_dir / str(global_run_index)
                    os.makedirs(split_dir, exist_ok=True)
                    split_path = split_dir / output_filename
                    current_writer = WriterAscii(str(split_path))
                    files_created.append(split_path)

                event.event_number = event_index % events_per_file
                current_writer.write_event(event)
                total_events = event_index + 1
    except Exception as e:
        print(f"Error during HEPMC splitting of {input_hepmc_path}: {e}")
    finally:
        if current_writer is not None:
            try:
                current_writer.close()
            except Exception:
                pass

    if total_events == 0:
        print(f"--- No events found or processed in {input_hepmc_path.name}. ---")
        return []

    # Discard final partial chunk
    remainder = total_events % events_per_file
    if remainder != 0 and len(files_created) > 0:
        last_path = files_created[-1]
        try:
            Path(last_path).unlink(missing_ok=True)
        except Exception as e_rm:
            print(f"Warning: failed to remove partial split file {last_path}: {e_rm}")
        try:
            Path(last_path).parent.rmdir()
        except Exception:
            pass
        files_created.pop()
        print(f"--- Discarded final partial chunk with {remainder} events (< {events_per_file}). ---")

    # Cap to max_files_per_mg_run if specified (for multi-node deterministic output)
    if max_files_per_mg_run is not None and len(files_created) > max_files_per_mg_run:
        excess_files = files_created[max_files_per_mg_run:]
        files_created = files_created[:max_files_per_mg_run]
        print(f"--- Capping output to {max_files_per_mg_run} files per MG run (discarding {len(excess_files)} excess files). ---")
        for excess_path in excess_files:
            try:
                Path(excess_path).unlink(missing_ok=True)
                Path(excess_path).parent.rmdir()
            except Exception as e_rm:
                print(f"Warning: failed to remove excess file {excess_path}: {e_rm}")

    print(f"--- Splitting complete. Processed {total_events} events from {input_hepmc_path.name} into {len(files_created)} files. ---")
    return files_created

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
        max_files_per_mg_run = splitting_config.get('max_files_per_mg_run', None)
        
        logger.info(f"Splitting config: enable={splitting_enabled}, events_per_file={split_events_per_file}, max_files_per_mg_run={max_files_per_mg_run}")
        
        return splitting_enabled, split_events_per_file, split_output_filename, max_files_per_mg_run
    except Exception as e:
        logger.warning(f"Error accessing splitting config: {e}. Using defaults")
        return False, 1000, 'events.hepmc', None

def run_generate_events_no_compile(process_dir: Path, run_name: str, logger: logging.Logger):
    """Run ./bin/generate_events in no-compile mode (-oxf) with optional run name."""
    exe = process_dir / "bin" / "generate_events"
    cmd = [str(exe), "-oxf"]
    if run_name:
        cmd.extend(["--name", run_name])
    logger.info(f"Executing no-compile generate_events (streaming): {' '.join(cmd)}")
    try:
        run_command(cmd, cwd=str(process_dir), stream=True, capture=False, merge_streams=True, logger=logger)
    except Exception as e:
        logger.error(f"Error running generate_events (no-compile): {e}")
        raise

    

def run_generate_events_compile(process_dir: Path, run_name: str, logger: logging.Logger):
    """Run ./bin/generate_events in compile mode with explicit run name (LO+MLM fast path)."""
    exe = process_dir / "bin" / "generate_events"
    cmd = [str(exe), run_name, "-f"]
    logger.info(f"Executing generate_events (compile) for LO+MLM (streaming): {' '.join(cmd)}")
    try:
        run_command(cmd, cwd=str(process_dir), stream=True, capture=False, merge_streams=True, logger=logger)
    except Exception as e:
        logger.error(f"Error running generate_events (compile): {e}")
        raise


def run_lo_mlm_with_mg5_script(process_dir: Path, mg5_exe: Path, job_scratch_dir: Path, config, logger: logging.Logger):
    """Launch LO+MLM with Pythia8 non-interactively via mg5 script.

    This avoids interactive prompts by providing a small command file to mg5_aMC.
    We rely on pre-customized cards on disk and only direct MG to use Pythia8.
    """
    script_path = job_scratch_dir / "mg5_launch_lo_mlm.txt"
    logger.info(f"Writing MG5 non-interactive launch script at {script_path}")

    lines = []
    # Point MG5 to the already prepared/compiled process directory
    lines.append(f"launch {process_dir}")
    # Force Pythia8 shower step without interactive prompt
    lines.append("shower=PYTHIA8")

    # Pass run-scoped controls (events/seed) in case MG5 reads them here
    if hasattr(config, 'events') and config.events is not None:
        lines.append(f"set nevents {config.events}")
    if hasattr(config, 'seed') and config.seed is not None:
        lines.append(f"set iseed {config.seed}")

    try:
        with open(script_path, 'w') as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        logger.error(f"Failed to write MG5 script: {e}")
        raise

    cmd = [str(mg5_exe), str(script_path)]
    logger.info(f"Executing MG5 non-interactive LO+MLM launch: {' '.join(cmd)}")
    try:
        run_command(cmd, stream=True, capture=False, merge_streams=True, logger=logger)
    except Exception as e:
        logger.error(f"Error running MG5 LO+MLM launch: {e}")
        raise

def process_output_files(copied_process_dir, staging_output_dir, run_name, splitting_enabled,
                        split_events_per_file, split_output_filename, mg_run_id, events_per_mg_run, 
                        max_files_per_mg_run, logger):
    """Process and move/split output files from MadGraph.
    
    Args:
        staging_output_dir: Directory for MG staging files (LHE, cards). For multi-node: runs/all/X/
    """
    events_dir_in_process = copied_process_dir / "Events"
    
    # Determine split output directory and global run offset
    # Multi-node SLURM: staging_output_dir = runs/all/X/ → split to runs/ with offset
    # Monolithic/Interactive: staging_output_dir = runs/ → split directly there
    is_multinode = mg_run_id is not None and mg_run_id >= 0
    
    if is_multinode:
        # Multi-node: staging is runs/all/X/, split goes to runs/
        # Navigate: runs/all/X/ -> runs/all/ -> runs/
        split_output_base_dir = staging_output_dir.parent.parent
        # Calculate global run offset for this MadGraph job
        # Use max_files_per_mg_run if capping is enabled, otherwise use theoretical file count
        if max_files_per_mg_run is not None:
            runs_per_mg_job = max_files_per_mg_run
        else:
            runs_per_mg_job = events_per_mg_run // split_events_per_file if events_per_mg_run and split_events_per_file else 0
        global_run_offset = mg_run_id * runs_per_mg_job
        logger.info(f"Multi-node mode: MG run {mg_run_id}, staging to {staging_output_dir}, {runs_per_mg_job} files/run, global offset {global_run_offset}, splitting to {split_output_base_dir}")
    else:
        # Monolithic/interactive: use staging_output_dir as-is
        split_output_base_dir = staging_output_dir
        global_run_offset = 0
        logger.info(f"Monolithic mode: staging and splitting to {split_output_base_dir}")
    
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
            # Process LHE files: move them to staging directory
            for pattern in ["*.lhe", "*.lhe.gz"]:
                for event_file_path in events_subdir_path.glob(pattern):
                    try:
                        destination_path = staging_output_dir / event_file_path.name
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
                            final_output_base_dir=split_output_base_dir,
                            events_per_file=split_events_per_file,
                            output_filename=split_output_filename,
                            global_run_offset=global_run_offset,
                            max_files_per_mg_run=max_files_per_mg_run
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
                                destination_path = split_output_base_dir / event_file_path.name
                                shutil.move(str(event_file_path), str(destination_path))
                                logger.info(f"Moved original HEPMC file {event_file_path.name} to {destination_path} (splitting failed/skipped).")
                                files_processed_count += 1
                            except Exception as e:
                                logger.error(f"Error moving original HEPMC file {event_file_path.name} after failed/skipped split: {e}")
                    else:
                        # Splitting not enabled, move the HEPMC file to staging directory
                        try:
                            if event_file_path.name.endswith('.hepmc.gz'):
                                destination_path = staging_output_dir / "events.hepmc.gz"
                            else:
                                destination_path = staging_output_dir / "events.hepmc3"
                            shutil.move(str(event_file_path), str(destination_path))
                            logger.info(f"Moved HEPMC file {event_file_path.name} to {destination_path} (splitting disabled, ACTS-compatible name).")
                            files_processed_count += 1
                        except Exception as e:
                            logger.error(f"Error moving HEPMC file {event_file_path.name}: {e}")
    
    if files_processed_count == 0:
        logger.warning("No event files were found, moved, or split from the MadGraph run.")

def copy_final_cards(copied_process_dir, process_type, process_name, staging_output_dir, config, logger):
    """
    Copy final cards to both the per-run output directory and central version directory.
    
    Args:
        staging_output_dir: Directory for MG staging files (for multi-node: runs/all/X/)
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
        
        # Copy to per-run staging directory (runs/all/X/final_cards/)
        run_cards_dir = staging_output_dir / "final_cards"
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
    splitting_enabled, split_events_per_file, split_output_filename, max_files_per_mg_run = setup_splitting_config(config, logger)

    # Warmup toggle: support either a simple boolean `warmup: true` or legacy dict with `enable`
    warmup_attr = getattr(config, 'warmup', False)
    warmup_enabled = False
    if isinstance(warmup_attr, dict):
        warmup_enabled = bool(warmup_attr.get('enable', False))
    else:
        warmup_enabled = bool(warmup_attr)

    try:
        # No warmup: grids are built during madgraph_init

        # STEP 1: Stage tarball to scratch and extract
        logger.info("=== STEP 1: Stage Tarball to Scratch and Extract ===")
        copied_process_dir, job_scratch_dir = stage_tarball_to_scratch(config)
        
        # STEP 2: Customize cards for this specific run
        logger.info("=== STEP 2: Customize Cards for Run ===")
        run_id = os.environ.get('SLURM_PROCID') or os.environ.get('SLURM_ARRAY_TASK_ID') or '0'
        customize_cards_for_run(copied_process_dir, config, run_id)
        
        # STEP 3: Launch MadGraph event generation
        run_mode = getattr(config, 'run_mode', 'nlo_fxfx')
        if str(run_mode).lower() == 'lo_mlm':
            logger.info("=== STEP 3: Launch MadGraph Event Generation (compile, LO+MLM) ===")
            # Use MG5 script pathway to trigger Pythia8 shower non-interactively
            run_name = None
            run_lo_mlm_with_mg5_script(copied_process_dir, mg5_exe, job_scratch_dir, config, logger)
        else:
            logger.info("=== STEP 3: Launch MadGraph Event Generation (no-compile, NLO/FxFx) ===")
            run_name = "run_build"
            run_generate_events_no_compile(copied_process_dir, run_name, logger)

        # STEP 4: Process and move/split output files
        logger.info("=== STEP 4: Process and Move/Split Output Files ===")
        
        # Detect multi-node mode: if output_subdir is numeric, extract mg_run_id
        mg_run_id = None
        staging_output_dir = effective_output_dir  # Default: same as effective_output_dir
        
        if args.output_subdir and args.output_subdir.isdigit():
            mg_run_id = int(args.output_subdir)
            logger.info(f"Detected multi-node SLURM mode with MG run ID: {mg_run_id}")
            
            # For multi-node: stage MG files to runs/all/X/ to avoid collision with split runs/X/
            staging_output_dir = effective_output_dir.parent / "all" / args.output_subdir
            staging_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Multi-node staging directory: {staging_output_dir}")
        
        events_per_mg_run = getattr(config, 'events', None)
        
        process_output_files(copied_process_dir, staging_output_dir, run_name, 
                           splitting_enabled, split_events_per_file, split_output_filename,
                           mg_run_id, events_per_mg_run, max_files_per_mg_run, logger)

        # STEP 5: Copy final cards
        logger.info("=== STEP 5: Copy Final Cards ===")
        # Decide which final card(s) to copy based on run_mode
        process_type = 'born' if str(getattr(config, 'run_mode', 'nlo_fxfx')).lower() != 'lo_mlm' else 'noborn'
        copy_final_cards(copied_process_dir, process_type, process_name, staging_output_dir, config, logger)

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