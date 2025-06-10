import time
from pathlib import Path
import acts
import acts.examples
from acts.examples import Sequencer
from acts.examples.simulation import addPythia8
from acts.examples.hepmc3 import (
        HepMC3AsciiWriter,
    )
import traceback
# Removed pyhepmc, numpy imports as they will be handled by the imported module

from utils.app_logging import setup_logging, TimingRecorder
from utils.config import create_base_parser, load_config

# Import merging and smearing functions
from merge_and_smear_madgraph import merge_hepmc_files as merge_external_signal_with_pileup

u = acts.UnitConstants

def parse_args():
    """Parse command line arguments"""
    parser = create_base_parser("Pythia8 event generation for ACTS")
    parser.add_argument(
        "--pileup",
        help="Number of pile-up events",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--hard-process",
        help="Pythia8 hard process",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--pythia-settings",
        help="Additional Pythia8 settings (comma-separated)",
        type=str,
        default=None,
    )
    # Merge signal flag
    parser.add_argument(
        "--merge-signal",
        help="Merge with existing signal file (events.hepmc3 or events.hepmc.gz) in output directory",
        action="store_true",
        default=None,
    )
    # Vertex smearing parameters 
    parser.add_argument(
        "--vertex-sigma-xy",
        help="Sigma for vertex smearing in x/y [mm]",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--vertex-sigma-z",
        help="Sigma for vertex smearing in z [mm]",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--vertex-sigma-t",
        help="Sigma for vertex smearing in time [ns]",
        type=float,
        default=None,
    )
    return parser.parse_args()

def create_vertex_generator(config, logger):
    """Create ACTS GaussianVertexGenerator from config parameters.
    
    Args:
        config: Configuration object with vertex_sigma_xy, vertex_sigma_z, vertex_sigma_t
        logger: Logger instance
        
    Returns:
        acts.examples.GaussianVertexGenerator or None if no smearing configured
    """
    logger.info(f"Configuring vertex smearing with sigma_xy={config.vertex_sigma_xy} mm, "
                f"sigma_z={config.vertex_sigma_z} mm, sigma_t={config.vertex_sigma_t} ns")
    
    if any(sigma is not None and sigma != 0 for sigma in [
        config.vertex_sigma_xy,
        config.vertex_sigma_z,
        config.vertex_sigma_t
    ]):
        return acts.examples.GaussianVertexGenerator(
            stddev=acts.Vector4(
                config.vertex_sigma_xy * u.mm,
                config.vertex_sigma_xy * u.mm,
                config.vertex_sigma_z * u.mm,
                config.vertex_sigma_t * u.ns
            ),
            mean=acts.Vector4(0, 0, 0, 0),
        )
    return None

def parse_pythia_settings(config, logger):
    """Parse Pythia8 settings from config and hard process.
    
    Args:
        config: Configuration object with pythia_settings and hard_process
        logger: Logger instance
        
    Returns:
        list: Pythia8 settings or None if no settings configured
    """
    pythia_settings = []
    
    # Add all settings from config if present
    if getattr(config, 'pythia_settings', None):
        if isinstance(config.pythia_settings, str):
            logger.debug("Processing command-line pythia settings")
            pythia_settings.extend([s.strip() for s in config.pythia_settings.split(',') if s.strip()])
        elif isinstance(config.pythia_settings, list):
            logger.debug("Processing YAML list pythia settings")
            pythia_settings.extend(config.pythia_settings)
        else:
            raise ValueError("pythia_settings must be either a comma-separated string or a list")
    
    # Always append the hard process at the end
    hard_process = getattr(config, 'hard_process', None)
    if isinstance(hard_process, str) and hard_process.strip():
        pythia_settings.append(f"{hard_process.strip()}=on")

    return pythia_settings if pythia_settings else None

def generate_pileup_events(output_dir, config, logger):
    """Generate pileup-only events with vertex smearing applied during generation.
    
    Args:
        output_dir: Path to output directory
        config: Configuration object
        logger: Logger instance
        
    Returns:
        Path: Path to generated pileup file (events_pileup.hepmc3)
    """
    logger.info(f"Generating {config.events} pileup events for merging.")
    
    # Create sequencer for pileup generation
    s_pileup = Sequencer(numThreads=1, events=config.events)
    s_pileup.config.logLevel = acts.logging.DEBUG
    seed_pileup = config.seed or int(time.time())
    logger.info(f"Using random seed for pileup: {seed_pileup}")
    rnd_pileup = acts.examples.RandomNumbers(seed=seed_pileup)
    
    # Apply vertex smearing to pileup during generation for efficiency
    vtxGen_pileup = create_vertex_generator(config, logger)
    
    # Pileup events output path
    pileup_path = output_dir / "events_pileup.hepmc3"
    
    logger.debug("Creating Pythia8 pileup-only generator...")
    addPythia8(
        s_pileup,
        npileup=config.pileup,
        nhard=0,
        hardProcess=None,
        outputDirCsv=None,
        outputDirRoot=None,
        outputEvent="pileup_events",
        rnd=rnd_pileup,
        logLevel=acts.logging.DEBUG,
        vtxGen=vtxGen_pileup,  # Apply vertex smearing during pileup generation
    )
    logger.debug("Pythia8 pileup-only generator created successfully.")

    s_pileup.addWriter(
        HepMC3AsciiWriter(
            acts.logging.INFO,
            inputEvent="pileup_events",
            outputPath=pileup_path,
        )
    )

    logger.info(f"Writing pileup events to {pileup_path}")
    s_pileup.run()
    logger.info("Pileup generation completed.")
    
    return pileup_path

def merge_signal_and_pileup(signal_file_path, pileup_path, output_dir, config, logger):
    """Merge pre-existing signal file with generated pileup events.
    
    Args:
        signal_file_path: Path to signal file (events.hepmc3 or events.hepmc.gz)
        pileup_path: Path to pileup file (events_pileup.hepmc3)
        output_dir: Output directory
        config: Configuration object
        logger: Logger instance
        
    Returns:
        Path: Path to merged output file (merged_events.hepmc3)
    """
    # For merge: only smear signal events (pileup is already smeared)
    vertex_sigmas_for_signal_only = {
        'xy': getattr(config, 'vertex_sigma_xy', 0.0),
        'z': getattr(config, 'vertex_sigma_z', 0.0),
        't': getattr(config, 'vertex_sigma_t', 0.0)  # Pass as ns
    }
    
    final_merged_path = output_dir / "merged_events.hepmc3"
    logger.info(f"Calling merge function for signal {signal_file_path} and pileup {pileup_path}")
    logger.info("Note: Pileup events already have vertex smearing applied, only signal events will be smeared during merge")
    
    merge_external_signal_with_pileup(
        signal_path=signal_file_path,
        pileup_path=pileup_path,
        num_events=config.events,
        output_path=final_merged_path,
        vertex_sigmas_mm_ns=vertex_sigmas_for_signal_only,
        config=None,
        smear_pileup=False,  # Pileup is already smeared during generation
        logger=logger
    )
    
    logger.info(f"Signal-pileup merge completed. Output: {final_merged_path}")
    return final_merged_path

def run_merge_signal_mode(output_dir, config, logger):
    """Run signal merging mode: generate pileup and merge with existing signal file.
    
    Args:
        output_dir: Path to output directory
        config: Configuration object
        logger: Logger instance
        
    Returns:
        Path: Path to final merged output file
    """
    # Check for existing signal file - try both naming conventions
    signal_file_candidates = [
        output_dir / "events.hepmc3",      # From MadGraph with splitting
        output_dir / "events.hepmc.gz"    # From MadGraph without splitting
    ]
    
    signal_file_path = None
    for candidate in signal_file_candidates:
        if candidate.exists():
            signal_file_path = candidate
            break
    
    if signal_file_path is None:
        available_files = list(output_dir.glob("*.hepmc*"))
        raise FileNotFoundError(
            f"No signal file found. Looked for: {[str(c) for c in signal_file_candidates]}. "
            f"Available files in {output_dir}: {[f.name for f in available_files]}"
        )
    
    logger.info(f"Signal file merging mode with: {signal_file_path}")
    
    # Generate pileup events with smearing
    pileup_path = generate_pileup_events(output_dir, config, logger)
    
    # Merge signal and pileup
    return merge_signal_and_pileup(signal_file_path, pileup_path, output_dir, config, logger)

def run_standard_pythia_mode(output_dir, config, logger):
    """Run standard Pythia8 generation mode (signal+pileup or pileup-only).
    
    Args:
        output_dir: Path to output directory
        config: Configuration object
        logger: Logger instance
        
    Returns:
        Path: Path to generated output file
    """
    logger.info("Standard Pythia8 generation mode (no signal file merging).")
    
    # Create sequencer for Pythia8
    s = Sequencer(numThreads=1, events=config.events)
    s.config.logLevel = acts.logging.DEBUG
    seed = config.seed or int(time.time())
    logger.info(f"Using random seed: {seed}")
    rnd = acts.examples.RandomNumbers(seed=seed)
    
    # Create vertex generator
    vtxGen = create_vertex_generator(config, logger)
    
    # Parse Pythia8 settings
    pythia_settings = parse_pythia_settings(config, logger)
    
    logger.info(f"Generating {config.events} events with {config.pileup} pileup each")
    logger.info(f"Final Pythia8 settings: {pythia_settings}")
    
    # Determine output path based on whether we have hard process
    if pythia_settings is not None:
        output_path = output_dir / "merged_events.hepmc3"  # Signal+pileup final product
    else:
        output_path = output_dir / "events_pileup.hepmc3"  # Pileup-only, needs later merging

    try:
        logger.debug("Creating Pythia8 generator...")
        nhardProcess = 1 if pythia_settings is not None else 0
        generator = addPythia8(
            s,
            npileup=config.pileup,
            nhard=nhardProcess,
            hardProcess=pythia_settings,
            outputDirCsv=None,
            outputDirRoot=None,
            outputEvent="events",
            rnd=rnd,
            logLevel=acts.logging.DEBUG,
            vtxGen=vtxGen,
        )
        logger.debug("Pythia8 generator created successfully")

        s.addWriter(
            HepMC3AsciiWriter(
                acts.logging.VERBOSE,
                inputEvent="events",
                outputPath=output_path,
            )
        )

        logger.debug(f"Writing HepMC3 events to {output_path}")
        logger.debug("About to start sequencer run...")
        
        s.run()
        logger.debug("Sequencer run completed")
            
    except Exception as e:
        logger.error(f"Error during Pythia8 generation: {str(e)}")
        logger.error(traceback.format_exc())
        if hasattr(s, 'config') and hasattr(s, 'currentEvent'):
            logger.error(f"Sequencer state at crash:")
            logger.error(f"  Number of events: {s.config.events}")
            logger.error(f"  Current event: {s.currentEvent}")
        raise
    
    # Verify output files exist
    if not output_path.exists():
        raise FileNotFoundError("Pythia8 failed to generate output files")
    
    return output_path

def run_pythia_stage(output_dir, config, logger=None):
    """Run Pythia8 stage to generate HepMC3 files.
    
    Supports two modes:
    1. Standard mode: Generate events using Pythia8 (signal+pileup or pileup-only)
    2. Merge mode: Generate pileup and merge with existing signal file from MadGraph
    
    Args:
        output_dir: Path to output directory
        config: Configuration object with stage parameters
        logger: Optional logger instance
        
    Returns:
        Path: Path to final output file
    """
    logger = logger or setup_logging("Pythia8Stage")
    logger.info("Initializing Pythia8 generation...")
    
    # Check if we're doing signal file merging
    merge_signal = getattr(config, 'merge_signal', False)
    
    if merge_signal:
        return run_merge_signal_mode(output_dir, config, logger)
    else:
        return run_standard_pythia_mode(output_dir, config, logger)

def main():
    try:
        # Parse arguments and load config
        args = parse_args()
        config = load_config(args)
        logger = setup_logging()
        
        # Create output directory structure
        output_dir = Path(args.output)
        if args.output_subdir:
            output_dir = output_dir / args.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize timing recorder
        timer = TimingRecorder(output_dir)
        
        # Run Pythia8 generation
        with timer.record("Pythia8 Generation"):
            output_path = run_pythia_stage(
                output_dir, config, logger
            )
        
        # Write timing report
        timer.write_report()
        
        logger.info("Pythia8 generation completed successfully")
        logger.info(f"Output file:")
        logger.info(f"  {output_path}")
        
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()