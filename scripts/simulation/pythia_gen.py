import time
from pathlib import Path
import acts
import acts.examples
from acts.examples import Sequencer
from acts.examples.simulation import addPythia8
import traceback

from utils.logging import setup_logging, TimingRecorder
from utils.config import create_base_parser, load_config

u = acts.UnitConstants

def parse_args():
    """Parse command line arguments"""
    parser = create_base_parser("Pythia8 event generation for ACTS")
    parser.add_argument(
        "--pileup",
        help="Number of pile-up events",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--hard-process",
        help="Pythia8 hard process",
        type=str,
        default="HardQCD:all",
    )
    parser.add_argument(
        "--pythia-settings",
        help="Additional Pythia8 settings (comma-separated)",
        type=str,
        default=None,
    )
    return parser.parse_args()

def run_pythia_stage(output_dir, config, logger=None):
    """Run Pythia8 stage to generate HepMC3 files"""
    logger = logger or setup_logging("Pythia8Stage")
    
    logger.info("Initializing Pythia8 generation...")
    
    # Create sequencer for Pythia8
    s = Sequencer(numThreads=1, events=config.events)
    s.config.logLevel = acts.logging.DEBUG
    seed = config.seed or int(time.time())
    logger.info(f"Using random seed: {seed}")
    rnd = acts.examples.RandomNumbers(seed=seed)
    
    # Use simple base names for files
    hard_scatter_path = output_dir / "events.hepmc3"
    pileup_path = output_dir / "events_pileup.hepmc3"
    
    # Initialize settings list from config
    pythia_settings = []
    
    # Add all settings from config if present
    if hasattr(config, 'pythia_settings') and config.pythia_settings is not None:
        if isinstance(config.pythia_settings, str):
            logger.debug("Processing command-line pythia settings")
            pythia_settings.extend([s.strip() for s in config.pythia_settings.split(',')])
        elif isinstance(config.pythia_settings, list):
            logger.debug("Processing YAML list pythia settings")
            pythia_settings.extend(config.pythia_settings)
        else:
            raise ValueError("pythia_settings must be either a comma-separated string or a list")
    
    # Always append the hard process at the end
    pythia_settings.append(f"{config.hard_process}=on")
    
    logger.info(f"Generating {config.events} events with {config.pileup} pileup each")
    logger.info(f"Final Pythia8 settings: {pythia_settings}")
    
    try:
        logger.debug("Creating Pythia8 generator...")
        generator = addPythia8(
            s,
            npileup=config.pileup,
            hardProcess=pythia_settings,
            outputHepMC=hard_scatter_path,
            outputHepMCPileup=pileup_path,
            rnd=rnd,
            logLevel=acts.logging.DEBUG
        )
        logger.debug("Pythia8 generator created successfully")
        
        logger.debug("About to start sequencer run...")
        try:
            s.run()
            logger.debug("Sequencer run completed")
        except Exception as e:
            logger.error(f"Error during sequencer run: {str(e)}")
            logger.error(traceback.format_exc())
            logger.error(f"Sequencer state at crash:")
            logger.error(f"  Number of events: {s.config.events}")
            logger.error(f"  Current event: {s.currentEvent}")
            raise
            
    except Exception as e:
        logger.error(f"Error during Pythia8 generation: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    
    # Verify output files exist
    if not hard_scatter_path.exists() or not pileup_path.exists():
        raise FileNotFoundError("Pythia8 failed to generate output files")
    
    return hard_scatter_path, pileup_path

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
            hard_scatter_path, pileup_path = run_pythia_stage(
                output_dir, config, logger
            )
        
        # Write timing report
        timer.write_report()
        
        logger.info("Pythia8 generation completed successfully")
        logger.info(f"Output files:")
        logger.info(f"  Hard scatter: {hard_scatter_path}")
        logger.info(f"  Pileup: {pileup_path}")
        
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()