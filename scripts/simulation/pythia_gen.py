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
    
    if "SUSY:" in config.hard_process:
        logger.info("Detected SUSY process, adding initialization settings...")
        base_settings = [
            # 1. Debug output (corrected)
            "Init:showProcesses = on",
            "Init:showChangedSettings = on",
            "Init:showAllSettings = on",
            "Init:showMultipartonInteractions = on",
            "Next:numberCount = 0",
            "Next:numberShowLHA = 1",
            "Next:numberShowInfo = 1",
            "Next:numberShowProcess = 1",
            "Print:quiet = off",
            
            # 2. SUSY setup (corrected)
            "SUSY:all = on",  # Enable all SUSY processes first
            "SLHA:file = none",  # Don't use an SLHA file
            "SUSY:model = 2",  # Use internal MSSM model
            
            # 3. Beam settings 
            "Beams:idA = 2212",
            "Beams:idB = 2212",
            "Beams:eCM = 13000.",
            
            # 4. Particle masses and properties
            "1000022:m0 = 100.0",  # LSP mass
            "1000022:tau0 = 0.0",  # LSP stable
            "1000023:m0 = 200.0",  # NLSP mass
            "1000023:tau0 = 0.0",  # NLSP stable
            "1000021:m0 = 800.0",  # Gluino mass
            "1000001:m0 = 1000.0", # Squark mass
            "2000001:m0 = 1000.0", # Squark mass
            
            # 5. Process selection
            "SUSY:qqbar2chi0chi0 = on",  # Turn on specific process after model setup
            
            # 6. General settings
            "PartonLevel:ISR = on",
            "PartonLevel:FSR = on",
            "PartonLevel:MPI = on",
            "HadronLevel:all = on",
            
            # 7. Additional SUSY settings
            "SUSY:qq2chi0chi0 = on",  # Additional production channels
            "SUSY:gg2chi0chi0 = on",
        ]
        pythia_settings = base_settings
    else:
        pythia_settings = [f"{config.hard_process}=on"]
    
    # Add any additional user settings, but avoid duplicates
    if hasattr(config, 'pythia_settings') and config.pythia_settings:
        if isinstance(config.pythia_settings, str):
            logger.debug("Processing command-line pythia settings")
            additional_settings = [s.strip() for s in config.pythia_settings.split(',')]
        elif isinstance(config.pythia_settings, list):
            logger.debug("Processing YAML list pythia settings")
            additional_settings = config.pythia_settings
        else:
            raise ValueError("pythia_settings must be either a comma-separated string or a list")
        
        # Only add settings that aren't already present
        for setting in additional_settings:
            setting_name = setting.split('=')[0].strip()
            if not any(s.startswith(setting_name) for s in pythia_settings):
                pythia_settings.append(setting)
        logger.info(f"Added user settings: {additional_settings}")
    
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
            logLevel=acts.logging.VERBOSE  # Even more detailed logging
        )
        logger.debug("Pythia8 generator created successfully")
        
        logger.debug("About to start sequencer run...")
        try:
            s.run()
            logger.debug("Sequencer run completed")
        except Exception as e:
            logger.error(f"Error during sequencer run: {str(e)}")
            logger.error(traceback.format_exc())
            # Try to get more info about sequencer state
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