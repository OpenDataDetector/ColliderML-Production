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
    return parser.parse_args()

def run_pythia_stage(output_dir, config, logger=None):
    """Run Pythia8 stage to generate HepMC3 files"""
    logger = logger or setup_logging("Pythia8Stage")
    
    # Create sequencer for Pythia8
    s = Sequencer(numThreads=1, events=config.events)
    s.config.logLevel = acts.logging.VERBOSE
    seed = config.seed or int(time.time())
    rnd = acts.examples.RandomNumbers(seed=seed)
    
    # Use simple base names for files
    hard_scatter_path = output_dir / "events.hepmc3"
    pileup_path = output_dir / "events_pileup.hepmc3"
    
    logger.info(f"Generating {config.events} events with {config.pileup} pileup each")
    addPythia8(
        s,
        npileup=config.pileup,
        hardProcess=[f"{config.hard_process}=on"],
        outputHepMC=hard_scatter_path,
        outputHepMCPileup=pileup_path,
        rnd=rnd,
        logLevel=acts.logging.VERBOSE,
    )
    
    s.run()
    
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