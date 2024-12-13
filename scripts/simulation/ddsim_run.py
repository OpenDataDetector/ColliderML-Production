import time
from pathlib import Path
import acts
from DDSim.DD4hepSimulation import DD4hepSimulation
import traceback
from acts.examples.odd import getOpenDataDetectorDirectory
from utils.logging import setup_logging, TimingRecorder
from utils.config import create_base_parser, load_config

def parse_args():
    """Parse command line arguments"""
    parser = create_base_parser("DD4hep simulation for ACTS")
    parser.add_argument(
        "--input-file",
        help="Input HepMC3 file (default: {output_dir}/merged_events.hepmc3)",
        type=Path,
        default=None
    )
    return parser.parse_args()

def run_ddsim(input_path, output_path, config, logger=None):
    """Run DD4hep simulation"""
    logger = logger or setup_logging("DD4hepStage")
    
    # Get detector XML
    odd_dir = getOpenDataDetectorDirectory()
    odd_xml = odd_dir / "xml" / "OpenDataDetector.xml"
    
    # Configure DD4hep
    ddsim = DD4hepSimulation()
    if isinstance(ddsim.compactFile, list):
        ddsim.compactFile = [str(odd_xml)]
    else:
        ddsim.compactFile = str(odd_xml)
    
    ddsim.inputFiles = [str(input_path)]
    ddsim.outputFile = str(output_path)
    ddsim.numberOfEvents = config.events
    ddsim.random.seed = config.seed or int(time.time())
    
    logger.info(f"Running DD4hep simulation with {config.events} events")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    
    ddsim.run()

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
        
        # Set default input path if not specified
        input_path = args.input_file or output_dir / "merged_events.hepmc3"
        output_path = output_dir / "edm4hep.root"
        
        # Initialize timing recorder
        timer = TimingRecorder(output_dir)
        
        # Run DD4hep simulation
        with timer.record("DD4hep Simulation"):
            run_ddsim(input_path, output_path, config, logger)
        
        # Write timing report
        timer.write_report()
        
        logger.info("DD4hep simulation completed successfully")
        logger.info(f"Output file: {output_path}")
        
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()