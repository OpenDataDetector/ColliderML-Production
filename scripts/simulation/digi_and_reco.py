import time
from pathlib import Path
import acts
import acts.examples
import acts.examples.edm4hep
from acts.examples import Sequencer
from acts.examples.odd import getOpenDataDetector, getOpenDataDetectorDirectory
from acts.examples.simulation import addDigitization
from acts.examples.reconstruction import (
    addSeeding,
    addCKFTracks,
    addVertexFitting,
    addAmbiguityResolution,
    VertexFinder
)
import traceback
from utils.logging import setup_logging, TimingRecorder
from utils.config import create_base_parser, load_config

u = acts.UnitConstants

def parse_args():
    """Parse command line arguments"""
    parser = create_base_parser("Digitization and reconstruction for ACTS")
    parser.add_argument(
        "--input-file",
        help="Input SLCIO file (default: {output_dir}/ddsim_output.slcio)",
        type=Path,
        default=None
    )
    parser.add_argument(
        "--digi-config",
        help="Digitization configuration file",
        type=Path,
    )
    parser.add_argument(
        "--material-config",
        help="Material map configuration file",
        type=Path,
    )
    return parser.parse_args()

def setup_acts_reconstruction(input_path, output_dir, config, logger=None):
    """Configure ACTS reconstruction chain"""
    logger = logger or setup_logging("ACTSReco")
    
    # Create sequencer
    s = Sequencer(numThreads=1, events=config.events)
    s.config.logLevel = acts.logging.VERBOSE
    
    # Get detector and field
    geoDir = getOpenDataDetectorDirectory()
    field = acts.ConstantBField(acts.Vector3(0.0, 0.0, 2.0 * u.T))
    
    # Load material map
    oddMaterialMap = (
        config.material_config
        if config.material_config
        else geoDir / "data/odd-material-maps.root"
    )
    oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)
    
    # Get detector
    detector, trackingGeometry, _ = getOpenDataDetector(
        odd_dir=geoDir,
        mdecorator=oddMaterialDeco
    )
    
    # Configure EDM4hep reader
    edm4hepReader = acts.examples.edm4hep.EDM4hepReader(
        level=acts.logging.VERBOSE,
        config=acts.examples.edm4hep.EDM4hepReader.Config(
            inputPath=str(input_path),
            inputSimHits=[
                "PixelBarrelReadout",
                "PixelEndcapReadout",
                "ShortStripBarrelReadout",
                "ShortStripEndcapReadout",
                "LongStripBarrelReadout",
                "LongStripEndcapReadout"
            ],
            outputParticlesGenerator="particles_input",
            outputParticlesSimulation="particles_simulated",
            outputSimHits="simhits",
            dd4hepDetector=detector,
            trackingGeometry=trackingGeometry
        )
    )
    s.addReader(edm4hepReader)
    
    # Add digitization if configured
    if config.digi_config:
        addDigitization(
            s,
            trackingGeometry,
            field,
            digiConfigFile=config.digi_config,
            outputDirRoot=output_dir,
            outputDirCsv=output_dir,
        )
    
    # Add reconstruction steps
    addSeeding(
        s,
        trackingGeometry,
        field,
        seedingConfig=geoDir / "config/odd-seeding-config.json",
    )
    
    addCKFTracks(
        s,
        trackingGeometry,
        field,
    )
    
    addAmbiguityResolution(s)
    
    addVertexFitting(
        s,
        field,
        vertexFinder=VertexFinder.ADAPTIVE,
        outputDirRoot=output_dir,
    )
    
    return s

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
        input_path = args.input_file or output_dir / "edm4hep.root"
        
        # Initialize timing recorder
        timer = TimingRecorder(output_dir)
        
        # Setup and run reconstruction
        with timer.record("ACTS Reconstruction"):
            s = setup_acts_reconstruction(input_path, output_dir, config, logger)
            s.run()
        
        # Write timing report
        timer.write_report()
        
        logger.info("ACTS reconstruction completed successfully")
        
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()