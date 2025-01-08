import time
from pathlib import Path
import acts
import acts.examples
import acts.examples.edm4hep
from acts.examples import Sequencer
from acts.examples.odd import getOpenDataDetector, getOpenDataDetectorDirectory
from acts.examples.simulation import addDigitization, addParticleSelection, ParticleSelectorConfig
from acts.examples.reconstruction import (
    addSeeding,
    addCKFTracks,
    addVertexFitting,
    addAmbiguityResolution,
    addAmbiguityResolutionML,
    addScoreBasedAmbiguityResolution,
    VertexFinder,
    TrackSelectorConfig,
    CkfConfig,
    AmbiguityResolutionConfig,
    AmbiguityResolutionMLConfig,
    ScoreBasedAmbiguityResolutionConfig
)
import traceback
from utils.logging import setup_logging, TimingRecorder
from utils.config import create_base_parser, load_config
from contextlib import contextmanager
import math

u = acts.UnitConstants

def parse_args():
    """Parse command line arguments"""
    parser = create_base_parser("Digitization and reconstruction for ACTS")
    parser.add_argument(
        "--input-file",
        help="Input EDM4hep file (default: {output_dir}/edm4hep.root)",
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
    parser.add_argument(
        "--ambi-solver",
        help="Ambiguity solver to use",
        choices=["greedy", "scoring", "ML"],
        default="greedy",
    )
    parser.add_argument(
        "--ambi-config",
        help="Score Based ambiguity resolution config",
        type=Path,
    )
    parser.add_argument(
        "--output-root",
        help="Write ROOT output files",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--digi",
        help="Run digitization",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--reco",
        help="Run reconstruction",
        action="store_true",
        default=True,
    )
    return parser.parse_args()

def setup_acts_reconstruction(input_path, output_dir, config, rnd, logger=None):
    """Configure ACTS reconstruction chain"""
    logger = logger or setup_logging("ACTSReco")
    
    # Create sequencer
    s = Sequencer(numThreads=1, events=config.events)
    s.config.logLevel = acts.logging.DEBUG
    
    # Get detector and field
    geoDir = getOpenDataDetectorDirectory()
    field = acts.ConstantBField(acts.Vector3(0.0, 0.0, 2.0 * u.T))
    
    # Load material map
    oddMaterialMap = (
        geoDir / f"data/{config.material_config}"
        if config.material_config
        else geoDir / "data/odd-material-maps.root"
    )

    oddDigiConfig = (
        geoDir / f"config/{config.digi_config}"
        if config.digi_config
        else geoDir / "config/odd-digi-smearing-config.json"
    )

    oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)
    
    # Get detector
    detector, trackingGeometry, _ = getOpenDataDetector(
        odd_dir=geoDir,
        mdecorator=oddMaterialDeco
    )
    
    # Configure EDM4hep reader
    edm4hepReader = acts.examples.edm4hep.EDM4hepReader(
        level=acts.logging.DEBUG,
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
    
    # Add particle selection
    addParticleSelection(
        s,
        config=ParticleSelectorConfig(
            rho=(0.0, 24 * u.mm),
            absZ=(0.0, 1.0 * u.m),
            eta=(-4.0, 4.0),
            pt=(0.0 * u.GeV, None),
            removeNeutral=False,
        ),
        inputParticles="particles_input",
        outputParticles="particles_selected",
    )
    
    # Add digitization if enabled
    if config.digi:
        logger.info("Adding digitization")
        addDigitization(
            s,
            trackingGeometry,
            field,
            digiConfigFile=oddDigiConfig,
            outputDirRoot=output_dir if config.output_root else None,
            outputDirCsv=output_dir if config.output_csv else None,
            rnd=rnd,
            logLevel=acts.logging.DEBUG,
        )
    
    # Add reconstruction components if enabled
    if config.reco:
        logger.info("Adding reconstruction chain")
        # Add seeding
        addSeeding(
            s,
            trackingGeometry,
            field,
            seedingConfig=geoDir / "config/odd-seeding-config.json",
        )
        
        # Add CKF tracking
        addCKFTracks(
            s,
            trackingGeometry,
            field,
            TrackSelectorConfig(
                pt=(1.0 * u.GeV, None),
                absEta=(None, 3.0),
                loc0=(-4.0 * u.mm, 4.0 * u.mm),
                nMeasurementsMin=7,
                maxHoles=2,
                maxOutliers=2,
            ),
            CkfConfig(
                chi2CutOffMeasurement=15.0,
                chi2CutOffOutlier=25.0,
                numMeasurementsCutOff=10,
                seedDeduplication=True,
                stayOnSeed=True,
                pixelVolumes=[16, 17, 18],
                stripVolumes=[23, 24, 25],
                maxPixelHoles=1,
                maxStripHoles=2,
            ),
            outputDirRoot=output_dir if config.output_root else None,
            writeCovMat=True,
        )
        
        # Add ambiguity resolution
        if config.ambi_solver == "ML":
            addAmbiguityResolutionML(
                s,
                AmbiguityResolutionMLConfig(
                    maximumSharedHits=3,
                    maximumIterations=1000000,
                    nMeasurementsMin=7,
                ),
                outputDirRoot=output_dir if config.output_root else None,
                onnxModelFile=str(config.ambi_config),
            )
        elif config.ambi_solver == "scoring":
            addScoreBasedAmbiguityResolution(
                s,
                ScoreBasedAmbiguityResolutionConfig(
                    minScore=0,
                    maxShared=2,
                    maxSharedTracksPerMeasurement=2,
                    pTMax=1400,
                    pTMin=0.5,
                ),
                outputDirRoot=output_dir if config.output_root else None,
                ambiVolumeFile=config.ambi_config,
            )
        else:
            addAmbiguityResolution(
                s,
                AmbiguityResolutionConfig(
                    maximumSharedHits=3,
                    maximumIterations=1000000,
                    nMeasurementsMin=7,
                ),
                outputDirRoot=output_dir if config.output_root else None,
            )
        
        # Add vertex fitting
        addVertexFitting(
            s,
            field,
            vertexFinder=VertexFinder.ADAPTIVE,
            outputDirRoot=output_dir if config.output_root else None,
        )
    
    # Add ROOT writers if enabled
    if config.output_root:
        add_root_writers(s, output_dir)
    
    return s

def add_root_writers(s, output_dir):
    """Add ROOT output writers to the sequencer"""
    # Write tracking hits
    s.addWriter(acts.examples.RootSimHitWriter(
        config=acts.examples.RootSimHitWriter.Config(
            filePath=str(output_dir / "simhits.root"),
            inputSimHits="simhits"
        ),
        level=acts.logging.INFO
    ))
    
    # Write particles
    s.addWriter(acts.examples.RootParticleFlatWriter(
        config=acts.examples.RootParticleFlatWriter.Config(
            filePath=str(output_dir / "particles.root"),
            inputParticles="particles_selected"
        ),
        level=acts.logging.INFO
    ))

def main():
    try:
        # Parse arguments and load config
        args = parse_args()
        config = load_config(args)
        logger = setup_logging()
        rnd = acts.examples.RandomNumbers(seed=config.seed)
        
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
            s = setup_acts_reconstruction(input_path, output_dir, config, rnd, logger)
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