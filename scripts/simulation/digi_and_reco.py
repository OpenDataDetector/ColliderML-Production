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
from utils.app_logging import setup_logging, TimingRecorder
from utils.config import create_base_parser, load_config
from contextlib import contextmanager
import math
from acts.examples.podio import PodioReader
from acts.examples.edm4hep import EDM4hepSimInputConverter

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
    )
    parser.add_argument(
        "--output-csv",
        help="Write CSV output files",
        action="store_true",
    )
    parser.add_argument(
        "--digi",
        help="Run digitization",
        action="store_true",
    )
    parser.add_argument(
        "--reco",
        help="Run reconstruction",
        action="store_true",
    )
    
    parser.add_argument(
        "--vertexing",
        help="Run vertexing",
        action="store_true",
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
    
    # Set performance output directory based on flag
    perf_output = output_dir if config.performance_metrics else None
    
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

    oddSeedingSel = geoDir / "config/odd-seeding-config.json"
    oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)
    
    # Get detector
    detector = getOpenDataDetector(
        odd_dir=geoDir,
        materialDecorator=oddMaterialDeco
    )
    trackingGeometry = detector.trackingGeometry()
    field = detector.field
    
    # Configure EDM4hep reader and converter
    # Step 1: PodioReader to read the EDM4hep file
    podioReader = PodioReader(
        level=acts.logging.DEBUG,
        inputPath=str(input_path),
        outputFrame="events",
        category="events",
    )
    s.addReader(podioReader)
    
    # Step 2: EDM4hepSimInputConverter algorithm to convert EDM4hep data to ACTS format
    edm4hepConverter = EDM4hepSimInputConverter(
        level=acts.logging.DEBUG,
        inputFrame="events",
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
        outputSimVertices="simvertices",
        dd4hepDetector=detector,
        trackingGeometry=trackingGeometry,
    )
    s.addAlgorithm(edm4hepConverter)
    s.addWhiteboardAlias("particles", "particles_input")
    
    # Add digitization if enabled
    if config.digi:
        logger.info("Adding digitization")
        addDigitization(
            s,
            trackingGeometry,
            field,
            digiConfigFile=oddDigiConfig,
            outputDirRoot=perf_output if config.output_root else None,
            outputDirCsv=None,
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
            initialSigmas=[
                1 * u.mm,
                1 * u.mm,
                1 * u.degree,
                1 * u.degree,
                0.1 * u.e / u.GeV,
                1 * u.ns,
            ],
            initialSigmaPtRel=0.1,
            initialVarInflation=[1.0] * 6,
            geoSelectionConfigFile=oddSeedingSel,
            outputDirRoot=perf_output if config.output_root else None
        )
        
        # Add CKF tracking
        addCKFTracks(
            s,
            trackingGeometry,
            field,
            trackSelectorConfig=TrackSelectorConfig(
                pt=(1.0 * u.GeV, None),
                absEta=(None, 3.0),
                loc0=(-4.0 * u.mm, 4.0 * u.mm),
                nMeasurementsMin=7,
                maxHoles=2,
                maxOutliers=2,
            ),
            ckfConfig=CkfConfig(
                chi2CutOffMeasurement=15.0,
                chi2CutOffOutlier=25.0,
                numMeasurementsCutOff=10,
                seedDeduplication=True,
                stayOnSeed=True,
                pixelVolumes=[16, 17, 18],
                stripVolumes=[23, 24, 25],
                maxPixelHoles=1,
                maxStripHoles=2,
                constrainToVolumes=[
                    2,  # beam pipe
                    32,
                    4,  # beam pip gap
                    16,
                    17,
                    18,  # pixel
                    20,  # PST
                    23,
                    24,
                    25,  # short strip
                    26,
                    8,  # long strip gap
                    28,
                    29,
                    30,  # long strip
                ],
            ),
            outputDirRoot=perf_output if config.output_root else None,
            outputDirCsv=perf_output if config.output_csv else None,
            writeCovMat=config.performance_metrics,
            writeTrackStates=config.performance_metrics,
            writeTrackSummary=config.performance_metrics,
            writePerformance=config.performance_metrics,
        )
        
        # Add ambiguity resolution
        if config.ambi_solver == "ML":
            addAmbiguityResolutionML(
                s,
                config=AmbiguityResolutionMLConfig(
                    maximumSharedHits=3,
                    maximumIterations=1000000,
                    nMeasurementsMin=7,
                ),
                outputDirRoot=perf_output if config.output_root else None,
                outputDirCsv=perf_output if config.output_csv else None,
                onnxModelFile=str(config.ambi_config),
            )
        elif config.ambi_solver == "scoring":
            addScoreBasedAmbiguityResolution(
                s,
                config=ScoreBasedAmbiguityResolutionConfig(
                    minScore=0,
                    maxShared=2,
                    maxSharedTracksPerMeasurement=2
                ),
                outputDirRoot=perf_output if config.output_root else None,
                outputDirCsv=perf_output if config.output_csv else None,
                ambiVolumeFile=config.ambi_config,
            )
        else:
            addAmbiguityResolution(
                s,
                config=AmbiguityResolutionConfig(
                    maximumSharedHits=3,
                    maximumIterations=1000000,
                    nMeasurementsMin=7,
                ),
                outputDirRoot=perf_output if config.output_root else None,
                outputDirCsv=perf_output if config.output_csv else None,
                writeCovMat=config.performance_metrics,
                writeTrackStates=config.performance_metrics,
                writeTrackSummary=config.performance_metrics,
                writePerformance=config.performance_metrics,
            )
        
        # Add vertex fitting
        if config.vertexing:
            addVertexFitting(
                s,
                field,
                vertexFinder=VertexFinder.AMVF,
                outputDirRoot=perf_output if config.output_root else None,
                outputDirCsv=perf_output if config.output_csv else None,
            )
    
    # Add ROOT writers if enabled and performance metrics are on
    if config.output_root and config.performance_metrics:
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

def main():
    logger = setup_logging()
    try:
        # Parse arguments and load config
        args = parse_args()
        config = load_config(args)
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