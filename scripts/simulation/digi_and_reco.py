import time
from pathlib import Path
import acts
import acts.examples
import acts.examples.edm4hep
from acts.examples import Sequencer
from acts.examples.odd import getOpenDataDetector, getOpenDataDetectorDirectory
from acts.examples.simulation import (
    addDigitization,
    addSimParticleSelection,
    addDigiParticleSelection,
    ParticleSelectorConfig,
)
from acts.examples.reconstruction import (
    addSeeding,
    SeedingAlgorithm,
    SeedFinderConfigArg,
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
        default=None
    )
    parser.add_argument(
        "--material-config",
        help="Material map configuration file",
        type=Path,
        default=None
    )
    parser.add_argument(
        "--ambi-solver",
        help="Ambiguity solver to use",
        choices=["greedy", "scoring", "ML"],
        default=None
    )
    parser.add_argument(
        "--ambi-config",
        help="Score Based ambiguity resolution config",
        type=Path,
        default=None
    )
    parser.add_argument(
        "--output-root",
        help="Write ROOT output files",
        action="store_true",
        default=None
    )
    parser.add_argument(
        "--output-csv",
        help="Write CSV output files",
        action="store_true",
        default=None
    )
    parser.add_argument(
        "--digi",
        help="Run digitization",
        action="store_true",
        default=None
    )
    parser.add_argument(
        "--reco",
        help="Run reconstruction",
        action="store_true",
        default=None
    )
    
    parser.add_argument(
        "--vertexing",
        help="Run vertexing",
        action="store_true",
        default=None
    )

    parser.add_argument(
        "--threads",
        help="Number of threads to use",
        type=int,
        default=None
    )

    return parser.parse_args()

def setup_acts_reconstruction(input_path, output_dir, config, rnd, logger=None):
    """Configure ACTS reconstruction chain"""
    logger = logger or setup_logging("ACTSReco")
    
    # Create sequencer
    s = Sequencer(
        numThreads=config.threads if config.threads is not None else 1,
        events=config.events,
        logLevel=acts.logging.DEBUG,
        trackFpes=False,
    )
    
    # Get detector and field
    geoDir = getOpenDataDetectorDirectory()
    
    # Set performance output directory based on flag
    perf_output = output_dir if getattr(config, 'performance_metrics', False) else None
    
    # Load material map
    material_config = getattr(config, 'material_config', None)
    oddMaterialMap = (
        geoDir / f"data/{material_config}"
        if material_config
        else geoDir / "data/odd-material-maps.root"
    )

    digi_config = getattr(config, 'digi_config', None)
    if digi_config:
        dc_path = Path(digi_config)
        oddDigiConfig = dc_path if dc_path.is_file() else (geoDir / f"config/{digi_config}")
    else:
        oddDigiConfig = geoDir / "config/odd-digi-smearing-config.json"

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
        sortSimHitsInTime=False,
        particleRMax=1080 * u.mm,
        particleZ=(-3030 * u.mm, 3030 * u.mm),
        particlePtMin=150 * u.MeV,
    )
    s.addAlgorithm(edm4hepConverter)
    s.addWhiteboardAlias("particles", edm4hepConverter.config.outputParticlesSimulation)
    
    # Add sim particle selection (filters particles from simulation)
    addSimParticleSelection(
        s,
        ParticleSelectorConfig(),
    )
    
    # Add digitization if enabled
    digi_enabled = getattr(config, 'digi', True)  # Default True
    if digi_enabled:
        logger.info("Adding digitization")
        addDigitization(
            s,
            trackingGeometry,
            field,
            digiConfigFile=oddDigiConfig,
            outputDirRoot=perf_output if getattr(config, 'output_root', True) else None,
            outputDirCsv=None,
            rnd=rnd,
            logLevel=acts.logging.DEBUG,
        )

        def make_geoid(vol=None, lay=None):
            geoid = acts.GeometryIdentifier()
            if vol is not None:
                geoid.volume = vol
            if lay is not None:
                geoid.layer = lay
            return geoid

        measurementCounter = acts.examples.ParticleSelector.MeasurementCounter()
        # At least 3 hits in the pixels
        measurementCounter.addCounter(
            [
                make_geoid(16),
                make_geoid(17),
                make_geoid(18),
            ],
            3,
        )
        
        # Add digi particle selection (filters particles with sufficient measurements)
        addDigiParticleSelection(
            s,
            ParticleSelectorConfig(
                rho=(0.0, 24 * u.mm),
                absZ=(0.0, 1.0 * u.m),
                eta=(-3.0, 3.0),
                pt=(0.999 * u.GeV, None),
                measurements=(6, None),
                removeNeutral=True,
                removeSecondaries=False,
                nMeasurementsGroupMin=measurementCounter,
            ),
        )
    
    # Add reconstruction components if enabled
    reco_enabled = getattr(config, 'reco', False)  # Default False
    if reco_enabled:
        logger.info("Adding reconstruction chain")
        # Add seeding
        addSeeding(
            s,
            trackingGeometry,
            field,
            seedingAlgorithm=SeedingAlgorithm.Default,
            particleHypothesis=acts.ParticleHypothesis.pion,
            seedFinderConfigArg=SeedFinderConfigArg(
                r=(33 * u.mm, 200 * u.mm),
                # kills efficiency at |eta|~2
                deltaR=(1 * u.mm, 300 * u.mm),
                collisionRegion=(-250 * u.mm, 250 * u.mm),
                z=(-2000 * u.mm, 2000 * u.mm),
                maxSeedsPerSpM=3,
                sigmaScattering=5,
                radLengthPerSeed=0.1,
                minPt=0.5 * u.GeV,
                impactMax=3 * u.mm,
            ),
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
            outputDirRoot=perf_output if getattr(config, 'output_root', True) else None
        )
        
        # Add CKF tracking
        addCKFTracks(
            s,
            trackingGeometry,
            field,
            trackSelectorConfig=TrackSelectorConfig(
                pt=(0.7 * u.GeV, None),
                absEta=(None, 3.5),
                nMeasurementsMin=6,
                maxHolesAndOutliers=3,
            ),
            ckfConfig=CkfConfig(
                chi2CutOffMeasurement=15.0,
                chi2CutOffOutlier=25.0,
                numMeasurementsCutOff=1,
                seedDeduplication=True,
                stayOnSeed=True,
            ),
            twoWay=True,
            outputDirRoot=perf_output if getattr(config, 'output_root', True) else None,
            outputDirCsv=perf_output if getattr(config, 'output_csv', False) else None,
            writeCovMat=getattr(config, 'performance_metrics', False),
            writeTrackStates=getattr(config, 'performance_metrics', False),
            writeTrackSummary=getattr(config, 'performance_metrics', False),
            writePerformance=getattr(config, 'performance_metrics', False),
        )
        
        # Add ambiguity resolution
        ambi_solver = getattr(config, 'ambi_solver', 'greedy')  # Default greedy
        ambi_config = getattr(config, 'ambi_config', None)
        
        if ambi_solver == "ML":
            addAmbiguityResolutionML(
                s,
                config=AmbiguityResolutionMLConfig(
                    maximumSharedHits=3,
                    maximumIterations=1000000,
                    nMeasurementsMin=7,
                ),
                outputDirRoot=perf_output if getattr(config, 'output_root', True) else None,
                outputDirCsv=perf_output if getattr(config, 'output_csv', False) else None,
                onnxModelFile=str(ambi_config),
            )
        elif ambi_solver == "scoring":
            addScoreBasedAmbiguityResolution(
                s,
                config=ScoreBasedAmbiguityResolutionConfig(
                    minScore=0,
                    maxShared=2,
                    maxSharedTracksPerMeasurement=2
                ),
                outputDirRoot=perf_output if getattr(config, 'output_root', True) else None,
                outputDirCsv=perf_output if getattr(config, 'output_csv', False) else None,
                ambiVolumeFile=ambi_config,
            )
        else:
            addAmbiguityResolution(
                s,
                config=AmbiguityResolutionConfig(
                    maximumSharedHits=3,
                    maximumIterations=1000000,
                    nMeasurementsMin=7,
                ),
                outputDirRoot=perf_output if getattr(config, 'output_root', True) else None,
                outputDirCsv=perf_output if getattr(config, 'output_csv', False) else None,
                writeCovMat=getattr(config, 'performance_metrics', False),
                writeTrackStates=getattr(config, 'performance_metrics', False),
                writeTrackSummary=getattr(config, 'performance_metrics', False),
                writePerformance=getattr(config, 'performance_metrics', False),
            )
        
        # Add vertex fitting
        vertexing_enabled = getattr(config, 'vertexing', False)  # Default False
        if vertexing_enabled:
            addVertexFitting(
                s,
                field,
                vertexFinder=VertexFinder.AMVF,
                outputDirRoot=perf_output if getattr(config, 'output_root', True) else None,
                outputDirCsv=perf_output if getattr(config, 'output_csv', False) else None,
            )
    
    # Add ROOT writers if enabled and performance metrics are on
    output_root = getattr(config, 'output_root', True)
    performance_metrics = getattr(config, 'performance_metrics', False)
    if output_root and performance_metrics:
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
    s.addWriter(acts.examples.RootParticleWriter(
        config=acts.examples.RootParticleWriter.Config(
            filePath=str(output_dir / "particles.root"),
            inputParticles="particles"
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