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

from utils.app_logging import setup_logging, TimingRecorder
from utils.config import create_base_parser, load_config

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
    
    # Add vertex generator using config values
    logger.info(f"Configuring vertex smearing with sigma_xy={config.vertex_sigma_xy} mm, sigma_z={config.vertex_sigma_z} mm, sigma_t={config.vertex_sigma_t} ns")
    vtxGen = None
    if any(sigma is not None and sigma != 0 for sigma in [
        config.vertex_sigma_xy,
        config.vertex_sigma_z,
        config.vertex_sigma_t
    ]):
        vtxGen = acts.examples.GaussianVertexGenerator(
            stddev=acts.Vector4(
                config.vertex_sigma_xy * u.mm,
                config.vertex_sigma_xy * u.mm,
                config.vertex_sigma_z * u.mm,
                config.vertex_sigma_t * u.ns
            ),
            mean=acts.Vector4(0, 0, 0, 0),
        )

    
    # Initialize settings list from config
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

    if not pythia_settings:
        pythia_settings = None
    
    logger.info(f"Generating {config.events} events with {config.pileup} pileup each")
    logger.info(f"Final Pythia8 settings: {pythia_settings}")
    
    if pythia_settings is not None:
        output_path = output_dir / "merged_events.hepmc3" # This is the final product
    else:
        output_path = output_dir / "events_pileup.hepmc3" # This is pileup only, and needs to be merged with signal events

    try:
        logger.debug("Creating Pythia8 generator...")
        generator = addPythia8(
            s,
            npileup=config.pileup,
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
    if not output_path.exists():
        raise FileNotFoundError("Pythia8 failed to generate output files")
    
    return output_path

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