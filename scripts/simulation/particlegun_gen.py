#!/usr/bin/env python3
"""Generate particle gun events with log-uniform energy distribution using ACTS"""

import time
from pathlib import Path
import traceback

import acts
from acts import UnitConstants as u
from acts.examples import Sequencer
from acts.examples.hepmc3 import HepMC3Writer

from utils.app_logging import setup_logging, TimingRecorder
from utils.config import create_base_parser, load_config


def pdg_name_to_code(particle_name):
    """Convert particle name to PDG code
    
    Args:
        particle_name: Particle name (e.g., 'gamma', 'e-') or PDG code as string/int
        
    Returns:
        int: PDG code
    """
    pdg_map = {
        'e-': 11, 'e+': -11, 'electron': 11, 'positron': -11,
        'mu-': 13, 'mu+': -13, 'muon': 13, 'muon-': 13, 'muon+': -13,
        'pi+': 211, 'pi-': -211, 'pi0': 111, 'pion+': 211, 'pion-': -211, 'pion0': 111,
        'gamma': 22, 'photon': 22,
        'proton': 2212, 'antiproton': -2212, 'p': 2212, 'pbar': -2212,
        'neutron': 2112, 'antineutron': -2112, 'n': 2112, 'nbar': -2112,
        'K+': 321, 'K-': -321, 'K0': 311, 'kaon+': 321, 'kaon-': -321, 'kaon0': 311,
        'tau-': 15, 'tau+': -15, 'tau': 15,
    }
    
    # Try to parse as integer first
    try:
        return int(particle_name)
    except (ValueError, TypeError):
        pass
    
    # Look up by name
    if isinstance(particle_name, str):
        name_lower = particle_name.lower()
        if name_lower in pdg_map:
            return pdg_map[name_lower]
        # Try case-sensitive for things like K+
        if particle_name in pdg_map:
            return pdg_map[particle_name]
    
    raise ValueError(
        f"Unknown particle name: {particle_name}. "
        f"Use PDG code (int) or name like 'gamma', 'e-', 'mu-', 'pi+', etc."
    )


def parse_args():
    """Parse command line arguments"""
    parser = create_base_parser("Particle gun event generation with log-uniform energy distribution")
    
    parser.add_argument(
        "--particle",
        help="Particle name (e.g., 'gamma', 'e-', 'mu-') or PDG code",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--energy-min",
        help="Minimum energy [GeV]",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--energy-max",
        help="Maximum energy [GeV]",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--log-uniform",
        help="Use log-uniform energy distribution",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--eta-min",
        help="Minimum pseudorapidity",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--eta-max",
        help="Maximum pseudorapidity",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--phi-min",
        help="Minimum azimuthal angle [rad]",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--phi-max",
        help="Maximum azimuthal angle [rad]",
        type=float,
        default=None,
    )
    
    return parser.parse_args()


def generate_particle_gun_events(output_dir, config, logger):
    """Generate single particle events using ACTS ParametricParticleGenerator
    
    Args:
        output_dir: Output directory path
        config: Configuration object with particle gun parameters
        logger: Logger instance
        
    Returns:
        Path: Path to generated HepMC3 file
    """
    # Get particle gun configuration
    particle_name = getattr(config, 'particle', 'mu-')  # Default: muon
    particle_pdg = pdg_name_to_code(particle_name)
    
    energy_min = getattr(config, 'energy_min', 1.0) * u.GeV
    energy_max = getattr(config, 'energy_max', 100.0) * u.GeV
    log_uniform = getattr(config, 'log_uniform', False)
    
    # Angular range
    eta_min = getattr(config, 'eta_min', -2.5)
    eta_max = getattr(config, 'eta_max', 2.5)
    phi_min = getattr(config, 'phi_min', 0.0)
    phi_max = getattr(config, 'phi_max', 2.0 * 3.14159265359)
    
    n_events = config.events
    seed = config.seed or int(time.time())
    
    logger.info(f"Generating {n_events} particle gun events")
    logger.info(f"  Particle: {particle_name} (PDG {particle_pdg})")
    logger.info(f"  Energy: [{energy_min/u.GeV}, {energy_max/u.GeV}] GeV")
    logger.info(f"  Log-uniform: {log_uniform}")
    logger.info(f"  Eta range: [{eta_min}, {eta_max}]")
    logger.info(f"  Phi range: [{phi_min}, {phi_max}]")
    logger.info(f"  Random seed: {seed}")
    
    # Create ACTS sequencer
    s = Sequencer(numThreads=1, events=n_events, logLevel=acts.logging.INFO)
    
    # Random number generator
    rnd = acts.examples.RandomNumbers(seed=seed)
    
    # Event generator with particle gun
    evGen = acts.examples.EventGenerator(
        level=acts.logging.INFO,
        generators=[
            acts.examples.EventGenerator.Generator(
                multiplicity=acts.examples.FixedMultiplicityGenerator(n=1),
                vertex=acts.examples.GaussianVertexGenerator(
                    mean=acts.Vector4(0, 0, 0, 0),
                    stddev=acts.Vector4(0, 0, 0, 0),
                ),
                particles=acts.examples.ParametricParticleGenerator(
                    p=(energy_min, energy_max),
                    pLogUniform=log_uniform,
                    eta=(eta_min, eta_max),
                    phi=(phi_min, phi_max),
                    etaUniform=True,
                    numParticles=1,
                    pdg=particle_pdg,
                ),
            )
        ],
        outputEvent="particle_gun_event",
        randomNumbers=rnd,
    )
    s.addReader(evGen)
    
    # Output path
    output_path = output_dir / "events.hepmc3"
    
    # Write to HepMC3
    s.addWriter(
        HepMC3Writer(
            acts.logging.INFO,
            inputEvent=evGen.config.outputEvent,
            outputPath=output_path,
            perEvent=False,
        )
    )
    
    logger.info(f"Writing events to {output_path}")
    s.run()
    logger.info(f"Particle gun generation completed")
    
    return output_path


def main():
    timer = None
    logger = setup_logging("ParticleGun")
    try:
        # Parse arguments and load config
        args = parse_args()
        config = load_config(args)
        
        # Create output directory
        output_dir = Path(args.output)
        if args.output_subdir:
            output_dir = output_dir / args.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=== Particle Gun Generation ===")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Events: {config.events}")
        
        # Initialize timing
        timer = TimingRecorder(output_dir)
        
        # Generate events
        with timer.record("Particle Gun Generation"):
            final_output = generate_particle_gun_events(output_dir, config, logger)
        
        logger.info("=== Generation Completed Successfully ===")
        logger.info(f"Output file: {final_output}")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    finally:
        if timer:
            try:
                timer.write_report()
            except Exception as report_e:
                logger.error(f"Error writing timing report: {str(report_e)}")


if __name__ == "__main__":
    main()
