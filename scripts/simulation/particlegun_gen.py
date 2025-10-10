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


def pdg_name_to_particle(particle_name):
    """Convert particle name to ACTS PdgParticle
    
    Args:
        particle_name: Particle name (e.g., 'gamma', 'e-', 'mu-') or PDG code as string/int
        
    Returns:
        acts.PdgParticle: ACTS particle enum
    """
    # Map common names to ACTS PdgParticle enums
    name_to_pdg = {
        'e-': acts.PdgParticle.eElectron,
        'e+': acts.PdgParticle.ePositron,
        'electron': acts.PdgParticle.eElectron,
        'positron': acts.PdgParticle.ePositron,
        'mu-': acts.PdgParticle.eMuon,
        'mu+': acts.PdgParticle.eAntiMuon,
        'muon': acts.PdgParticle.eMuon,
        'muon-': acts.PdgParticle.eMuon,
        'muon+': acts.PdgParticle.eAntiMuon,
        'gamma': acts.PdgParticle.eGamma,
        'photon': acts.PdgParticle.eGamma,
        'pi+': acts.PdgParticle.ePionPlus,
        'pi-': acts.PdgParticle.ePionMinus,
        'pi0': acts.PdgParticle.ePionZero,
        'pion+': acts.PdgParticle.ePionPlus,
        'pion-': acts.PdgParticle.ePionMinus,
        'pion0': acts.PdgParticle.ePionZero,
        'proton': acts.PdgParticle.eProton,
        'antiproton': acts.PdgParticle.eAntiProton,
        'p': acts.PdgParticle.eProton,
        'pbar': acts.PdgParticle.eAntiProton,
        'neutron': acts.PdgParticle.eNeutron,
        'antineutron': acts.PdgParticle.eAntiNeutron,
        'n': acts.PdgParticle.eNeutron,
        'nbar': acts.PdgParticle.eAntiNeutron,
    }
    
    # Check if it's a string name
    if isinstance(particle_name, str):
        name_lower = particle_name.lower()
        if name_lower in name_to_pdg:
            return name_to_pdg[name_lower]
        # Try case-sensitive for things like pi+
        if particle_name in name_to_pdg:
            return name_to_pdg[particle_name]
        
        # Try to parse as integer PDG code
        try:
            pdg_code = int(particle_name)
            return acts.PdgParticle(pdg_code)
        except (ValueError, TypeError):
            pass
    
    # Try as integer directly
    try:
        return acts.PdgParticle(int(particle_name))
    except (ValueError, TypeError):
        pass
    
    raise ValueError(
        f"Unknown particle: {particle_name}. "
        f"Use name like 'gamma', 'e-', 'mu-', 'pi+', 'proton' or PDG code."
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
    particle_pdg = pdg_name_to_particle(particle_name)
    
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
    logger.info(f"  Particle: {particle_name} ({particle_pdg})")
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


def setup_splitting_config(config, logger):
    """Set up splitting configuration for multi-directory generation"""
    try:
        splitting_config = getattr(config, 'splitting_config', {})
        if isinstance(splitting_config, dict):
            splitting_enabled = splitting_config.get('enable', False)
            n_runs = splitting_config.get('n_runs', 1)
        else:
            splitting_enabled = False
            n_runs = 1
        
        logger.info(f"Splitting config: enable={splitting_enabled}, n_runs={n_runs}")
        return splitting_enabled, n_runs
    except Exception as e:
        logger.warning(f"Error accessing splitting config: {e}. Using defaults")
        return False, 1


def main():
    timer = None
    logger = setup_logging("ParticleGun")
    try:
        # Parse arguments and load config
        args = parse_args()
        config = load_config(args)
        
        # Base output directory
        base_output_dir = Path(args.output)
        
        # Check splitting configuration
        splitting_enabled, n_runs = setup_splitting_config(config, logger)
        
        logger.info("=== Particle Gun Generation ===")
        logger.info(f"Base output directory: {base_output_dir}")
        logger.info(f"Events per run: {config.events}")
        
        # Initialize timing at base level
        timer = TimingRecorder(base_output_dir)
        
        # Check if we should do monolithic splitting (multiple directories in one job)
        # This happens when splitting is enabled AND either:
        #   1. No output_subdir specified, OR
        #   2. output_subdir is "all" (from CLI for monolithic runs)
        do_monolithic_splitting = splitting_enabled and (
            not args.output_subdir or args.output_subdir == "all"
        )
        
        if do_monolithic_splitting:
            # Monolithic mode: loop over N run directories
            logger.info(f"Generating {n_runs} separate run directories")
            
            with timer.record("Particle Gun Generation (all runs)"):
                for run_id in range(n_runs):
                    run_output_dir = base_output_dir / str(run_id)
                    run_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    logger.info(f"\n=== Generating run {run_id}/{n_runs} ===")
                    
                    # Create a modified config with run-specific seed
                    import copy
                    run_config = copy.copy(config)
                    
                    # Modify seed for this run
                    base_seed = config.seed or 42
                    run_config.seed = base_seed + run_id
                    
                    logger.info(f"Run {run_id} seed: {run_config.seed}")
                    
                    # Generate events for this run
                    final_output = generate_particle_gun_events(run_output_dir, run_config, logger)
                    logger.info(f"Run {run_id} completed: {final_output}")
            
            logger.info(f"\n=== All {n_runs} runs completed successfully ===")
            
        else:
            # Single directory mode (distributed SLURM with numeric output_subdir)
            output_dir = base_output_dir
            if args.output_subdir and args.output_subdir != "all":
                output_dir = base_output_dir / args.output_subdir
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Output directory: {output_dir}")
            
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
