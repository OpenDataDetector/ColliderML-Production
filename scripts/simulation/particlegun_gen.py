import time
from pathlib import Path
import pyhepmc as hep
import numpy as np
import traceback
from tqdm import tqdm

from utils.app_logging import setup_logging, TimingRecorder
from utils.config import create_base_parser, load_config


def parse_args():
    """Parse command line arguments"""
    parser = create_base_parser("Particle gun event generation with log-uniform energy distribution")
    
    parser.add_argument(
        "--particle",
        help="Particle type (PDG name or code)",
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
        "--energy-distribution",
        help="Energy distribution type",
        type=str,
        choices=["uniform", "log-uniform"],
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


def pdg_name_to_code(particle_name):
    """Convert particle name to PDG code"""
    pdg_map = {
        'e-': 11, 'e+': -11,
        'mu-': 13, 'mu+': -13,
        'pi+': 211, 'pi-': -211, 'pi0': 111,
        'gamma': 22,
        'proton': 2212, 'antiproton': -2212,
        'neutron': 2112, 'antineutron': -2112,
        'K+': 321, 'K-': -321, 'K0': 311,
        'tau-': 15, 'tau+': -15,
    }
    
    try:
        return int(particle_name)
    except ValueError:
        if particle_name in pdg_map:
            return pdg_map[particle_name]
        else:
            raise ValueError(f"Unknown particle name: {particle_name}. Use PDG code or known name.")


def sample_energy(energy_min, energy_max, distribution='uniform'):
    """Sample energy from specified distribution
    
    Args:
        energy_min: Minimum energy [GeV]
        energy_max: Maximum energy [GeV]
        distribution: 'uniform' or 'log-uniform'
        
    Returns:
        float: Sampled energy in GeV
    """
    if distribution == 'log-uniform':
        log_min = np.log(energy_min)
        log_max = np.log(energy_max)
        return np.exp(np.random.uniform(log_min, log_max))
    else:
        return np.random.uniform(energy_min, energy_max)


def theta_from_eta(eta):
    """Convert pseudorapidity to polar angle theta"""
    return 2.0 * np.arctan(np.exp(-eta))


def generate_particle_gun_events(output_dir, config, logger):
    """Generate single particle events with configurable energy distribution
    
    Args:
        output_dir: Output directory path
        config: Configuration object with particle gun parameters
        logger: Logger instance
        
    Returns:
        Path: Path to generated HepMC3 file
    """
    # Get particle gun configuration
    particle_name = getattr(config, 'particle', 'mu-')
    particle_pdg = pdg_name_to_code(particle_name)
    
    energy_min = getattr(config, 'energy_min', 1.0)
    energy_max = getattr(config, 'energy_max', 100.0)
    energy_distribution = getattr(config, 'energy_distribution', 'uniform')
    
    # Angular range (default: uniform in eta)
    eta_min = getattr(config, 'eta_min', -2.5)
    eta_max = getattr(config, 'eta_max', 2.5)
    phi_min = getattr(config, 'phi_min', 0.0)
    phi_max = getattr(config, 'phi_max', 2.0 * np.pi)
    
    n_events = config.events
    seed = config.seed or int(time.time())
    
    logger.info(f"Generating {n_events} particle gun events")
    logger.info(f"  Particle: {particle_name} (PDG {particle_pdg})")
    logger.info(f"  Energy: [{energy_min}, {energy_max}] GeV ({energy_distribution} distribution)")
    logger.info(f"  Eta range: [{eta_min}, {eta_max}]")
    logger.info(f"  Phi range: [{phi_min}, {phi_max}]")
    logger.info(f"  Random seed: {seed}")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Create output file
    output_path = output_dir / "events.hepmc3"
    
    # Generate events and write to HepMC3
    with hep.open(str(output_path), 'w') as f:
        for event_id in tqdm(range(n_events)):
            # Create new event
            evt = hep.GenEvent(hep.Units.GEV, hep.Units.MM)
            evt.event_number = event_id
            
            # Sample energy
            energy = sample_energy(energy_min, energy_max, energy_distribution)
            
            # Sample direction uniformly in eta-phi space
            eta = np.random.uniform(eta_min, eta_max)
            phi = np.random.uniform(phi_min, phi_max)
            
            # Convert eta to theta
            theta = theta_from_eta(eta)
            
            # Calculate momentum components (massless approximation)
            pt = energy * np.sin(theta)
            px = pt * np.cos(phi)
            py = pt * np.sin(phi)
            pz = energy * np.cos(theta)
            
            # Create vertex at origin
            vertex = hep.GenVertex()
            evt.add_vertex(vertex)
            
            # Create particle
            particle = hep.GenParticle(
                hep.FourVector(px, py, pz, energy),
                particle_pdg,
                1  # status: final state particle
            )
            
            # Add particle to vertex
            vertex.add_particle_out(particle)
            
            # Write event
            f.write(evt)
            
            if (event_id + 1) % 1000 == 0:
                logger.info(f"  Generated {event_id + 1}/{n_events} events")
    
    logger.info(f"Particle gun generation completed: {output_path}")
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
