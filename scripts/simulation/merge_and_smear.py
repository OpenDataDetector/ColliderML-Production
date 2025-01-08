import time
from pathlib import Path
import acts
import pyhepmc as hep
from pyhepmc.io import WriterAscii
import numpy as np
import traceback
from utils.logging import setup_logging, TimingRecorder
from utils.config import create_base_parser, load_config

u = acts.UnitConstants

def parse_args():
    """Parse command line arguments"""
    parser = create_base_parser("Merge and smear HepMC3 events")
    parser.add_argument(
        "--signal-file",
        help="Input signal HepMC3 file (default: {output_dir}/events.hepmc3)",
        type=Path,
        default=None
    )
    parser.add_argument(
        "--pileup-file",
        help="Input pileup HepMC3 file (default: {output_dir}/events_pileup.hepmc3)",
        type=Path,
        default=None
    )
    # Vertex smearing parameters
    parser.add_argument(
        "--vertex-sigma-xy",
        help="Sigma for vertex smearing in x/y [mm]",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--vertex-sigma-z",
        help="Sigma for vertex smearing in z [mm]",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--vertex-sigma-t",
        help="Sigma for vertex smearing in time [ns]",
        type=float,
        default=0.0
    )
    return parser.parse_args()

def smear_vertex_position(event, vertex_sigmas):
    """Apply Gaussian smearing to all vertices in an event"""
    # Generate one smearing offset for the whole event
    offset = np.array([
        np.random.normal(0, vertex_sigmas['xy']),  # mm
        np.random.normal(0, vertex_sigmas['xy']),  # mm
        np.random.normal(0, vertex_sigmas['z']),   # mm
        np.random.normal(0, vertex_sigmas['t'])    # mm (time in light-mm)
    ])
    
    # Create new position vector for each vertex
    for vertex in event.vertices:
        new_pos = hep.FourVector(
            vertex.position.x + offset[0],
            vertex.position.y + offset[1],
            vertex.position.z + offset[2],
            vertex.position.t + offset[3]
        )
        vertex.position = new_pos
    
    return event

def merge_events(signal_event, pileup_events, vertex_sigmas, logger):
    """Merge signal and multiple pileup events into a single event with vertex smearing"""
    # Create new event with same units as input
    merged = hep.GenEvent(hep.Units.GEV, hep.Units.MM)
    
    # First add signal event with smearing
    signal_event = smear_vertex_position(signal_event, vertex_sigmas)
    
    # Create new signal vertices
    sig_vertices = []
    for vertex in signal_event.vertices:
        v1 = hep.GenVertex(vertex.position)
        sig_vertices.append(v1)
    
    # Add signal particles and connect them to vertices
    for particle in signal_event.particles:
        p1 = hep.GenParticle(
            particle.momentum,
            particle.pid,
            particle.status
        )
        p1.generated_mass = particle.generated_mass
        
        # Handle production vertex
        if particle.production_vertex.id < 0:
            production_vertex = particle.production_vertex.id
            sig_vertices[abs(production_vertex)-1].add_particle_out(p1)
            merged.add_particle(p1)
        else:
            merged.add_particle(p1)
        
        # Handle end vertex if it exists
        if particle.end_vertex:
            end_vertex = particle.end_vertex.id
            sig_vertices[abs(end_vertex)-1].add_particle_in(p1)
    
    # Add all signal vertices
    for vertex in sig_vertices:
        merged.add_vertex(vertex)
    
    # Now add pileup events with smearing
    for pileup_event in pileup_events:
        pileup_event = smear_vertex_position(pileup_event, vertex_sigmas)
        
        # Create new pileup vertices
        pileup_vertices = []
        for vertex in pileup_event.vertices:
            v1 = hep.GenVertex(vertex.position)
            pileup_vertices.append(v1)
        
        # Add pileup particles and connect them to vertices
        for particle in pileup_event.particles:
            p1 = hep.GenParticle(
                particle.momentum,
                particle.pid,
                particle.status
            )
            p1.generated_mass = particle.generated_mass
            
            if particle.production_vertex.id < 0:
                production_vertex = particle.production_vertex.id
                pileup_vertices[abs(production_vertex)-1].add_particle_out(p1)
                merged.add_particle(p1)
            else:
                merged.add_particle(p1)
            
            if particle.end_vertex:
                end_vertex = particle.end_vertex.id
                pileup_vertices[abs(end_vertex)-1].add_particle_in(p1)
        
        # Add all pileup vertices
        for vertex in pileup_vertices:
            merged.add_vertex(vertex)
    
    return merged

def merge_hepmc_files(signal_path, pileup_path, output_path, vertex_sigmas, logger=None):
    """Merge signal and pileup HepMC3 files into a single file with vertex smearing"""
    logger = logger or setup_logging("MergeHepMC")
    logger.info(f"Merging HepMC3 files:")
    logger.info(f"Signal: {signal_path}")
    logger.info(f"Pileup: {pileup_path}")
    logger.info(f"Output: {output_path}")
    
    # Load all signal events
    signal_events = []
    with hep.open(signal_path) as f:
        for event in f:
            signal_events.append(event)
    
    # Load all pileup events
    pileup_events = []
    with hep.open(pileup_path) as f:
        for event in f:
            pileup_events.append(event)
    
    # Calculate pileup events per signal event
    n_pileup_per_signal = len(pileup_events) // len(signal_events)
    logger.info(f"Found {len(signal_events)} signal events with {n_pileup_per_signal} pileup events each")
    
    # Write merged events
    with WriterAscii(str(output_path)) as f:
        for i, signal_event in enumerate(signal_events):
            # Get corresponding pileup events for this signal event
            start_idx = i * n_pileup_per_signal
            end_idx = start_idx + n_pileup_per_signal
            event_pileup = pileup_events[start_idx:end_idx]
            
            # Merge events with vertex smearing
            merged_event = merge_events(signal_event, event_pileup, vertex_sigmas, logger)
            merged_event.event_number = i
            
            # Write merged event
            f.write_event(merged_event)
    
    logger.info("Merge complete")

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
        
        # Set default input paths if not specified
        signal_path = args.signal_file or output_dir / "events.hepmc3"
        pileup_path = args.pileup_file or output_dir / "events_pileup.hepmc3"
        
        # Initialize timing recorder
        timer = TimingRecorder(output_dir)
        
        # Set up vertex smearing parameters with proper unit conversion
        vertex_sigmas = {
            'xy': config.vertex_sigma_xy,                    # mm
            'z': config.vertex_sigma_z,                      # mm
            't': config.vertex_sigma_t * u.ns               # Convert ns to light-mm
        }
        
        # Run merge and smear
        merged_path = output_dir / "merged_events.hepmc3"
        with timer.record("Merge and Smear"):
            merge_hepmc_files(
                signal_path,
                pileup_path,
                merged_path,
                vertex_sigmas,
                logger
            )
        
        # Write timing report
        timer.write_report()
        
        logger.info("Merge and smear completed successfully")
        logger.info(f"Output file: {merged_path}")
        
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()