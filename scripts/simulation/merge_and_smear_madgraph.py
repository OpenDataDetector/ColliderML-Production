import time
from pathlib import Path
import acts
import pyhepmc as hep
from pyhepmc.io import WriterAscii
import numpy as np
import traceback
from utils.app_logging import setup_logging, TimingRecorder
from utils.config import create_base_parser, load_config
from tqdm import tqdm

u = acts.UnitConstants

def parse_args():
    """Parse command line arguments"""
    parser = create_base_parser("Merge and smear HepMC3 events")
    parser.add_argument(
        "--signal-file",
        help="Input signal HepMC3 file (default: {output_dir}/events.hepmc3 or events.hepmc.gz)",
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
        default=None
    )
    parser.add_argument(
        "--vertex-sigma-z",
        help="Sigma for vertex smearing in z [mm]",
        type=float,
        default=None
    )
    parser.add_argument(
        "--vertex-sigma-t",
        help="Sigma for vertex smearing in time [ns]",
        type=float,
        default=None
    )
    return parser.parse_args()

def smear_vertex_position(event, vertex_sigmas_mm_ns):
    """Apply Gaussian smearing to all vertices in an event.
    vertex_sigmas_mm_ns: dict with 'xy' (mm), 'z' (mm), 't' (ns)
    """
    # Generate one smearing offset for the whole event
    offset = np.array([
        np.random.normal(0, vertex_sigmas_mm_ns['xy']),      # mm
        np.random.normal(0, vertex_sigmas_mm_ns['xy']),      # mm
        np.random.normal(0, vertex_sigmas_mm_ns['z']),       # mm
        np.random.normal(0, vertex_sigmas_mm_ns['t']) * u.ns # Convert time smearing from ns to light-mm for HepMC
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

def merge_events(signal_event, pileup_event, vertex_sigmas, smear_pileup=True, logger=None):
    """Merge signal and pileup events into a single event with vertex smearing"""
    # Create new event with same units as input
    merged = hep.GenEvent(hep.Units.GEV, hep.Units.MM)
    
    # First add signal event with smearing (always smear signal)
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
    
    # Now add pileup events with conditional smearing
    if smear_pileup:
        pileup_event = smear_vertex_position(pileup_event, vertex_sigmas)
        if logger:
            logger.debug("Applied vertex smearing to pileup event")
    else:
        if logger:
            logger.debug("Skipping vertex smearing for pileup event (already smeared)")
    
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

def merge_hepmc_files(signal_path, pileup_path, num_events, output_path, vertex_sigmas_mm_ns=None, config=None, smear_pileup=True, logger=None):
    """Merge signal and pileup HepMC3 files into a single file with vertex smearing.
    vertex_sigmas_mm_ns: Optional dict with 'xy' (mm), 'z' (mm), 't' (ns) for smearing.
                         If None, and config is provided, values are taken from config.
    config: Optional config object to retrieve vertex smearing if vertex_sigmas_mm_ns is None.
    smear_pileup: If True, apply vertex smearing to pileup events. If False, only smear signal events.
    """
    logger = logger or setup_logging("MergeHepMC")
    logger.info(f"Merging HepMC3 files:")
    logger.info(f"Signal: {signal_path}")
    logger.info(f"Pileup: {pileup_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Smear pileup events: {smear_pileup}")
    
    effective_vertex_sigmas = {}
    if vertex_sigmas_mm_ns:
        effective_vertex_sigmas = vertex_sigmas_mm_ns
        logger.info(f"Using provided vertex_sigmas: {effective_vertex_sigmas}")
    elif config:
        logger.info("Using vertex_sigmas from config object.")
        effective_vertex_sigmas = {
            'xy': getattr(config, 'vertex_sigma_xy', 0.0),
            'z': getattr(config, 'vertex_sigma_z', 0.0),
            't': getattr(config, 'vertex_sigma_t', 0.0)  # Keep as ns, smearing function will convert
        }
        logger.info(f"Derived vertex_sigmas from config: {effective_vertex_sigmas}")
    else:
        logger.warning("No vertex_sigmas or config provided for smearing. Assuming zero smearing.")
        effective_vertex_sigmas = {'xy': 0.0, 'z': 0.0, 't': 0.0}

    # Load all signal events
    logger.info(f"Loading signal events from {signal_path}")
    signal_events = []
    with hep.open(signal_path) as f:
        for i, event in tqdm(enumerate(f), total=num_events):
            if i >= num_events:
                break
            signal_events.append(event)
    
    # Load all pileup events
    logger.info(f"Loading pileup events from {pileup_path}")
    pileup_events = []
    with hep.open(pileup_path) as f:
        for i, event in tqdm(enumerate(f), total=num_events):
            if i >= num_events:
                break
            pileup_events.append(event)

    if len(signal_events) != len(pileup_events):
        print(f"Warning: Number of signal events ({len(signal_events)}) does not match number of pileup events ({len(pileup_events)})")
        print(f"Truncating to min of {min(len(signal_events), len(pileup_events))}")
        signal_events = signal_events[:min(len(signal_events), len(pileup_events))]
        pileup_events = pileup_events[:min(len(signal_events), len(pileup_events))]
    
    # Calculate pileup events per signal event
    n_pileup_per_signal = len(pileup_events) // len(signal_events)
    logger.info(f"Found {len(signal_events)} signal events with {n_pileup_per_signal} pileup events each")
    
    # Write merged events
    logger.info(f"Writing merged events to {output_path}")
    with WriterAscii(str(output_path)) as f:
        for i, (signal_event, pileup_event) in tqdm(enumerate(zip(signal_events, pileup_events)), total=len(signal_events)):
            # Merge events with vertex smearing
            merged_event = merge_events(signal_event, pileup_event, effective_vertex_sigmas, smear_pileup, logger)
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
        if args.signal_file:
            signal_path = args.signal_file
        else:
            # Try both naming conventions from MadGraph
            signal_candidates = [
                output_dir / "events.hepmc3",     # From MadGraph with splitting  
                output_dir / "events.hepmc.gz"   # From MadGraph without splitting
            ]
            signal_path = None
            for candidate in signal_candidates:
                if candidate.exists():
                    signal_path = candidate
                    break
            if signal_path is None:
                # Default to the first option for error message consistency
                signal_path = signal_candidates[0]
        
        pileup_path = args.pileup_file or output_dir / "events_pileup.hepmc3"
        num_events = config.events or 1
        
        # Initialize timing recorder
        timer = TimingRecorder(output_dir)
        
        # Run merge and smear
        merged_path = output_dir / "merged_events.hepmc3"
        with timer.record("Merge and Smear"):
            merge_hepmc_files(
                signal_path,
                pileup_path,
                num_events,
                merged_path,
                config=config,
                smear_pileup=True,
                logger=logger
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