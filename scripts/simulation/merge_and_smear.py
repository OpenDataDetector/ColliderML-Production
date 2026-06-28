import time
from pathlib import Path
import acts
import acts.examples  # noqa: F401  (UniformVertexGenerator / RandomEngine for uniform smearing)
import pyhepmc as hep
from pyhepmc.io import WriterAscii
import numpy as np
import traceback
from utils.app_logging import setup_logging, TimingRecorder
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

def make_offset_fn(config, logger):
    """Return a sampler offset_fn() -> (dx, dy, dz, dt) in HepMC units (mm, mm, mm, light-mm).

    Uniform "drifting beamspot" when the config sets vertex_uniform + vertex_min/vertex_max
    (ACTS UniformVertexGenerator, seeded by config.seed for reproducibility); otherwise the
    original Gaussian smearing from vertex_sigma_xy/z/t (back-compat). Each call returns one
    offset for a whole (sub-)event.
    """
    seed = int(getattr(config, 'seed', 0) or 0)

    if getattr(config, 'vertex_uniform', False) and getattr(config, 'vertex_min', None) \
            and getattr(config, 'vertex_max', None):
        vmin, vmax = config.vertex_min, config.vertex_max
        vg = acts.examples.UniformVertexGenerator(
            min=acts.Vector4(vmin[0] * u.mm, vmin[1] * u.mm, vmin[2] * u.mm, vmin[3] * u.ns),
            max=acts.Vector4(vmax[0] * u.mm, vmax[1] * u.mm, vmax[2] * u.mm, vmax[3] * u.ns),
        )
        rng = acts.examples.RandomEngine(seed=seed)
        logger.info(f"Uniform vertex smearing: min={vmin}, max={vmax} (x,y,z mm; t ns), seed={seed}")
        counter = {'i': 0}

        def offset_fn():
            i = counter['i']
            counter['i'] += 1
            o = vg(rng, i)
            return (o[0], o[1], o[2], o[3])

        return offset_fn

    # Back-compat: Gaussian smearing from sigma_xy/z/t (global np.random, exactly as before)
    sigma_xy = getattr(config, 'vertex_sigma_xy', 0.0) or 0.0
    sigma_z = getattr(config, 'vertex_sigma_z', 0.0) or 0.0
    sigma_t = (getattr(config, 'vertex_sigma_t', 0.0) or 0.0) * u.ns  # ns -> light-mm
    logger.info(f"Gaussian vertex smearing: sigma_xy={sigma_xy} mm, sigma_z={sigma_z} mm, "
                f"sigma_t={sigma_t} (light-mm)")

    def offset_fn():
        return (np.random.normal(0, sigma_xy), np.random.normal(0, sigma_xy),
                np.random.normal(0, sigma_z), np.random.normal(0, sigma_t))

    return offset_fn


def smear_vertex_position(event, offset):
    """Translate all vertices in an event by a precomputed offset (dx, dy, dz, dt)."""
    for vertex in event.vertices:
        p = vertex.position
        vertex.position = hep.FourVector(
            p.x + offset[0], p.y + offset[1], p.z + offset[2], p.t + offset[3]
        )
    return event

def merge_events(signal_event, pileup_events, offset_fn, logger):
    """Merge signal and multiple pileup events into a single event with vertex smearing"""
    # Create new event with same units as input
    merged = hep.GenEvent(hep.Units.GEV, hep.Units.MM)

    # First add signal event with smearing
    signal_event = smear_vertex_position(signal_event, offset_fn())
    
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
        pileup_event = smear_vertex_position(pileup_event, offset_fn())
        
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

def smear_only_hepmc_file(signal_path, output_path, offset_fn, logger=None):
    """Smear vertices of a signal-only HepMC3 file (no pileup) and write merged_events."""
    logger = logger or setup_logging("MergeHepMC")
    logger.info(f"Smear-only (no pileup): {signal_path} -> {output_path}")
    n = 0
    with hep.open(str(signal_path)) as f_in, WriterAscii(str(output_path)) as f_out:
        for event in f_in:
            smear_vertex_position(event, offset_fn())
            event.event_number = n
            f_out.write_event(event)
            n += 1
    logger.info(f"Smear-only complete: {n} events")


def merge_hepmc_files(signal_path, pileup_path, output_path, offset_fn, logger=None):
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
        
        # Set default input paths if not specified (signal filename is configurable, e.g.
        # ttbar writes events.hepmc; particle/pythia gen writes events.hepmc3).
        input_filename = getattr(config, 'input_filename', 'events.hepmc3')
        signal_path = args.signal_file or output_dir / input_filename
        pileup_path = args.pileup_file or output_dir / "events_pileup.hepmc3"

        # Initialize timing recorder
        timer = TimingRecorder(output_dir)

        # Build the per-event vertex-offset sampler (uniform drifting beamspot, or Gaussian)
        offset_fn = make_offset_fn(config, logger)

        # Merge+smear when pileup is present; otherwise smear-only (e.g. hardscatter, no pileup)
        merged_path = output_dir / "merged_events.hepmc3"
        with timer.record("Merge and Smear"):
            if Path(pileup_path).exists():
                merge_hepmc_files(signal_path, pileup_path, merged_path, offset_fn, logger)
            else:
                logger.info(f"No pileup file at {pileup_path}; running smear-only.")
                smear_only_hepmc_file(signal_path, merged_path, offset_fn, logger)
        
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