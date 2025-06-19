#!/usr/bin/env python3

import argparse
from pathlib import Path
import traceback

import acts
import acts.examples
import acts.examples.hepmc3

from utils.app_logging import setup_logging, TimingRecorder
from utils.config import create_base_parser, load_config

u = acts.UnitConstants

def parse_args():
    """Parse command line arguments"""
    parser = create_base_parser("ACTS-based HepMC3 event merging")
    parser.add_argument(
        "--hard-scatter", "--hs", 
        type=Path, 
        help="Hard scatter file (auto-detect if not specified)", 
        default=None
    )
    parser.add_argument(
        "--pileup", "--pu", 
        type=Path, 
        help="Pileup file (auto-detect if not specified)", 
        default=None
    )
    parser.add_argument(
        "--pileup-multiplicity", "--npu",
        type=int,
        help="Pileup multiplicity (overrides config.pileup)",
        default=None,
    )
    parser.add_argument(
        "--force", "-f", 
        action="store_true", 
        help="Force overwrite existing output"
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

def create_vertex_generator(config, logger):
    """Create ACTS GaussianVertexGenerator from config parameters."""
    # Get vertex smearing parameters from config or command line
    sigma_xy = getattr(config, 'vertex_sigma_xy', 0.0) or 0.0
    sigma_z = getattr(config, 'vertex_sigma_z', 0.0) or 0.0
    sigma_t = getattr(config, 'vertex_sigma_t', 0.0) or 0.0
    
    logger.info(f"Vertex smearing: sigma_xy={sigma_xy} mm, sigma_z={sigma_z} mm, sigma_t={sigma_t} ns")
    
    if any(sigma != 0 for sigma in [sigma_xy, sigma_z, sigma_t]):
        return acts.examples.GaussianVertexGenerator(
            stddev=acts.Vector4(
                sigma_xy * u.mm,
                sigma_xy * u.mm,
                sigma_z * u.mm,
                sigma_t * u.ns
            ),
            mean=acts.Vector4(0, 0, 0, 0),
        )
    else:
        logger.info("No vertex smearing configured")
        return None

def detect_compression(output_path):
    """Detect compression mode from file extension"""
    extensions = {
        acts.examples.hepmc3.compressionExtension(c): c
        for c in acts.examples.hepmc3.availableCompressionModes()
        if c != acts.examples.hepmc3.Compression.none
    }
    
    compression = extensions.get(
        output_path.suffix, acts.examples.hepmc3.Compression.none
    )
    
    if compression != acts.examples.hepmc3.Compression.none:
        actual_output_path = Path(output_path.stem)
    else:
        actual_output_path = output_path
    
    return actual_output_path, compression

def find_hard_scatter_file(output_dir, explicit_path=None):
    """Find hard scatter file automatically or use provided path"""
    if explicit_path and explicit_path.exists():
        return explicit_path
    
    # Auto-detect signal file in output directory
    candidates = [
        output_dir / "events_signal.hepmc3",  # From Pythia8 generation
        output_dir / "events.hepmc3",         # From MadGraph with splitting
        output_dir / "events.hepmc.gz",      # From MadGraph without splitting  
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    # Look for any hepmc file that might be signal
    hepmc_files = [f for f in output_dir.glob("*.hepmc*") if "pileup" not in f.name.lower()]
    if len(hepmc_files) == 1:
        return hepmc_files[0]
    
    raise FileNotFoundError(
        f"Hard scatter file not found. Looked for: {[str(c) for c in candidates]}. "
        f"Available files in {output_dir}: {[f.name for f in output_dir.glob('*.hepmc*')]}"
    )

def find_pileup_file(output_dir, explicit_path=None):
    """Find pileup file automatically or use provided path"""
    if explicit_path and explicit_path.exists():
        return explicit_path
    
    # Auto-detect pileup file
    pileup_candidate = output_dir / "events_pileup.hepmc3"
    if pileup_candidate.exists():
        return pileup_candidate
    
    raise FileNotFoundError(f"Pileup file not found: {pileup_candidate}")

def run_acts_merge(hard_scatter_path, pileup_path, output_path, pileup_multiplicity, 
                   vertex_generator, num_events, num_threads, logger):
    """Run ACTS-based merging using HepMC3Reader"""
    logger.info(f"ACTS merging configuration:")
    logger.info(f"  Hard scatter: {hard_scatter_path}")
    logger.info(f"  Pileup: {pileup_path}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Pileup multiplicity: {pileup_multiplicity}")
    logger.info(f"  Events: {num_events}")
    logger.info(f"  Threads: {num_threads}")
    
    # Detect compression
    actual_output_path, compression = detect_compression(output_path)
    
    # Create ACTS sequencer
    s = acts.examples.Sequencer(
        numThreads=num_threads, 
        events=num_events,
        logLevel=acts.logging.INFO
    )
    
    # Random number generator
    rng = acts.examples.RandomNumbers(seed=42)
    
    # Add HepMC3Reader with merging
    s.addReader(
        acts.examples.hepmc3.HepMC3Reader(
            inputPaths=[
                (hard_scatter_path, 1),               # Read each signal event once
                (pileup_path, pileup_multiplicity),   # Read pileup with multiplicity
            ],
            level=acts.logging.INFO,
            outputEvent="merged_events",
            randomNumbers=rng,
            vertexGenerator=vertex_generator,
        )
    )
    
    # Add HepMC3Writer
    s.addWriter(
        acts.examples.hepmc3.HepMC3Writer(
            inputEvent="merged_events",
            outputPath=actual_output_path,
            level=acts.logging.INFO,
            compression=compression,
            writeEventsInOrder=False,
        )
    )
    
    logger.info("Starting ACTS merge...")
    s.run()
    logger.info("ACTS merge completed successfully")

def main():
    try:
        # Parse arguments and load config
        args = parse_args()
        config = load_config(args)
        logger = setup_logging("ACTSMerge")
        
        # Create output directory structure
        output_dir = Path(args.output)
        if args.output_subdir:
            output_dir = output_dir / args.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find input files
        hard_scatter_path = find_hard_scatter_file(output_dir, args.hard_scatter)
        pileup_path = find_pileup_file(output_dir, args.pileup)
        
        logger.info(f"Found hard scatter file: {hard_scatter_path}")
        logger.info(f"Found pileup file: {pileup_path}")
        
        # Determine pileup multiplicity
        pileup_multiplicity = args.pileup_multiplicity or getattr(config, 'pileup', 1)
        
        # Create vertex generator
        vertex_generator = create_vertex_generator(config, logger)
        
        # Determine output file
        output_path = output_dir / "merged_events.hepmc3"
        
        # Check for existing output
        if output_path.exists() and not args.force:
            raise FileExistsError(
                f"Output file {output_path} already exists, use --force to overwrite"
            )
        
        # Initialize timing recorder
        timer = TimingRecorder(output_dir)
        
        # Run ACTS merge
        with timer.record("ACTS Merge"):
            run_acts_merge(
                hard_scatter_path=hard_scatter_path,
                pileup_path=pileup_path,
                output_path=output_path,
                pileup_multiplicity=pileup_multiplicity,
                vertex_generator=vertex_generator,
                num_events=config.events,
                num_threads=getattr(config, 'jobs', 1),
                logger=logger
            )
        
        # Write timing report
        timer.write_report()
        
        logger.info("ACTS merge completed successfully")
        logger.info(f"Output file: {output_path}")
        
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main() 