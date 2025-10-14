#!/usr/bin/env python3
"""
Calorimeter digitization using k4ODD.

Reads EDM4hep from simulation stage, applies calo digitization via Key4hep/k4ODD.
Environment setup (Key4hep) is handled by the pipeline via env_setup.yaml.
"""

import subprocess
import sys
from pathlib import Path
import traceback

# Add utils to path (same pattern as other simulation scripts)
sys.path.insert(0, str(Path(__file__).parent))

from utils.app_logging import setup_logging, TimingRecorder
from utils.config import create_base_parser, load_config


def run_k4odd_digitization(input_file, output_dir, config, logger):
    """
    Run k4ODD calorimeter digitization.
    
    Assumes Key4hep environment is already loaded by the pipeline.
    Simply calls k4run with appropriate arguments.
    
    Args:
        input_file: Path to input EDM4hep file
        output_dir: Output directory
        config: Configuration object
        logger: Logger instance
        
    Returns:
        Path: Output file path
        
    Raises:
        FileNotFoundError: If k4ODD script not found
        RuntimeError: If digitization fails
    """
    
    # Get k4ODD script path from environment variable (set by env_setup.yaml)
    import os
    k4odd_base = os.environ.get('K4ODD_PATH')
    if not k4odd_base:
        raise ValueError("K4ODD_PATH not found in environment. This should be set by env_setup.yaml")
    
    k4odd_script = Path(k4odd_base) / "k4ODD/options/ODDdigitisation.py"
    
    if not k4odd_script.exists():
        raise FileNotFoundError(f"k4ODD script not found: {k4odd_script}")
    
    # Get number of events (default to all if not specified)
    events = getattr(config, 'events', -1)
    
    # Convert paths to absolute
    input_file = input_file.resolve()
    output_file = (output_dir / "edm4hep_digitized.root").resolve()
    
    logger.info(f"Calorimeter Digitization (k4ODD)")
    logger.info(f"  Input:  {input_file}")
    logger.info(f"  Output: {output_file}")
    logger.info(f"  Events: {events if events > 0 else 'all'}")
    
    # Build k4run command - environment is already set up by pipeline
    cmd = [
        "k4run",
        str(k4odd_script),
        f"--inputFile={input_file}",
        f"--outputFile={output_file}",
        f"--events={events}"
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    # Execute k4run
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check for errors
    if result.returncode != 0:
        logger.error(f"k4run failed with return code {result.returncode}")
        logger.error(f"STDOUT (last 2000 chars): {result.stdout[-2000:]}")
        logger.error(f"STDERR (last 2000 chars): {result.stderr[-2000:]}")
        raise RuntimeError("Calorimeter digitization failed")
    
    # Validate output file exists
    if not output_file.exists():
        raise RuntimeError(f"Output file not created: {output_file}")
    
    # Log file sizes
    output_size = output_file.stat().st_size / (1024**2)  # MB
    input_size = input_file.stat().st_size / (1024**2)  # MB
    
    logger.info(f"✓ Digitization complete")
    logger.info(f"  Input:  {input_size:.1f} MB")
    logger.info(f"  Output: {output_size:.1f} MB")
    if input_size > 0:
        logger.info(f"  Size ratio: {output_size/input_size:.2f}x")
    
    # Basic sanity check
    if output_size < 0.001:
        logger.warning(f"Output file very small ({output_size:.3f} MB) - may indicate failure")
    
    return output_file


def main():
    """Main entry point for calorimeter digitization."""
    logger = setup_logging()
    
    try:
        # Parse arguments (following standard ColliderML pattern)
        parser = create_base_parser("Calorimeter digitization via k4ODD")
        parser.add_argument(
            "--input-file",
            help="Input EDM4hep file (default: {output_dir}/edm4hep.root)",
            type=Path,
            default=None
        )
        args = parser.parse_args()
        config = load_config(args)
        
        # Setup output directory
        output_dir = Path(args.output)
        if hasattr(args, 'output_subdir') and args.output_subdir:
            output_dir = output_dir / args.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find input EDM4hep file (default from simulation stage)
        if args.input_file:
            input_file = Path(args.input_file)
        else:
            input_file = output_dir / "edm4hep.root"
        
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        logger.info(f"=" * 80)
        logger.info(f"Starting Calorimeter Digitization")
        logger.info(f"=" * 80)
        
        # Initialize timing
        timer = TimingRecorder(output_dir)
        
        # Run digitization
        with timer.record("Calo Digitization"):
            output_file = run_k4odd_digitization(input_file, output_dir, config, logger)
        
        # Write timing report
        timer.write_report()
        
        logger.info(f"=" * 80)
        logger.info(f"✓ Calorimeter digitization completed successfully")
        logger.info(f"  Output: {output_file}")
        logger.info(f"=" * 80)
        
    except Exception as e:
        logger.error(f"Fatal error in calo digitization: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

