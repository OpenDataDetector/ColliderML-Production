import yaml
from pathlib import Path
import argparse
import hashlib
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def hash_seed_string(seed_str: str) -> int:
    """Convert a string seed pattern into a deterministic positive integer.
    
    All seeds are constrained to Pythia8's allowed range 1..900000000 to avoid
    out-of-range errors while remaining compatible with MadGraph and others.
    
    Examples:
        "123" -> 123
        "job_1:proc_2" -> <hash-based positive integer>
        "$JOB_ID:$PROCESS_ID" -> Will be evaluated at runtime with env vars
    """
    logger.info(f"Constructing seed from input: {seed_str}")
    
    # If it's just a number, return it (constrained to Pythia8's allowed range)
    try:
        seed = int(seed_str)
        # Ensure positive and within Pythia8 range [1, 900000000]
        seed = abs(seed) % 900000000
        seed = 1 if seed == 0 else seed
        logger.info(f"Input is numeric, using as seed: {seed}")
        return seed
    except ValueError:
        # Hash the string to get a fixed-length bytes object
        hash_obj = hashlib.md5(seed_str.encode())
        # Convert first 4 bytes to signed integer (using big-endian, interpret as signed)
        seed = int.from_bytes(hash_obj.digest()[:4], 'big', signed=True)
        # Ensure positive and within Pythia8 range [1, 900000000]
        seed = abs(seed) % 900000000
        seed = 1 if seed == 0 else seed
        logger.info(f"Input is string pattern, hashed to seed: {seed} (from pattern: {seed_str})")
        return seed

def create_base_parser(description):
    """Create parser with common arguments, no defaults"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output", "-o",
        help="Output directory",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--events", "-n",
        help="Number of events",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--config",
        help="YAML configuration file",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--seed",
        help="Random seed. Can be an integer or a string like '$JOB_ID:$PROCESS_ID'",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output-subdir",
        help="Output subdirectory (useful for parallel processing)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--performance-metrics",
        help="Enable performance metrics collection and output",
        action="store_true",
        default=None,
    )
    return parser

def load_config(args):
    """Load and merge configuration from YAML file, with CLI args taking precedence"""
    if args.config is not None:
        with open(args.config) as f:
            config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {args.config}")
        # Only set values from config if the arg is None
        for key, value in config.items():
            if getattr(args, key, None) is None:
                setattr(args, key, value)
            
    # Convert seed if it's a string pattern
    if args.seed is not None:
        original_seed = args.seed
        args.seed = hash_seed_string(str(args.seed))
        logger.info(f"Final seed value: {args.seed} (from original input: {original_seed})")
    else:
        # Use current time as default seed, constrained to Pythia8's allowed range
        args.seed = abs(int(time.time())) % 900000000
        args.seed = 1 if args.seed == 0 else args.seed
        logger.info(f"No seed provided, using time-based seed: {args.seed}")
        
    return args