import yaml
from pathlib import Path
import argparse
import hashlib
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def hash_seed_string(seed_str: str) -> int:
    """Convert a string seed pattern into a deterministic integer.
    
    Examples:
        "123" -> 123
        "job_1:proc_2" -> <hash-based integer>
        "$JOB_ID:$PROCESS_ID" -> Will be evaluated at runtime with env vars
    """
    logger.info(f"Constructing seed from input: {seed_str}")
    
    # If it's just a number, return it
    try:
        seed = int(seed_str)
        logger.info(f"Input is numeric, using directly as seed: {seed}")
        return seed
    except ValueError:
        # Hash the string to get a fixed-length bytes object
        hash_obj = hashlib.md5(seed_str.encode())
        # Convert first 4 bytes to integer (using big-endian)
        seed = int.from_bytes(hash_obj.digest()[:4], 'big')
        logger.info(f"Input is string pattern, hashed to seed: {seed} (from pattern: {seed_str})")
        return seed

def create_base_parser(description):
    """Create parser with common arguments"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output", "-o",
        help="Output directory",
        type=Path,
        default=Path.cwd() / "pda_output",
    )
    parser.add_argument(
        "--events", "-n",
        help="Number of events",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--config",
        help="YAML configuration file",
        type=Path,
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
        default="",
    )
    return parser

def load_config(args):
    """Load and merge configuration from YAML file"""
    if args.config is not None:
        with open(args.config) as f:
            config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {args.config}")
        
        # Update args with config values
        for key, value in config.items():
            setattr(args, key, value)
            
    # Convert seed if it's a string pattern
    if args.seed is not None:
        original_seed = args.seed
        args.seed = hash_seed_string(str(args.seed))
        logger.info(f"Final seed value: {args.seed} (from original input: {original_seed})")
    else:
        logger.info("No seed provided, will use time-based seed")
        
    return args