import yaml
from pathlib import Path
import argparse

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
        help="Random seed. If not specified, uses current time.",
        type=int,
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
        # Update args with config values
        for key, value in config.items():
            setattr(args, key, value)
    return args