"""
Configuration utilities for postprocessing scripts.
"""

import yaml
from pathlib import Path
import argparse
import logging
logger = logging.getLogger(__name__)

def create_base_parser(description: str) -> argparse.ArgumentParser:
    """
    Create parser with common arguments for postprocessing scripts.
    Only the config file is required, all other arguments are optional
    and will be loaded from the config file if not specified.
    
    Args:
        description: Description of the script
        
    Returns:
        ArgumentParser with common arguments
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        help="YAML configuration file",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--base-dir",
        help="Base directory containing EDM4HEP files",
        type=Path,
        default=None
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for processed files",
        type=Path,
        default=None
    )
    parser.add_argument(
        "--dataset-name",
        help="Name of the dataset",
        type=str,
        default=None
    )
    parser.add_argument(
        "--chunk-size",
        help="Number of events per output file",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--run-size",
        help="Number of events per run",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--chunk-index",
        help="Process only this chunk index (0-based); overrides max-chunks",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-chunks",
        help="Maximum number of chunks to process in interactive/testing",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-runs",
        help="Maximum number of simulation runs to consider (caps input)",
        type=int,
        default=None,
    )
    return parser

def load_config(args: argparse.Namespace) -> argparse.Namespace:
    """
    Load and merge configuration from YAML file.
    Command line arguments take precedence over YAML config.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Updated arguments with YAML config merged in
    """
    logger.info(f"Loading configuration from {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f)
            
    # Get the original values that were set from command line
    cli_values = {}
    for key, val in vars(args).items():
        if val is not None and key != 'config':
            cli_values[key] = val
                
    # Update args with config values
    for key, value in config.items():
        if key not in cli_values:  # Don't overwrite CLI values
            setattr(args, key, value)
                
    # Validate required fields are present
    required_fields = ['base_dir', 'output_dir', 'dataset_name']
    missing_fields = [field for field in required_fields if getattr(args, field, None) is None]
    if missing_fields:
        raise ValueError(f"Missing required fields in config: {', '.join(missing_fields)}")
                
    logger.info("Configuration loaded and merged with command line arguments")
        
    return args 