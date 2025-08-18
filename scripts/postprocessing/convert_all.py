#!/usr/bin/env python3
"""
Run all EDM4HEP to HDF5 conversions in sequence, driven by a YAML config.
"""

import argparse
import time
from pathlib import Path
import yaml
import logging

from convert_particles import convert_particles
from convert_calorimeter import convert_calorimeter
from convert_tracks import convert_tracks
from convert_digihits import convert_digihits

from utils.path_utils import make_dir

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_all_from_config(config_path: str) -> None:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    campaign = config["campaign"]
    dataset = config["dataset"]
    version = config["version"]

    common_cfg = config.get("common", {})
    # Use a single root for both sim and postprocessing
    input_base_dir = Path(common_cfg["output_base_dir"]) / campaign / dataset / version
    output_base_dir = Path(common_cfg["output_base_dir"]) 

    # Chunking
    chunk_size = int(config.get("chunk_size", 1000))
    run_size = int(config.get("run_size", 10))

    # Objects to convert
    objects = config.get("objects", ["tracker_hits", "tracks", "particles", "calorimeter"])  # default set
    objects = [obj.lower() for obj in objects]

    dataset_base = f"{campaign}/{dataset}/{version}"

    start_time = time.time()

    if "tracker_hits" in objects:
        logger.info("\n=== Converting Tracker Hits (digitised) ===")
        # convert_digihits manages its own subpath (now reco/tracker_hits)
        convert_digihits(input_base_dir, output_base_dir, dataset_base, chunk_size, run_size)

    if "tracks" in objects:
        logger.info("\n=== Converting Tracks ===")
        convert_tracks(input_base_dir, output_base_dir, f"{dataset_base}/tracks", chunk_size, run_size)

    if "particles" in objects:
        logger.info("\n=== Converting Particles ===")
        # convert_particles manages its own subpath (truth/particles)
        convert_particles(input_base_dir, output_base_dir, dataset_base, chunk_size, run_size)

    if "calorimeter" in objects:
        logger.info("\n=== Converting Calorimeter Data ===")
        convert_calorimeter(input_base_dir, output_base_dir, f"{dataset_base}/calorimeter", chunk_size, run_size)

    end_time = time.time()
    logger.info(f"\nTotal conversion time: {end_time - start_time:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description="Convert all EDM4HEP data to HDF5 (config-driven)")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    convert_all_from_config(args.config)

if __name__ == "__main__":
    main() 