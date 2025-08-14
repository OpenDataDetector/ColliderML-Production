#!/usr/bin/env python3
"""
Run all EDM4HEP to HDF5 conversions in sequence.
"""

import argparse
import time
from pathlib import Path

from convert_hits import convert_hits
from convert_particles import convert_particles
from convert_calorimeter import convert_calorimeter
from convert_tracks import convert_tracks
from convert_digihits import convert_digihits

def convert_all(
    base_dir: str,
    output_base_dir: str,
    dataset_name: str,
    chunk_size: int = 1000,
    run_size: int = 10,
) -> None:
    """
    Run all EDM4HEP to HDF5 conversions.
    
    Args:
        base_dir: Base directory containing EDM4HEP files
        output_base_dir: Base directory for output files
        dataset_name: Name of the dataset
        chunk_size: Number of events per output file
        run_size: Number of events per run
    """
    start_time = time.time()
    
    print("\n=== Converting Digitized Tracker Measurements ===")
    convert_digihits(base_dir, output_base_dir, f"{dataset_name}/digihits", chunk_size, run_size)

    print("\n=== Converting Tracker Hits ===")
    convert_hits(base_dir, output_base_dir, f"{dataset_name}/hits", chunk_size, run_size)
    
    print("\n=== Converting Tracks ===")
    convert_tracks(base_dir, output_base_dir, f"{dataset_name}/tracks", chunk_size, run_size)
    
    print("\n=== Converting Particles ===")
    convert_particles(base_dir, output_base_dir, f"{dataset_name}/particles", chunk_size, run_size)
    
    print("\n=== Converting Calorimeter Data ===")
    convert_calorimeter(base_dir, output_base_dir, f"{dataset_name}/calorimeter", chunk_size, run_size)
    
    end_time = time.time()
    print(f"\nTotal conversion time: {end_time - start_time:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description="Convert all EDM4HEP data to HDF5")
    parser.add_argument("base_dir", help="Base directory containing EDM4HEP files")
    parser.add_argument("output_dir", help="Output directory for HDF5 files")
    parser.add_argument("dataset_name", help="Name of the dataset")
    parser.add_argument("--chunk-size", type=int, default=1000,
                      help="Number of events per output file")
    parser.add_argument("--run-size", type=int, default=10,
                      help="Number of events per run")
    
    args = parser.parse_args()
    
    convert_all(
        args.base_dir,
        args.output_dir,
        args.dataset_name,
        args.chunk_size,
        args.run_size
    )

if __name__ == "__main__":
    main() 