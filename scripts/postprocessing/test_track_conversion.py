#!/usr/bin/env python3
"""
Test script for track conversion functionality.
"""

import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

from convert_tracks import process_run_for_tracks, build_hdf5_tracks

def test_run_conversion(
    run_dir: str,
    run_num: int = 0,
    run_size: int = 10,
    tracks_csv_pattern: str = "event{:09d}-tracks_ambi.csv",
    tracksummary_file: str = "tracksummary_ambi.root",
    simhits_file: str = "simhits.root", 
    edm4hep_file: str = "edm4hep.root"
):
    """
    Test conversion of a full run.
    
    Args:
        run_dir: Path to run directory
        run_num: Run number to test
        run_size: Number of events in the run
        tracks_csv_pattern: Pattern for tracks CSV filenames
        tracksummary_file: Name of track summary ROOT file
        simhits_file: Name of simulated hits ROOT file
        edm4hep_file: Name of EDM4hep ROOT file
    """
    print(f"\nTesting run conversion for run {run_num}")
    print("=" * 80)
    
    # Process run
    print("\nProcessing run...")
    run_events = process_run_for_tracks(
        run_dir,
        run_num,
        run_size,
        tracks_csv_pattern,
        tracksummary_file,
        simhits_file,
        edm4hep_file
    )
    print("Run processed successfully")
    
    if not run_events:
        print("No events processed in run")
        return
        
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 40)
    print(f"Number of events processed: {len(run_events)}")
    
    all_events_df = pd.concat(run_events, ignore_index=True)
    events_summary = all_events_df.groupby('event_id').agg({
        'track_id': 'count',
        'num_hits': 'mean',
        'chi2': 'mean',
        'majority_particle_id': lambda x: len(x.unique())
    }).rename(columns={
        'track_id': 'num_tracks',
        'majority_particle_id': 'unique_particles'
    })
    
    print("\nPer-event statistics:")
    print(events_summary)
    
    print("\nOverall statistics:")
    print(f"Total tracks: {len(all_events_df)}")
    print(f"Average tracks per event: {events_summary['num_tracks'].mean():.2f}")
    print(f"Average hits per track: {all_events_df['num_hits'].mean():.2f}")
    print(f"Average chi2: {all_events_df['chi2'].mean():.2f}")
    print(f"Total unique particles: {len(all_events_df['majority_particle_id'].unique())}")
    
    # Test HDF5 writing
    print("\nTesting HDF5 writing...")
    test_output = "test_output.h5"
    if os.path.exists(test_output):
        os.remove(test_output)
        
    build_hdf5_tracks(all_events_df, test_output)
    
    # Verify HDF5 structure
    print("\nVerifying HDF5 structure...")
    with h5py.File(test_output, 'r') as f:
        events = f['events']
        
        print("\nHDF5 Structure:")
        print("-" * 40)
        print("events/")
        for event_id in range(len(run_events)):
            event_group = events[f'event_{event_id}']
            print(f"  event_{event_id}/")
            print("    tracks/")
            print("    hit_ids/")
            
            tracks = event_group['tracks'][()]
            hit_ids = event_group['hit_ids'][()]
            
            print(f"\nEvent {event_id}:")
            print(f"  Number of tracks: {len(tracks)}")
            print(f"  Number of hit_id arrays: {len(hit_ids)}")
        
    os.remove(test_output)
    print("\nTest completed successfully!")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test track conversion functionality")
    parser.add_argument("run_dir", help="Directory containing run data")
    parser.add_argument("--run-num", type=int, default=0,
                      help="Run number to test")
    parser.add_argument("--run-size", type=int, default=10,
                      help="Number of events in run")
    
    args = parser.parse_args()
    test_run_conversion(args.run_dir, args.run_num, args.run_size)

if __name__ == "__main__":
    main()