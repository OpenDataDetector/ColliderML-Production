"""
HDF5 dataset management utilities for storing and reading EDM4HEP data.
"""

import h5py
import pandas as pd
from typing import List, Dict, Any, Optional
import numpy as np

def build_hdf5_dataset(
    df: pd.DataFrame,
    output_file: str,
    compression: str = "gzip",
    compression_opts: int = 9
) -> None:
    """
    Build HDF5 file with event/data hierarchy.
    
    Structure:
    /events/
        /event_0/
            /data    # Dataset containing properties
        /event_1/
            ...
            
    Args:
        df: DataFrame containing event data with event_id column
        output_file: Path to output HDF5 file
        compression: Compression algorithm to use
        compression_opts: Compression options/level
    """
    with h5py.File(output_file, 'a') as f:
        # Create events group if it doesn't exist
        if 'events' not in f:
            events_group = f.create_group('events')
        else:
            events_group = f['events']
            
        # Group DataFrame by event_id
        for event_id, event_df in df.groupby('event_id'):
            # Create event group
            event_group = events_group.create_group(f'event_{event_id}')
            
            # Store data
            event_group.create_dataset(
                'data',
                data=event_df.drop(columns=['event_id']).to_records(index=False),
                compression=compression,
                compression_opts=compression_opts
            )

def read_event_data(
    input_file: str,
    event_id: int,
    group_name: str = "data"
) -> pd.DataFrame:
    """
    Read data for a specific event from HDF5 file.
    
    Args:
        input_file: Path to HDF5 file
        event_id: Event ID to read
        group_name: Name of the dataset group containing the data
        
    Returns:
        DataFrame containing event data
    """
    with h5py.File(input_file, 'r') as f:
        event_group = f[f'events/event_{event_id}']
        
        # Read data
        data = event_group[group_name][:]
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['event_id'] = event_id
        return df

def read_events_data(
    input_files: List[str],
    event_ids: List[int],
    group_name: str = "data"
) -> pd.DataFrame:
    """
    Read multiple events from HDF5 files.
    
    Args:
        input_files: List of HDF5 file paths
        event_ids: List of event IDs to read
        group_name: Name of the dataset group containing the data
        
    Returns:
        DataFrame containing all requested events
    """
    events_data = []
    
    for event_id in event_ids:
        for file_path in input_files:
            try:
                event_df = read_event_data(file_path, event_id, group_name)
                events_data.append(event_df)
                break
            except KeyError:
                continue
                
        if not events_data or events_data[-1]['event_id'].iloc[0] != event_id:
            print(f"Could not find event {event_id} in any input file")
    
    return pd.concat(events_data, ignore_index=True) if events_data else pd.DataFrame() 