import h5py
import pandas as pd
from typing import List

def read_event_tracks(
    input_file: str,
    event_id: int
) -> pd.DataFrame:
    """
    Read a single event from an HDF5 file.
    
    Args:
        input_file: Path to HDF5 file
        event_id: Event ID to read
        
    Returns:
        DataFrame containing track data for the event
    """
    with h5py.File(input_file, 'r') as f:
        event_group = f[f'events/event_{event_id}']
        
        # Read track data and hit arrays
        tracks = event_group['tracks'][:]
        hit_ids = event_group['hit_ids'][:]
        
        # Convert structured array to DataFrame
        df = pd.DataFrame(tracks)
        df['hit_ids'] = hit_ids
        df['event_id'] = event_id
        return df

def read_chunk_tracks(
    input_file: str
) -> pd.DataFrame:
    """
    Read all events from an HDF5 file.
    
    Args:
        input_file: Path to HDF5 file
        
    Returns:
        DataFrame containing track data for all events
    """
    events_data = []
    
    with h5py.File(input_file, 'r') as f:
        events_group = f['events']
        
        # Read each event
        for event_name in events_group:
            event_id = int(event_name.split('_')[1])
            event_df = read_event_tracks(input_file, event_id)
            events_data.append(event_df)
    
    return pd.concat(events_data, ignore_index=True)

def read_events_tracks(
    base_dir: str,
    event_ids: List[int],
    dataset_name: str = "mu10.ttbar"
) -> pd.DataFrame:
    """
    Read specific events from HDF5 files.
    
    Args:
        base_dir: Directory containing HDF5 files
        event_ids: List of event IDs to read
        dataset_name: Name of the dataset
        
    Returns:
        DataFrame containing track data for requested events
    """
    events_data = []
    chunk_size = 1000  # Must match the chunk size used in writing
    
    for event_id in event_ids:
        # Calculate which file contains this event
        chunk_num = event_id // chunk_size
        start_event = chunk_num * chunk_size
        
        # Find the file starting with this event using glob
        import glob
        pattern = f"{base_dir}/{dataset_name}.tracks.events{start_event}-*.h5"
        matching_files = glob.glob(pattern)
        
        if not matching_files:
            print(f"Could not find file containing event {event_id}")
            continue
            
        # Use the first (and should be only) matching file
        filename = matching_files[0]
        
        try:
            event_df = read_event_tracks(filename, event_id)
            events_data.append(event_df)
        except KeyError as e:
            print(f"Could not read event {event_id}: {e}")
            continue
    
    return pd.concat(events_data, ignore_index=True) if events_data else pd.DataFrame()