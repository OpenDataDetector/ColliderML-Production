"""
Track-specific utility functions for data processing.
"""

import numpy as np
import pandas as pd
import uproot
import awkward as ak
from typing import Dict, Any, List, Tuple

def get_particle_ids_from_events(events, tracker_readouts):
    """Get particle IDs from events for each tracker readout.
    
    Args:
        events: uproot events object
        tracker_readouts: list of tracker readout names
        
    Returns:
        DataFrame containing event_id and particle_id columns
    """
    all_particle_ids = []
    for det in tracker_readouts:
        if det not in events:
            continue
        hits = events[det].arrays()
        hits_df = ak.to_dataframe(hits[[f"{det}.position.x", f"{det}.position.y", f"{det}.position.z"]]).reset_index(drop=False).rename(columns={"entry": "event_id", f"{det}.position.x": "x", f"{det}.position.y": "y", f"{det}.position.z": "z"}).drop(columns=["subentry"])
        particle_links = events[f"_{det}_MCParticle"].arrays()
        particle_links_df = ak.to_dataframe(particle_links[f"_{det}_MCParticle.index"]) \
            .reset_index(drop=False) \
            .rename(columns={"entry": "event_id", "values": "particle_id"}) \
            .drop(columns=["subentry"])
        # horizontal concat
        hits_df = pd.concat([hits_df, particle_links_df[["particle_id"]]], axis=1)
        all_particle_ids.append(hits_df)

    particle_ids = pd.concat(all_particle_ids, ignore_index=True)
    return particle_ids

def convert_hit_ids(hit_ids_str: str) -> np.ndarray:
    """
    Convert string representation of hit IDs '[1,2,3,...]' to numpy array.
    
    Args:
        hit_ids_str: String representation of hit IDs array
        
    Returns:
        numpy array of integers
    """
    hit_ids = hit_ids_str.strip('[]').split(',')
    result = np.array([int(x) for x in hit_ids if x.strip()], dtype=np.int32)
    return result

# def load_root_file(file_path: str, tree_name: str = None) -> pd.DataFrame:
#     """
#     Load ROOT file and convert to DataFrame.
    
#     Args:
#         file_path: Path to ROOT file
#         tree_name: Name of tree to load. If None, uses first tree found.
        
#     Returns:
#         DataFrame containing ROOT data
#     """
#     root_file = uproot.open(file_path)
    
#     if tree_name is None:
#         # Get the keys and sort them by cycle number
#         keys = root_file.keys()
#         cycles = [int(key.split(';')[1]) for key in keys]
#         tree_name = keys[cycles.index(max(cycles))]
    
#     # Get arrays from tree
#     arrays = root_file[tree_name].arrays()
    
#     # Convert to DataFrame
#     df = ak.to_dataframe(arrays)
    
#     if isinstance(df.index, pd.MultiIndex):
#         df = df.reset_index()
#         # Remove entry/subentry columns if they exist
#         drop_cols = [col for col in ['entry', 'subentry'] if col in df.columns]
#         if drop_cols:
#             df = df.drop(columns=drop_cols)
    
#     return df

def load_root_file(file_path, event_offset=0, event_id=None, ignore_variable_columns=True):
    """Load data from a single root file with optional event filtering
    
    Parameters:
    -----------
    file_path : Path or str
        Path to the root file
    event_offset : int
        Offset to add to event_id (for backwards compatibility)
    event_id : int, optional
        If provided, only load this specific event
    
    Returns:
    --------
    pd.DataFrame or None
        DataFrame containing the loaded data, or None if loading fails
    """
    try:
        tree = uproot.open(file_path)
        # Get the keys and sort them by cycle number
        keys = tree.keys()
        cycles = [int(key.split(';')[1]) for key in keys]
        latest_key = keys[cycles.index(max(cycles))]
        data = tree[latest_key].arrays()

        # Separate regular and variable length columns
        if ignore_variable_columns:
            regular_columns = []
            variable_columns = []
            for field in data.fields:
                if 'var' in str(data[field].type):
                    variable_columns.append(field)
                else:
                    regular_columns.append(field)            
            # Warn about dropped columns
            if variable_columns:
                print(f"Warning: Dropping variable length columns: {', '.join(variable_columns)}")
        else:
            regular_columns = data.fields

        # Convert to dataframe using only regular columns
        df = ak.to_dataframe(data[regular_columns])
            
        # Apply event offset
        if event_offset and 'event_id' in df.columns:
            df['event_id'] += event_offset
            
        # Filter for specific event if requested
        if event_id is not None:
            if 'event_id' in df.columns:
                df = df[df['event_id'] == event_id]
            elif 'event_nr' in df.columns:
                df = df[df['event_nr'] == event_id]
            if len(df) == 0:
                return None
                
        return df
    
    except (FileNotFoundError, uproot.exceptions.KeyInFileError) as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None


def load_track_summary(file_path: str) -> Dict[str, Any]:
    """
    Load track summary data from ROOT file.
    
    Args:
        file_path: Path to track summary ROOT file
        
    Returns:
        Dictionary containing track arrays
    """
    tracksummary_root = uproot.open(file_path)
    arrays = tracksummary_root["tracksummary"].arrays()
    return arrays

def process_track_summary(arrays: Any, event_num: int) -> pd.DataFrame:
    """
    Process track summary arrays into DataFrame.
    
    Args:
        arrays: Track summary arrays from ROOT file
        event_num: Event number to process
        
    Returns:
        DataFrame containing track summary data
    """
    track_data = {}
    if not hasattr(arrays[event_num], 'fields'):
        return pd.DataFrame()
        
    for field in arrays[event_num].fields:
        if field == 'event_nr':
            continue
        try:
            array = ak.to_numpy(arrays[event_num][field])
            assert len(array.shape) == 1
            track_data[field] = array
        except Exception as e:
            print(f"Failed to process field {field}: {str(e)}")
            continue
            
    df = pd.DataFrame(track_data).rename(columns={"track_nr": "track_id"})
    return df

def analyze_coordinate_matches(comparison_df, tolerance=1e-3):
    """
    Analyze how well coordinates match between two sets of points in comparison_df.
    
    Args:
        comparison_df: DataFrame containing x,y,z and tx,ty,tz columns to compare
        tolerance: Absolute tolerance for np.isclose comparison
        
    Returns:
        dict: Dictionary containing match statistics and example mismatches
    """
    # Check if values are close
    is_close = np.isclose(
        comparison_df[["x", "y", "z"]],
        comparison_df[["tx", "ty", "tz"]],
        atol=tolerance
    )

    # Calculate matches per coordinate
    x_matches = (is_close[:, 0].sum() / len(comparison_df)) * 100
    y_matches = (is_close[:, 1].sum() / len(comparison_df)) * 100
    z_matches = (is_close[:, 2].sum() / len(comparison_df)) * 100

    total_matches = np.mean([x_matches, y_matches, z_matches])
    assert total_matches > 99.9, f"Simhits to edm4hep hit match below threshold. Matching percentage: {total_matches:.2f}%"

def create_particle_barcode_map(
    edm4hep_hits_df: pd.DataFrame,
    simhits_df: pd.DataFrame
) -> Dict[int, int]:
    """
    Create a mapping between particle barcodes and particle IDs by matching hit coordinates.
    
    Args:
        edm4hep_hits_df: DataFrame containing edm4hep hits with x,y,z coordinates
        simhits_df: DataFrame containing ROOT simhits with tx,ty,tz coordinates
        
    Returns:
        DataFrame: Mapping between particle_barcode and particle_id
    """
    # Convert coordinate columns to float32 using loc
    coord_cols = ["x", "y", "z"]
    edm4hep_hits_df.loc[:, coord_cols] = edm4hep_hits_df[coord_cols].astype(np.float32)
    
    sim_cols = ["tx", "ty", "tz"]
    simhits_df.loc[:, sim_cols] = simhits_df[sim_cols].astype(np.float32)

    # Reset indices and sort both DataFrames
    edm4hep_hits_df_sorted = edm4hep_hits_df.reset_index(drop=True).sort_values(by=coord_cols)
    simhits_df_sorted = simhits_df.reset_index(drop=True).sort_values(by=sim_cols)

    # Rename particle_id to particle_barcode
    edm4hep_hits_df_sorted = edm4hep_hits_df_sorted.rename(columns={"particle_id": "particle_barcode"})

    # Now concatenate
    comparison_df = pd.concat(
        [
            edm4hep_hits_df_sorted[["x", "y", "z", "particle_barcode"]].reset_index(drop=True),
            simhits_df_sorted[["tx", "ty", "tz", "particle_id"]].reset_index(drop=True)
        ],
        axis=1
    )

    # Drop duplicates
    comparison_df = comparison_df.drop_duplicates(subset=["x", "y", "z", "tx", "ty", "tz"], keep="first")

    analyze_coordinate_matches(comparison_df)

    particle_barcode_map = comparison_df[["particle_barcode", "particle_id"]].drop_duplicates(subset=["particle_id"]).drop_duplicates(subset=["particle_barcode"])
    particle_barcode_map = particle_barcode_map.set_index('particle_id')['particle_barcode']
    return particle_barcode_map

def get_majority_particle_id(hit_ids, simhits_root_df, particle_barcode_map):
    """Calculate the majority particle ID for a given track.
    
    Args:
        hit_ids: List of hit IDs for the track
        simhits_root_df: DataFrame containing simulated hits
        particle_barcode_map: Mapping between particle IDs and barcodes
        
    Returns:
        The most common particle barcode for hits in this track
    """
    # Get the hits for this track
    track_hits = simhits_root_df.iloc[convert_hit_ids(hit_ids)]
    
    # Map particle IDs to barcodes and get the most common one
    try:
        return track_hits.particle_id.map(particle_barcode_map).mode()[0]
    except:
        raise