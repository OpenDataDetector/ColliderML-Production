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
    print(f"\nConverting hit IDs from string: {hit_ids_str}")
    # Remove brackets and split by comma
    hit_ids = hit_ids_str.strip('[]').split(',')
    # Convert to integers
    result = np.array([int(x) for x in hit_ids if x.strip()], dtype=np.int32)
    print(f"Converted to array: {result}")
    return result

def load_root_file(file_path: str, tree_name: str = None) -> pd.DataFrame:
    """
    Load ROOT file and convert to DataFrame.
    
    Args:
        file_path: Path to ROOT file
        tree_name: Name of tree to load. If None, uses first tree found.
        
    Returns:
        DataFrame containing ROOT data
    """
    print(f"\nLoading ROOT file: {file_path}")
    root_file = uproot.open(file_path)
    print(f"Available keys: {root_file.keys()}")
    
    if tree_name is None:
        # Get the keys and sort them by cycle number
        keys = root_file.keys()
        cycles = [int(key.split(';')[1]) for key in keys]
        tree_name = keys[cycles.index(max(cycles))]
        print(f"Selected tree: {tree_name}")
    
    # Get arrays from tree
    arrays = root_file[tree_name].arrays()
    print(f"Array fields: {arrays.fields if hasattr(arrays, 'fields') else 'No fields'}")
    
    # Convert to DataFrame
    df = ak.to_dataframe(arrays)
    print(f"Initial DataFrame shape: {df.shape}")
    
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
        # Remove entry/subentry columns if they exist
        drop_cols = [col for col in ['entry', 'subentry'] if col in df.columns]
        if drop_cols:
            print(f"Dropping columns: {drop_cols}")
            df = df.drop(columns=drop_cols)
    
    print(f"Final DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {df.columns.tolist()}")
    return df

def load_track_summary(file_path: str) -> Dict[str, Any]:
    """
    Load track summary data from ROOT file.
    
    Args:
        file_path: Path to track summary ROOT file
        
    Returns:
        Dictionary containing track arrays
    """
    print(f"\nLoading track summary from: {file_path}")
    tracksummary_root = uproot.open(file_path)
    print(f"Available keys: {tracksummary_root.keys()}")
    arrays = tracksummary_root["tracksummary"].arrays()
    print(f"Array fields: {arrays.fields if hasattr(arrays, 'fields') else 'No fields'}")
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
    print(f"\nProcessing track summary for event {event_num}")
    track_data = {}
    if not hasattr(arrays[event_num], 'fields'):
        print(f"WARNING: No fields found in arrays[{event_num}]")
        return pd.DataFrame()
        
    print(f"Available fields: {arrays[event_num].fields}")
    for field in arrays[event_num].fields:
        if field == 'event_nr':
            continue
        try:
            array = ak.to_numpy(arrays[event_num][field])
            assert len(array.shape) == 1
            track_data[field] = array
            print(f"Processed field {field}: shape {array.shape}")
        except Exception as e:
            print(f"Failed to process field {field}: {str(e)}")
            continue
            
    df = pd.DataFrame(track_data).rename(columns={"track_nr": "track_id"})
    print(f"Created DataFrame with shape: {df.shape}")
    print(f"DataFrame columns: {df.columns.tolist()}")
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
    # First set dtype of x,y,z and tx,ty,tz to float32
    edm4hep_hits_df[["x", "y", "z"]] = edm4hep_hits_df[["x", "y", "z"]].astype(np.float32)
    simhits_df[["tx", "ty", "tz"]] = simhits_df[["tx", "ty", "tz"]].astype(np.float32)

    # Reset indices and sort both DataFrames
    edm4hep_hits_df_sorted = edm4hep_hits_df.reset_index(drop=True).sort_values(by=["x", "y", "z"])
    simhits_df_sorted = simhits_df.reset_index(drop=True).sort_values(by=["tx", "ty", "tz"])

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
        print(f"No particle ID found for track {hit_ids}")
        print(track_hits)
        print(particle_barcode_map)
        print(track_hits.particle_id.map(particle_barcode_map))
        print(track_hits.particle_id.map(particle_barcode_map).mode())
        raise