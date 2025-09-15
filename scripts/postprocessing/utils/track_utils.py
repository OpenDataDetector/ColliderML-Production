"""
Track-specific utility functions for data processing.
"""

import numpy as np
import pandas as pd
import uproot
import awkward as ak
import h5py
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

def load_root_file(file_path, event_offset=0, event_id=None, ignore_variable_columns=True, included_columns=None):
    """Load data from a single root file with optional event filtering
    
    Parameters:
    -----------
    file_path : Path or str
        Path to the root file
    event_offset : int
        Offset to add to event_id (for backwards compatibility)
    event_id : int, optional
        If provided, only load this specific event
    ignore_variable_columns : bool
        Whether to ignore variable length columns
    included_columns : list, optional
        If provided, only load these specific columns
    
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
        
        # Determine filtering parameters
        filter_params = {}
        
        # Column filtering
        if included_columns:
            filter_params['filter_name'] = included_columns
        elif ignore_variable_columns:
            def exclude_var_columns(branch_name):
                branch = tree[latest_key][branch_name]
                return 'var' not in str(branch.typename)
            filter_params['filter_name'] = exclude_var_columns
        
        # Event filtering - this is the key optimization!
        if event_id is not None:
            # Check if we have event_id or event_nr column to determine which to filter on
            available_branches = tree[latest_key].keys()
            if 'event_id' in available_branches:
                # Adjust for event_offset if needed
                target_event = event_id - event_offset if event_offset else event_id
                filter_params['cut'] = f"event_id == {target_event}"
            elif 'event_nr' in available_branches:
                filter_params['cut'] = f"event_nr == {event_id}"
            else:
                print("Warning: No event_id or event_nr column found for event filtering")
        
        # Load arrays with all filtering applied at once
        data = tree[latest_key].arrays(**filter_params)
        
        # Check if we got any data when filtering for specific event
        if event_id is not None and len(data) == 0:
            print(f"No data found for event_id {event_id}")
            return None

        # Convert to dataframe 
        df = ak.to_dataframe(data)
            
        # Apply event offset (only if we're not filtering to specific event)
        if event_offset and event_id is None and 'event_id' in df.columns:
            df['event_id'] += event_offset
                
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


def build_hdf5_tracks(df: pd.DataFrame, output_file: str) -> None:
    """
    Build HDF5 file with event/track/hit hierarchy.
    Uses CSR ragged encoding for hit ids: hit_ids_data (int32) and hit_ids_indptr (int64).
    """
    if df is None or df.empty:
        return
    with h5py.File(output_file, 'a') as f:
        events_group = f.create_group('events') if 'events' not in f else f['events']
        for event_id, event_df in df.groupby('event_id'):
            event_group_name = f'event_{event_id}'
            if event_group_name in events_group:
                del events_group[event_group_name]
            event_group = events_group.create_group(event_group_name)

            # Prepare CSR components from hit_ids lists
            hit_lists = (
                event_df['hit_ids'].tolist() if 'hit_ids' in event_df.columns else []
            )
            lengths = np.fromiter(
                (len(a) if a is not None else 0 for a in hit_lists),
                dtype=np.int64,
                count=len(hit_lists)
            ) if hit_lists else np.array([], dtype=np.int64)
            indptr = np.empty(len(lengths) + 1, dtype=np.int64)
            indptr[0] = 0
            if len(lengths) > 0:
                np.cumsum(lengths, out=indptr[1:])
            data = (
                np.concatenate([
                    np.asarray(a, dtype=np.int32) for a in hit_lists if a is not None and len(a) > 0
                ]) if indptr[-1] > 0 else np.array([], dtype=np.int32)
            )

            # Store track-level data (exclude hit_ids and event_id)
            track_data = event_df.drop(columns=['hit_ids', 'event_id'], errors='ignore')
            tracks_ds = event_group.create_dataset(
                'tracks',
                data=track_data.to_records(index=False),
                compression='gzip',
                compression_opts=6,
                shuffle=True,
            )

            # Store CSR arrays for hit ids
            data_chunk = (min(max(1, data.size), 65536),) if data.size > 0 else (1,)
            ptr_chunk = (min(max(1, indptr.size), 65536),)
            event_group.create_dataset(
                'hit_ids_data',
                data=data,
                dtype='int32',
                compression='gzip',
                compression_opts=6,
                shuffle=True,
                chunks=data_chunk,
            )
            event_group.create_dataset(
                'hit_ids_indptr',
                data=indptr,
                dtype='int64',
                compression='gzip',
                compression_opts=6,
                shuffle=True,
                chunks=ptr_chunk,
            )

            # Minimal metadata for readers
            event_group.attrs['encoding'] = 'csr_v1'
            event_group.attrs['nnz'] = int(data.size)


def write_tracks_with_selection(
    df: pd.DataFrame,
    output_file: str,
    columns_keep: List[str] | None = None,
) -> None:
    """
    Write tracks DataFrame to H5 with optional column selection.

    Ensures required columns for storage are present:
      - 'event_id' (for grouping)
      - 'hit_ids' (used to build CSR arrays hit_ids_data/indptr under /events/event_#)
    """
    if df is None or df.empty:
        return
    filtered = df
    if columns_keep:
        cols = [c for c in columns_keep if c in df.columns]
        # Ensure required columns are present
        if 'event_id' not in cols and 'event_id' in df.columns:
            cols = cols + ['event_id']
        if 'hit_ids' not in cols and 'hit_ids' in df.columns:
            cols = cols + ['hit_ids']
        filtered = df[cols].copy()
    build_hdf5_tracks(filtered, output_file)