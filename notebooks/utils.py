import numpy as np
import pandas as pd
import awkward as ak
import uproot
from pathlib import Path
from tqdm import tqdm


def load_root_file(file_path, event_offset=0, event_id=None):
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
        df = ak.to_dataframe(data)
        
        # Apply event offset
        if event_offset:
            df['event_id'] += event_offset
            
        # Filter for specific event if requested
        if event_id is not None:
            df = df[df['event_id'] == event_id]
            if len(df) == 0:
                return None
                
        return df
    
    except (FileNotFoundError, uproot.exceptions.KeyInFileError) as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

def load_single_process(base_dir, proc_num, events_per_process):
    """Load data from a single process directory"""
    proc_dir = base_dir / f"proc_{proc_num}"
    event_offset = proc_num * events_per_process
    
    # Load each file type
    hits_df = load_root_file(proc_dir / "hits.root", event_offset)
    particles_df = load_root_file(proc_dir / "particles_simulation.root", event_offset)
    pythia_df = load_root_file(proc_dir / "pythia8_particles.root", event_offset)
    
    return hits_df, particles_df, pythia_df

def load_config_data(base_dir, config_name, num_processes=32, events_per_process=3):
    """Load and combine data from all processes for a given config"""
    base_dir = Path(base_dir)
    config_dir = base_dir / f"odd_output_{config_name}"
    
    all_hits = []
    all_particles = []
    all_pythia = []
    for proc in tqdm(range(num_processes)):
        hits_df, particles_df, pythia_df = load_single_process(config_dir, proc, events_per_process)
        if hits_df is not None:
            all_hits.append(hits_df)
        if particles_df is not None:
            all_particles.append(particles_df)
        if pythia_df is not None:
            all_pythia.append(pythia_df)
    
    # Combine all dataframes
    if all_hits and all_particles:
        combined_hits = pd.concat(all_hits, ignore_index=True)
        combined_particles = pd.concat(all_particles, ignore_index=True)
        combined_pythia = pd.concat(all_pythia, ignore_index=True)
        return combined_hits, combined_particles, combined_pythia
    else:
        print(f"No valid data found for config {config_name}")
        return None, None, None

def get_hist_data(data, bins, event_ids):
    """Calculate histogram data normalized by number of events
    
    Parameters:
    -----------
    data : array-like
        The data to histogram
    bins : array-like
        The bin edges
    event_ids : array-like
        The event IDs to count unique events
        
    Returns:
    --------
    bin_centers : array
        Centers of bins
    counts : array
        Normalized counts (per event)
    bin_widths : array
        Half-width of bins
    errors : array
        Normalized errors (per event)
    """
    # Get number of unique events
    n_events = len(np.unique(event_ids))
    
    # Calculate histogram
    counts, bin_edges = np.histogram(data, bins=bins)
    
    # Normalize counts and errors by number of events
    counts = counts / n_events
    errors = np.sqrt(counts / n_events)  # Scale Poisson errors by n_events
    
    # Calculate bin centers and widths
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = (bin_edges[1:] - bin_edges[:-1]) / 2
    
    return bin_centers, counts, bin_widths, errors