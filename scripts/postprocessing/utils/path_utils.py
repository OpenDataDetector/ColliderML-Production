"""
Common utilities for EDM4HEP to HDF5 conversion and dataset management.
"""

import os
import glob
from pathlib import Path
from typing import List, Tuple

def get_run_paths(base_dir: str | Path) -> List[Path]:
    """
    Get paths to all run directories.
    
    Args:
        base_dir: Base directory containing run directories
        
    Returns:
        List of paths to run directories, sorted numerically by directory name
    """
    base_dir = Path(base_dir)
    run_dirs = [d for d in (base_dir / "runs").glob("*") if d.is_dir()]
    run_dirs.sort(key=lambda x: int(x.name))
    return run_dirs

def make_dir(base_dir: str | Path, *path_parts: str) -> Path:
    """
    Ensure output directory exists and return full path.
    
    Args:
        base_dir: Base directory for outputs
        *path_parts: Variable number of path components to concatenate
        
    Returns:
        Full path to output directory
    """
    output_dir = Path(base_dir)
    for part in path_parts:
        output_dir = output_dir / part
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def get_chunk_info(num_runs: int, run_size: int, chunk_size: int) -> Tuple[int, int, int]:
    """
    Calculate chunk-related information for dataset processing.
    
    Args:
        num_runs: Total number of runs
        run_size: Number of events per run
        chunk_size: Target size of each chunk in events
        
    Returns:
        Tuple of (num_events, runs_per_chunk, num_chunks)
    """
    num_events = num_runs * run_size
    runs_per_chunk = chunk_size // run_size
    num_chunks = (num_runs + runs_per_chunk - 1) // runs_per_chunk
    
    return num_events, runs_per_chunk, num_chunks

def get_event_file_path(base_dir: str, event_id: int, chunk_size: int, dataset_name: str) -> str:
    """
    Get the HDF5 file path containing a specific event.
    
    Args:
        base_dir: Base directory containing the HDF5 files
        event_id: ID of the event to locate
        chunk_size: Size of chunks used in file organization
        dataset_name: Name of the dataset
        
    Returns:
        Path to the HDF5 file containing the event
    """
    chunk_num = event_id // chunk_size
    start_event = chunk_num * chunk_size
    
    pattern = f"{base_dir}/{dataset_name}.events{start_event}-*.h5"
    matching_files = glob.glob(pattern)
    
    if not matching_files:
        raise FileNotFoundError(f"Could not find file containing event {event_id}")
        
    return matching_files[0] 