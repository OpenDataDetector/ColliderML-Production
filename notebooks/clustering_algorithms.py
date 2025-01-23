import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

def heuristic_energy_clustering(cells_df, seed_fraction=0.1):
    """
    Perform heuristic clustering using high-energy cells as seeds.
    Uses KDTree for efficient nearest neighbor search.
    
    Args:
        cells_df (pd.DataFrame): DataFrame containing cell information with columns:
            - energy: Cell energy
            - x, y, z: Cell positions
        seed_fraction (float): Fraction of highest energy cells to use as seeds (default: 0.1)
    
    Returns:
        pd.DataFrame: Input DataFrame with additional column:
            - cluster_id: ID of the cluster each cell belongs to
    """
    # Sort cells by energy and identify seeds
    sorted_cells = cells_df.sort_values('energy', ascending=False)
    n_seeds = int(len(cells_df) * seed_fraction)
    seeds = sorted_cells.iloc[:n_seeds]
    remaining = sorted_cells.iloc[n_seeds:]
    
    print(f"Seeds: {len(seeds)}")
    print(f"Remaining: {len(remaining)}")

    # Get positions for distance calculation
    seed_positions = seeds[['x', 'y', 'z']].values
    remaining_positions = remaining[['x', 'y', 'z']].values
    
    print("Buliding KDTree")
    
    # Build KD-tree on seed positions
    tree = KDTree(seed_positions)
    
    print("Querying KDTree")

    # Find nearest seed for each remaining cell
    _, nearest_seed_idx = tree.query(remaining_positions, k=1)
    nearest_seed_idx = nearest_seed_idx.flatten()
    
    print("Creating cluster assignments")
    # Create cluster assignments
    cluster_ids = np.arange(len(seeds))
    cells_df = cells_df.copy()
    cells_df['cluster_id'] = -1
    
    # Assign cluster IDs
    cells_df.loc[seeds.index, 'cluster_id'] = cluster_ids
    cells_df.loc[remaining.index, 'cluster_id'] = nearest_seed_idx
    
    return cells_df
