import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from dataclasses import dataclass
import sklearn.neighbors as spatial
from typing import List, Tuple, Optional


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

@dataclass
class TopoClusterConfig:
    """Configuration for topological clustering"""
    seed_threshold: float = 4.0        # High threshold (4σ)
    neighbor_threshold: float = 2.0     # Medium threshold (2σ)
    final_threshold: float = 0.0       # Low threshold (0σ)
    
class TopoClustering:
    def __init__(self, config: TopoClusterConfig):
        self.config = config
        
    def cluster(self, hits_df: pd.DataFrame) -> pd.DataFrame:
        """
        Main clustering function
        
        Args:
            hits_df: DataFrame with columns:
                - x, y, z: position
                - energy: cell energy
                - noise: cell noise level
                - layer: detector layer
        Returns:
            DataFrame with additional columns:
                - cluster_id: assigned cluster ID
                - is_seed: boolean if cell is seed
                - is_shared: boolean if cell is shared between clusters
                - weight: weight for shared cells
        """
        # Calculate signal/noise for all cells
        hits_df['s2n'] = hits_df['energy'] / hits_df['noise']
        
        # 1. Find seed cells (vectorized)
        seed_mask = hits_df['s2n'] > self.config.seed_threshold
        seeds_df = hits_df[seed_mask].copy()
        seeds_df['cluster_id'] = np.arange(len(seeds_df))
        
        # 2. Build KD-tree for efficient neighbor finding
        tree = spatial.KDTree(hits_df[['x', 'y', 'z']].values)
        
        # 3. Grow clusters (still need to iterate but with vectorized operations)
        clusters = self._grow_clusters(hits_df, seeds_df, tree)
        
        # 4. Find local maxima (vectorized)
        local_maxima = self._find_local_maxima(clusters)
        
        # 5. Split clusters (partially vectorized)
        final_clusters = self._split_clusters(clusters, local_maxima)
        
        return final_clusters
    
    def _grow_clusters(self, 
                      hits_df: pd.DataFrame, 
                      seeds_df: pd.DataFrame,
                      tree: spatial.KDTree) -> pd.DataFrame:
        """Grow clusters from seeds using vectorized operations"""
        # Initialize all cells as unassigned
        hits_df['cluster_id'] = -1
        hits_df.loc[seeds_df.index, 'cluster_id'] = seeds_df['cluster_id']
        
        # Find neighbors within radius (vectorized)
        radius = 1.0  # Adjust based on detector geometry
        neighbors = tree.query_ball_point(seeds_df[['x', 'y', 'z']].values, radius)
        
        # Grow clusters iteratively but with vectorized operations
        while True:
            # Find cells above neighbor threshold adjacent to existing clusters
            new_cells = self._find_new_cells(hits_df, tree)
            if len(new_cells) == 0:
                break
                
            # Add cells to clusters (vectorized)
            hits_df.loc[new_cells.index, 'cluster_id'] = new_cells['assigned_cluster']
            
        return hits_df
    
    def _find_local_maxima(self, clusters_df: pd.DataFrame) -> pd.DataFrame:
        """Find local maxima in energy within each cluster"""
        maxima = []
        
        # Group by cluster and find local maxima (vectorized per cluster)
        for cluster_id, cluster in clusters_df.groupby('cluster_id'):
            # Create 3D energy grid for this cluster
            pos = cluster[['x', 'y', 'z']].values
            energies = cluster['energy'].values
            
            # Find cells that are local maxima in 3D space
            is_maximum = np.ones(len(cluster), dtype=bool)
            tree = spatial.KDTree(pos)
            
            # Vectorized local maxima finding
            for i, point in enumerate(pos):
                neighbors = tree.query_ball_point(point, radius=1.0)
                if any(energies[neighbors] > energies[i]):
                    is_maximum[i] = False
                    
            maxima.append(cluster[is_maximum])
            
        return pd.concat(maxima)
    
    def _split_clusters(self, 
                       clusters_df: pd.DataFrame, 
                       maxima_df: pd.DataFrame) -> pd.DataFrame:
        """Split clusters with multiple maxima"""
        # Initialize output
        clusters_df['is_shared'] = False
        clusters_df['weight'] = 1.0
        
        # For each cluster with multiple maxima
        for cluster_id, cluster_maxima in maxima_df.groupby('cluster_id'):
            if len(cluster_maxima) > 1:
                # Get cluster cells
                cluster_mask = clusters_df['cluster_id'] == cluster_id
                cluster = clusters_df[cluster_mask]
                
                # Calculate distances to all maxima (vectorized)
                distances = spatial.distance_matrix(
                    cluster[['x', 'y', 'z']].values,
                    cluster_maxima[['x', 'y', 'z']].values
                )
                
                # Assign cells to nearest maximum
                nearest_max = np.argmin(distances, axis=1)
                
                # Find cells that should be shared (close to multiple maxima)
                dist_ratio = np.sort(distances, axis=1)[:, 0] / np.sort(distances, axis=1)[:, 1]
                shared_mask = dist_ratio > 0.7  # Adjustable threshold
                
                # Update cluster assignments
                new_cluster_ids = cluster_maxima.index[nearest_max]
                clusters_df.loc[cluster_mask, 'cluster_id'] = new_cluster_ids
                clusters_df.loc[cluster_mask & shared_mask, 'is_shared'] = True
                
                # Calculate weights for shared cells
                weights = 1.0 / distances[shared_mask]
                weights = weights / weights.sum(axis=1)[:, np.newaxis]
                clusters_df.loc[cluster_mask & shared_mask, 'weight'] = weights
                
        return clusters_df
