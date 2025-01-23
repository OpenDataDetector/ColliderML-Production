import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

def evaluate_clustering(df, label_column='labels_pred', energy_column='energy', 
                        particle_id_column='particle_id', additional_features=None,
                        energy_weighted=False):
    """
    Evaluate clustering performance by comparing cluster assignments to true particle IDs
    of highest energy cells.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the clustering data
    label_column : str, default='labels_pred'
        Name of column containing cluster labels
    energy_column : str, default='energy'
        Name of column containing energy values
    particle_id_column : str, default='particle_id'
        Name of column containing true particle IDs
    additional_features : list of str, optional
        Additional features to track from highest energy cell
    energy_weighted : bool, default=False
        If True, weight purity/efficiency calculations by cell energy
        
    Returns
    -------
    metrics_df : pandas.DataFrame
        DataFrame containing cluster metrics
    overall_metrics : dict
        Dictionary containing overall purity and efficiency
    """
    
    # ---------------------------------------------------------
    # Define features to aggregate
    # ---------------------------------------------------------
    agg_dict = {
        energy_column: 'first',          # The highest-energy cell's energy in each cluster
        particle_id_column: 'first',     # The highest-energy cell's particle ID in each cluster
        label_column: 'count'            # Total cells in each cluster
    }
    
    # Add any additional features
    if additional_features:
        for feature in additional_features:
            agg_dict[feature] = 'first'
    
    
    # ---------------------------------------------------------
    # Find highest-energy cell and cluster-level "dominant" particle ID
    # ---------------------------------------------------------
    cluster_info = (
        df
        .sort_values(energy_column, ascending=False)
        .groupby(label_column)
        .agg(agg_dict)
        .rename(columns={
            label_column: 'total_cells',
            particle_id_column: 'dominant_particle_id'  # separate name for the cluster-level ID
        })
    )
    # ---------------------------------------------------------
    # Merge the "dominant_particle_id" back to each row
    # ---------------------------------------------------------
    df_with_true = df.merge(
        cluster_info[['dominant_particle_id']], 
        left_on=label_column,
        right_index=True
    )
    
    # ---------------------------------------------------------
    # Purity and efficiency calculations
    # ---------------------------------------------------------
    if energy_weighted:
        # Energy-weighted: compare sum of energies in which cell's original ID matches the cluster's ID
        correct_mask = df_with_true[particle_id_column] == df_with_true['dominant_particle_id']
        
        # Group by cluster and calculate total energy
        cluster_metrics = df_with_true.groupby(label_column).agg({
            energy_column: 'sum',            # total energy in cluster
            'dominant_particle_id': 'first'  # the single dominant ID we're referencing
        })
        
        # Correct energy is the sum of energies where cell's ID = dominant ID
        cluster_metrics['correct_energy'] = (
            df_with_true[correct_mask].groupby(label_column)[energy_column].sum()
        )
        
        # Purity = ratio of correct_energy to total
        cluster_metrics['purity'] = (
            cluster_metrics['correct_energy'] / cluster_metrics[energy_column]
        )
        
        # Efficiency: how much of that "dominant" particle's total energy was actually captured
        particle_total_energy = df_with_true.groupby(particle_id_column)[energy_column].sum()
        # Map the cluster's dominant ID back to total energies
        cluster_metrics['total_particle_energy'] = cluster_metrics['dominant_particle_id'].map(particle_total_energy)
        cluster_metrics['efficiency'] = (
            cluster_metrics['correct_energy'] / cluster_metrics['total_particle_energy']
        )
        
        # Calculate overall metrics
        overall_purity = (cluster_metrics.purity * cluster_metrics.energy).sum() / cluster_metrics.energy.sum()
        overall_efficiency = (cluster_metrics.efficiency * cluster_metrics.energy).sum() / cluster_metrics.energy.sum()
        
    else:
        # Count-based approach: number of correct cells in each cluster vs total cells
        correct_mask = df_with_true[particle_id_column] == df_with_true['dominant_particle_id']
        correct_counts = correct_mask.groupby(df_with_true[label_column]).sum().rename('correct_cells')
        
        # cluster_info has total_cells and dominant_particle_id
        cluster_metrics = pd.concat([cluster_info, correct_counts], axis=1)
        
        # Purity = fraction of cells in the cluster that match the dominant ID
        cluster_metrics['purity'] = (
            cluster_metrics['correct_cells'] / cluster_metrics['total_cells']
        )
        
        # Efficiency = fraction of the total "dominant" particle's cells that appear in the cluster
        # We look up how many cells in the entire dataframe belong to that "dominant_particle_id"
        particle_total_cells = df_with_true.groupby(particle_id_column).size()
        cluster_metrics['total_particle_cells'] = cluster_metrics['dominant_particle_id'].map(particle_total_cells)
        cluster_metrics['efficiency'] = (
            cluster_metrics['correct_cells'] / cluster_metrics['total_particle_cells']
        )
        # Calculate overall metrics
        overall_purity = (cluster_metrics.purity * cluster_metrics.total_cells).sum() / cluster_metrics.total_cells.sum()
        overall_efficiency = (cluster_metrics.efficiency * cluster_metrics.total_cells).sum() / cluster_metrics.total_cells.sum()
        
    overall_metrics = {
        'purity': overall_purity,
        'efficiency': overall_efficiency,
        'energy_weighted': energy_weighted
    }
    
    return cluster_metrics, overall_metrics

def plot_clustering_metrics(metrics_df, overall_metrics):
    """
    Plot distribution of cluster purities and efficiencies.
    
    Parameters
    ----------
    metrics_df : pandas.DataFrame
        DataFrame containing cluster metrics
    overall_metrics : dict
        Dictionary containing overall metrics
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot purity distribution
    ax1.hist(metrics_df['purity'].dropna(), bins=50, range=(0, 1))
    ax1.set_xlabel('Cluster Purity')
    ax1.set_ylabel('Number of Clusters')
    ax1.set_title(
        f'Distribution of Cluster Purities\nOverall Purity: {overall_metrics["purity"]:.3f}'
    )
    
    # Plot efficiency distribution
    ax2.hist(metrics_df['efficiency'].dropna(), bins=50, range=(0, 1))
    ax2.set_xlabel('Cluster Efficiency')
    ax2.set_ylabel('Number of Clusters')
    ax2.set_title(
        f'Distribution of Cluster Efficiencies\nOverall Efficiency: {overall_metrics["efficiency"]:.3f}'
    )
    
    weight_type = "Energy-weighted" if overall_metrics["energy_weighted"] else "Count-based"
    plt.suptitle(f'{weight_type} Clustering Metrics')
    plt.tight_layout()
    plt.show()