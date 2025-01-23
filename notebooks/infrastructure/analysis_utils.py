import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

class ParticleChain:
    def __init__(self, particles_df, parents_df, daughters_df, tracker_df, hits_df):
        """Initialize with all relevant dataframes"""
        self.particles_df = particles_df
        self.parents_df = parents_df
        self.daughters_df = daughters_df
        self.tracker_df = tracker_df
        self.hits_df = hits_df
        
        # Cache for memoization
        self._parent_cache = {}
        self._daughter_cache = {}
        self._hit_cache = {}
    
    def get_parents(self, particle_id):
        """Get all parent IDs for a given particle"""
        if particle_id in self._parent_cache:
            return self._parent_cache[particle_id]
        
        try:
            particle = self.particles_df.loc[particle_id]
            parent_indices = range(int(particle.parents_begin), int(particle.parents_end))
            parents = [self.parents_df.iloc[idx]['particle_id'] for idx in parent_indices 
                      if self.parents_df.iloc[idx]['particle_id'] in self.particles_df.index]
        except (KeyError, IndexError):
            parents = []
            
        self._parent_cache[particle_id] = parents
        return parents
    
    def get_daughters(self, particle_id):
        """Get all daughter IDs for a given particle"""
        if particle_id in self._daughter_cache:
            return self._daughter_cache[particle_id]
        
        try:
            particle = self.particles_df.loc[particle_id]
            daughter_indices = range(int(particle.daughters_begin), int(particle.daughters_end))
            daughters = [self.daughters_df.iloc[idx]['particle_id'] for idx in daughter_indices 
                        if self.daughters_df.iloc[idx]['particle_id'] in self.particles_df.index]
        except (KeyError, IndexError):
            daughters = []
            
        self._daughter_cache[particle_id] = daughters
        return daughters
    
    def get_detector_hits(self, particle_id):
        """Get all detector hits for a particle"""
        if particle_id in self._hit_cache:
            return self._hit_cache[particle_id]
        
        try:
            tracker_hits = self.tracker_df[self.tracker_df.particle_id == particle_id]
            calo_hits = self.hits_df[self.hits_df.particle_id == particle_id]
        except KeyError:
            tracker_hits = pd.DataFrame()
            calo_hits = pd.DataFrame()
            
        hits = {
            'tracker': tracker_hits,
            'calo': calo_hits
        }
        
        self._hit_cache[particle_id] = hits
        return hits
    
    def is_final_visible(self, particle_id, detector_type=None):
        """
        Check if particle is a final visible particle.
        
        Args:
            particle_id: ID of particle to check
            detector_type: None (any hits), 'tracker', or 'calo'
        """
        # Get hits for this particle
        hits = self.get_detector_hits(particle_id)
        
        # Check if particle has relevant hits
        has_hits = False
        if detector_type is None:
            has_hits = len(hits['tracker']) > 0 or len(hits['calo']) > 0
        elif detector_type == 'tracker':
            has_hits = len(hits['tracker']) > 0
        elif detector_type == 'calo':
            has_hits = len(hits['calo']) > 0
            
        if not has_hits:
            return False
            
        # Check if any daughters have hits
        daughters = self.get_daughters(particle_id)
        for daughter_id in daughters:
            daughter_hits = self.get_detector_hits(daughter_id)
            if detector_type is None:
                if len(daughter_hits['tracker']) > 0 or len(daughter_hits['calo']) > 0:
                    return False
            elif detector_type == 'tracker':
                if len(daughter_hits['tracker']) > 0:
                    return False
            elif detector_type == 'calo':
                if len(daughter_hits['calo']) > 0:
                    return False
                    
        return True
        
    def get_final_visible_particles(self, detector_type=None):
        """Get all final visible particles efficiently using set operations"""
        # First get all particles that have hits in the relevant detectors
        if detector_type is None:
            visible_particles = set(self.hits_df.particle_id) | set(self.tracker_df.particle_id)
        elif detector_type == 'tracker':
            visible_particles = set(self.tracker_df.particle_id)
        elif detector_type == 'calo':
            visible_particles = set(self.hits_df.particle_id)
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
            
        # Get all daughters of visible particles
        visible_daughters = set()
        for pid in tqdm(visible_particles):
            try:
                particle = self.particles_df.loc[pid]
                daughter_indices = range(int(particle.daughters_begin), int(particle.daughters_end))
                daughters = {self.daughters_df.iloc[idx]['particle_id'] for idx in daughter_indices 
                           if self.daughters_df.iloc[idx]['particle_id'] in self.particles_df.index}
                visible_daughters.update(daughters & visible_particles)  # Only add daughters that are also visible
            except (KeyError, IndexError):
                continue
                
        # Final visible particles are those that have hits but no visible daughters
        final_visible = visible_particles - visible_daughters
        
        return list(final_visible)
        
    def count_ancestors_with_hits(self, particle_id, detector_type=None):
        """Count number of ancestors that have hits efficiently using sets"""
        # Get all ancestors first using sets for faster lookups
        ancestors = set()
        to_check = {particle_id}
        checked = set()
        
        # Build complete ancestor set first
        while to_check:
            current = to_check.pop()
            if current in checked:
                continue
            checked.add(current)
            
            parents = self.get_parents(current)
            ancestors.update(parents)
            to_check.update(parents)
        
        # Now check which ancestors have hits - use set operations for efficiency
        if detector_type is None:
            visible_ancestors = ancestors & (set(self.hits_df.particle_id) | set(self.tracker_df.particle_id))
        elif detector_type == 'tracker':
            visible_ancestors = ancestors & set(self.tracker_df.particle_id)
        elif detector_type == 'calo':
            visible_ancestors = ancestors & set(self.hits_df.particle_id)
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
            
        return len(visible_ancestors)

    def analyze_final_visible_particles(self):
        """Analyze final visible particles and their ancestors with hits"""
        # Get all particles with hits for comparison
        if not hasattr(self, '_visible_particles_cache'):
            self._visible_particles_cache = set(self.hits_df.particle_id) | set(self.tracker_df.particle_id)
            
        # Get final visible particles (reuse cached visible particles)
        final_visible = self.get_final_visible_particles()
        final_visible_calo = self.get_final_visible_particles('calo')
        final_visible_tracker = self.get_final_visible_particles('tracker')
        
        print("Summary:")
        print(f"Total particles with hits: {len(self._visible_particles_cache)}")
        print(f"Final visible particles (any detector): {len(final_visible)}")
        print(f"Final visible particles (calorimeter): {len(final_visible_calo)}")
        print(f"Final visible particles (tracker): {len(final_visible_tracker)}")
        
        # Analyze ancestors with hits
        print("Analyzing ancestors with hits")
        ancestors_any = [self.count_ancestors_with_hits(pid) for pid in tqdm(final_visible)]
        ancestors_calo = [self.count_ancestors_with_hits(pid, 'calo') for pid in tqdm(final_visible_calo)]
        ancestors_tracker = [self.count_ancestors_with_hits(pid, 'tracker') for pid in tqdm(final_visible_tracker)]
        
        # Plot distributions
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        ax1.hist(ancestors_any, bins=20)
        ax1.set_title("Ancestors with hits\n(Any detector)")
        ax1.set_xlabel("Number of ancestors")
        ax1.set_ylabel("Count")
        
        ax2.hist(ancestors_calo, bins=20)
        ax2.set_title("Ancestors with hits\n(Calorimeter)")
        ax2.set_xlabel("Number of ancestors")
        
        ax3.hist(ancestors_tracker, bins=20)
        ax3.set_title("Ancestors with hits\n(Tracker)")
        ax3.set_xlabel("Number of ancestors")
        
        plt.tight_layout()
        plt.show()


class DecayChainViz:
    def __init__(self, particle_chain):
        self.chain = particle_chain
        self.G = nx.DiGraph()
        self._depths = {}  # Store depths relative to starting particle
        
    def build_chain(self, start_particle_id, max_depth=3):
        """Build the decay chain graph for a particle and its ancestors/descendants"""
        if start_particle_id not in self.chain.particles_df.index:
            raise ValueError(f"Particle ID {start_particle_id} not found in dataset")
            
        self.G.clear()
        self._depths.clear()
        
        # Set starting particle depth to 0
        self._depths[start_particle_id] = 0
        
        # Get all ancestors and descendants with their relative depths
        ancestors = self._get_all_ancestors(start_particle_id, max_depth)
        descendants = self._get_all_descendants(start_particle_id, max_depth)
        
        # Add all relevant particles to graph
        all_particles = {start_particle_id} | ancestors | descendants
        for particle_id in all_particles:
            self._add_particle_node(particle_id)
            
        # Add edges for ancestors and descendants
        for particle_id in all_particles:
            for daughter_id in self.chain.get_daughters(particle_id):
                if daughter_id in all_particles:
                    self.G.add_edge(particle_id, daughter_id)
        
        # Prune nodes that don't have hits and don't lead to hits
        self._prune_non_detecting_branches()
        
    def _get_all_ancestors(self, particle_id, max_depth):
        """Get all ancestors up to max_depth generations back"""
        ancestors = set()
        current_generation = {particle_id}
        depth = -1  # Start at -1 for ancestors
        
        while current_generation and abs(depth) <= max_depth:
            next_generation = set()
            for pid in current_generation:
                for parent_id in self.chain.get_parents(pid):
                    next_generation.add(parent_id)
                    self._depths[parent_id] = depth
            ancestors.update(next_generation)
            current_generation = next_generation
            depth -= 1
            
        return ancestors
        
    def _get_all_descendants(self, particle_id, max_depth):
        """Get all descendants up to max_depth generations forward"""
        descendants = set()
        current_generation = {particle_id}
        depth = 1  # Start at 1 for descendants
        
        while current_generation and depth <= max_depth:
            next_generation = set()
            for pid in current_generation:
                for daughter_id in self.chain.get_daughters(pid):
                    next_generation.add(daughter_id)
                    self._depths[daughter_id] = depth
            descendants.update(next_generation)
            current_generation = next_generation
            depth += 1
            
        return descendants

    def _has_hits(self, particle_id):
        """Check if particle has any hits in tracker or calorimeter"""
        hits = self.chain.get_detector_hits(particle_id)
        return len(hits['tracker']) > 0 or len(hits['calo']) > 0
    
    def _has_detecting_descendants(self, node, visited=None):
        """Check if node or any of its descendants have detector hits"""
        if visited is None:
            visited = set()
            
        if node in visited:
            return False
        visited.add(node)
        
        # Check if this node has hits
        if self._has_hits(node):
            return True
            
        # Check all descendants
        for _, child in self.G.out_edges(node):
            if self._has_detecting_descendants(child, visited):
                return True
                
        return False
        
    def _prune_non_detecting_branches(self):
        """Remove nodes that don't have hits and don't lead to hits"""
        nodes_to_remove = []
        for node in self.G.nodes():
            if not self._has_detecting_descendants(node):
                nodes_to_remove.append(node)
                
        self.G.remove_nodes_from(nodes_to_remove)
    
    def _add_particle_node(self, particle_id):
        """Add a particle node with its properties"""
        try:
            particle = self.chain.particles_df.loc[particle_id]
            hits = self.chain.get_detector_hits(particle_id)
            
            # Node attributes
            self.G.add_node(particle_id, 
                           pdg=particle.PDG,
                           pt=particle.pt,
                           eta=particle.eta,
                           phi=particle.phi,
                           n_tracker_hits=len(hits['tracker']),
                           n_calo_hits=len(hits['calo']))
        except KeyError:
            pass  # Skip particles that don't exist in our dataset
    
    def plot(self, figsize=(12, 8)):
        """Plot the decay chain"""
        plt.figure(figsize=figsize)
        
        # Position nodes using hierarchical layout
        pos = nx.spring_layout(self.G)
        
        # Draw nodes
        node_colors = [self._get_node_color(n) for n in self.G.nodes()]
        node_sizes = [self._get_node_size(n) for n in self.G.nodes()]
        
        nx.draw_networkx_nodes(self.G, pos, 
                             node_color=node_colors,
                             node_size=node_sizes)
        
        # Draw edges
        nx.draw_networkx_edges(self.G, pos, 
                             edge_color='gray',
                             arrows=True)
        
        # Add labels
        labels = {n: f"PDG: {self.G.nodes[n]['pdg']}\n" + 
                    f"pT: {self.G.nodes[n]['pt']:.2f}" 
                 for n in self.G.nodes()}
        nx.draw_networkx_labels(self.G, pos, labels)
        
        plt.title("Particle Decay Chain")
        plt.axis('off')
        plt.tight_layout()
        
    def _get_node_color(self, node):
        """Color nodes based on detector hits"""
        has_tracker = self.G.nodes[node]['n_tracker_hits'] > 0
        has_calo = self.G.nodes[node]['n_calo_hits'] > 0
        
        if has_tracker and has_calo:
            return 'green'
        elif has_tracker:
            return 'blue'
        elif has_calo:
            return 'red'
        return 'gray'
    
    def _get_node_size(self, node):
        """Size nodes based on pT"""
        return 1000 * np.log1p(self.G.nodes[node]['pt'])
    
    def _get_node_depth(self, node):
        """Get depth relative to starting particle"""
        return self._depths.get(node, 0)  # Default to 0 if not found
        
    def plot_hits(self, figsize=(12, 12), projection='xy', max_depth=3):
        """Plot hits with markers indicating relative depth from starting particle"""
        plt.figure(figsize=figsize)
        
        # Define markers for different depths
        markers = ['o', 's', '^', 'v', 'D', 'p']  # Add more if needed
        # Center the color map around depth 0
        max_abs_depth = max(abs(min(self._depths.values())), abs(max(self._depths.values())))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.G.nodes())))
        
        # For each node in the graph
        for i, node in enumerate(self.G.nodes()):
            # Get hits for this particle
            hits = self.chain.get_detector_hits(node)
            
            # Get node depth in the decay chain
            depth = self._get_node_depth(node)
            if abs(depth) > max_depth:
                continue
                
            marker = markers[abs(depth) % len(markers)]
            color = colors[i]
            
            # Plot tracker hits
            if len(hits['tracker']) > 0:
                df = hits['tracker']
                if projection == 'xy':
                    plt.scatter(df.x, df.y, 
                              marker=marker, color=color, alpha=0.6,
                              label=f'Tracker (depth={depth})')
                elif projection == 'xz':
                    plt.scatter(df.x, df.z,
                              marker=marker, color=color, alpha=0.6,
                              label=f'Tracker (depth={depth})')
                elif projection == 'yz':
                    plt.scatter(df.y, df.z,
                              marker=marker, color=color, alpha=0.6,
                              label=f'Tracker (depth={depth})')
                elif projection == 'eta_phi':
                    plt.scatter(df.eta, df.phi,
                              marker=marker, color=color, alpha=0.6,
                              label=f'Tracker (depth={depth})')
            
            # Plot calorimeter hits
            if len(hits['calo']) > 0:
                df = hits['calo']
                # Calculate normalized hit sizes based on energy
                hit_sizes = df.energy
                normalized_sizes = hit_sizes / hit_sizes.max() 
                hit_sizes = normalized_sizes * 100  # Scale factor for visibility
                
                if projection == 'xy':
                    plt.scatter(df.x, df.y,
                              marker=marker, color=color, alpha=normalized_sizes,
                              s=hit_sizes,  # Size by energy
                              label=f'Calo (depth={depth})')
                elif projection == 'xz':
                    plt.scatter(df.x, df.z,
                              marker=marker, color=color, alpha=normalized_sizes,
                              s=hit_sizes,
                              label=f'Calo (depth={depth})')
                elif projection == 'yz':
                    plt.scatter(df.y, df.z,
                              marker=marker, color=color, alpha=normalized_sizes,
                              s=hit_sizes,
                              label=f'Calo (depth={depth})')
                elif projection == 'eta_phi':
                    plt.scatter(df.eta, df.phi,
                              marker=marker, color=color, alpha=normalized_sizes,
                              s=hit_sizes,
                              label=f'Calo (depth={depth})')
        
        # Add labels and title
        plt.xlabel(projection.split('_')[0])
        plt.ylabel(projection.split('_')[1])
        plt.title(f"Hit Positions by Decay Chain Depth ({projection} projection)")
        
        # Add legend but combine duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

def analyze_cell_energy_distributions(hits, cells_df, energy_percentile=90, title_prefix=""):
    """Analyze the energy distributions in calorimeter cells.
    
    Args:
        hits: DataFrame containing hit information (either raw hits or particle-summed hits)
        cells_df: DataFrame containing cell information
        energy_percentile: Percentile threshold for "high energy" cells (default: 90)
        title_prefix: Optional prefix for plot titles
    """
    # Get highest energy hit per cell and corresponding particle ID
    highest_energy_hits = hits.sort_values('energy').groupby('cellID').last()[['energy', 'particle_id']].reset_index()
    
    # Merge with cell energies
    energy_comparison = pd.merge(
        highest_energy_hits,
        cells_df[['cellID', 'energy', 'eta', 'phi']],
        on='cellID',
        suffixes=('_max', '_total')
    )
    
    # Calculate threshold and filter high energy cells
    energy_threshold = energy_comparison.energy_total.quantile(energy_percentile/100)
    high_energy = energy_comparison[energy_comparison.energy_total > energy_threshold].copy()
    
    # Calculate ratios
    high_energy['remaining_energy'] = high_energy.energy_total - high_energy.energy_max
    high_energy['highest_hit_ratio'] = high_energy.remaining_energy / high_energy.energy_max
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Scatter plot of high energy cells
    ax1.scatter(high_energy.energy_total, 
               high_energy.energy_max, 
               alpha=0.5, 
               s=1)
    max_val = max(high_energy.energy_total.max(), high_energy.energy_max.max())
    ax1.plot([energy_threshold, max_val], [energy_threshold, max_val], 
             'r--', label='y=x')
    ax1.set_xlabel('Total Cell Energy')
    ax1.set_ylabel('Highest Single Hit Energy')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.set_title(f'{title_prefix}High Energy Cells (Top {100-energy_percentile}%)')
    
    # 2. Histogram of energy fractions
    ax2.hist(high_energy.highest_hit_ratio, bins=50, alpha=0.7)
    ax2.set_xlabel('Fraction of Energy in Highest Hit')
    ax2.set_ylabel('Number of Cells')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title(f'{title_prefix}Energy Fraction Distribution\nHigh Energy Cells Only')
    
    # Add to ax2 a line at x=1, and add a textbox with the number of cells with a highest hit ratio of less than 1
    ax2.axvline(x=1, color='r', linestyle='--')
    ax2.text(1.1, ax2.get_ylim()[1], 
             f"Ratio < 1: {len(high_energy[high_energy.highest_hit_ratio < 1])}\n" +
             f"Ratio > 1: {len(high_energy[high_energy.highest_hit_ratio > 1])}")
    
    ax2.axvline(x=0.1, color='r', linestyle='--')
    ax2.text(0.11, ax2.get_ylim()[1], 
             f"Ratio < 0.1: {len(high_energy[high_energy.highest_hit_ratio < 0.1])}\n" +
             f"Ratio > 0.1: {len(high_energy[high_energy.highest_hit_ratio > 0.1])}")
    
    # 3. Scatter of highest hit vs remaining energy
    ax3.scatter(high_energy.remaining_energy, 
               high_energy.energy_max, 
               alpha=0.5, 
               s=1)
    ax3.set_xlabel('Remaining Energy in Cell')
    ax3.set_ylabel('Highest Hit Energy')
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    ax3.set_title(f'{title_prefix}Highest Hit vs Remaining Energy')

    # Add x=y line
    ax3.plot([0, max_val], [0, max_val], 'r--', label='y=x')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nAnalysis of cells with energy > {energy_threshold:.2e}:")
    print(f"Number of cells: {len(high_energy)}")
    print("\nRatio statistics (highest hit / total energy):")
    print(high_energy.highest_hit_ratio.describe())
    print("\nMedian remaining energy:", high_energy.remaining_energy.median())
    
    return high_energy