import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any, Union
import networkx as nx
import pandas as pd

# Import utility functions
from .utils import get_simulator_status_bits

if TYPE_CHECKING:
    from .event import EDM4hepEvent
    from .particle import Particle

class PlottingHandler:
    """
    Handles plotting and visualization of the event data.
    
    This class adapts functionality from viz_utils.py and particle_decay_tree.py
    to work with the EDM4hepEvent object model, providing methods to visualize
    particles, hits, and decay trees.
    """
    
    def __init__(self, event: 'EDM4hepEvent'):
        """
        Initialize the PlottingHandler.
        
        Args:
            event (EDM4hepEvent): The event to visualize.
        """
        self._event = event
    
    def event_overview(self, figsize=(12, 10)):
        """
        Plot an overview of the event showing hits, particle vertices, etc.
        
        Requires that detector_params were provided during EDM4hepEvent initialization.
        
        Args:
            figsize: Size of the figure as (width, height)
                
        Returns:
            matplotlib.figure.Figure: The figure object containing the plots.
        """
        if self._event.detector_params is None:
            print("Error: event_overview requires detector_params to be set during EDM4hepEvent initialization.")
            return None
            
        fig = plt.figure(figsize=figsize)
        
        # Get DataFrames
        tracker_df = self._event._tracker_hits_df
        particles_df = self._event._particles_df
        parents_df = self._event._parents_df
        daughters_df = self._event._daughters_df
        hits_df = self._event._calo_hits_df
        contrib_df = self._event._calo_contributions_df
        
        # Use the particles_df directly from the event (contains pre-calculated columns)
        particles_df = self._event._particles_df # Use directly, no copy needed if not modifying further?
                                                # Copy just in case for safety:
        particles_df = self._event._particles_df.copy()
        
        # Ensure required geometry-dependent columns exist (should have been created in __init__)
        required_cols = ["vr", "endpoint_r", "created_in_simulation", "backscatter", 
                         "created_inside_tracker", "ended_inside_tracker"]
        if not all(col in particles_df.columns for col in required_cols):
            print("Error: Required pre-calculated geometry columns missing in particles_df. Ensure detector_params were provided during event init.")
            # Optional: Add fallback calculation here if desired, but ideally init handles it.
            return None # Cannot proceed without geometry flags
        
        # Get geometry parameters from the event object
        tracking_radius = self._event.detector_params['tracking_radius']
        tracking_z_max = self._event.detector_params['tracking_z_max']
        
        # Find particles with hits
        particles_with_hits_ids = set()
        
        # Add particles from tracker hits
        if not self._event._tracker_links_df.empty:
            particles_with_hits_ids.update(self._event._tracker_links_df['particle_id'].unique())
        
        # Add particles from calo contributions
        if not self._event._calo_links_df.empty:
            particles_with_hits_ids.update(self._event._calo_links_df['particle_id'].unique())
        
        # Create DataFrame of particles with hits
        particles_with_hits = particles_df.loc[particles_df.index.isin(particles_with_hits_ids)]
        
        # Plots:
        
        # 1. Particle distribution by origin, location, and backscatter
        self._plot_particle_distribution(particles_df, particles_with_hits)
        
        # 2. Plot hits in r-z plane and tracking volume
        plt.figure(figsize=(12, 8))
        plt.scatter(hits_df.z, hits_df.r, c="red", s=1, label="Calo hits")
        plt.scatter(tracker_df.z, tracker_df.r, c="blue", s=2, label="Tracker hits")
        
        # Plot tracking volume outline
        plt.plot(
            [-tracking_z_max, -tracking_z_max, tracking_z_max, tracking_z_max, -tracking_z_max], 
            [0, tracking_radius, tracking_radius, 0, 0], 
            c="black", lw=1, label="Tracking cylinder"
        )
        
        plt.gca().set_aspect('equal', 'box')
        plt.title(f"Event {self._event.event_index}")
        plt.xlabel("z [mm]")
        plt.ylabel("r [mm]")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # 3. Histogram of particle vertex z coordinate
        plt.figure(figsize=(12, 6))
        sns.histplot(particles_df.vz, bins=30, binrange=(-1.2*tracking_z_max, 1.2*tracking_z_max), 
                   label="All particles", alpha=0.7)
        sns.histplot(particles_with_hits.vz, bins=30, binrange=(-1.2*tracking_z_max, 1.2*tracking_z_max), 
                   label="Particles with hits", alpha=0.7)
        plt.axvline(tracking_z_max, color="black", linestyle="--", lw=1)
        plt.axvline(-tracking_z_max, color="black", linestyle="--", lw=1)
        plt.yscale('log')
        plt.title(f"Production vertex z coordinate for event {self._event.event_index}")
        plt.legend()
        plt.xlabel("z [mm]")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()
        
        # 4. Histogram of particle vertex r coordinate
        plt.figure(figsize=(12, 6))
        sns.histplot(particles_df.vr, bins=30, binrange=(0, 1.2*tracking_radius), 
                   label="All particles", alpha=0.7)
        sns.histplot(particles_with_hits.vr, bins=30, binrange=(0, 1.2*tracking_radius), 
                   label="Particles with hits", alpha=0.7)
        plt.axvline(tracking_radius, color="black", linestyle="--", lw=1)
        plt.yscale('log')
        plt.title(f"Production vertex r coordinate for event {self._event.event_index}")
        plt.legend()
        plt.xlabel("r [mm]")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def _plot_particle_distribution(self, particles_df, particles_with_hits):
        """
        Plot distribution of particles by origin, location, and backscatter status.
        
        Args:
            particles_df: Full particles DataFrame
            particles_with_hits: DataFrame of particles that have hits
        """
        # Generate the counts for each combination
        data = {}
        for origin_val, origin_name in [(0, "Generator"), (1, "Simulation")]:
            for loc_val, loc_name in [(1, "Inside"), (0, "Outside")]:
                # Count non-backscatter (0) and backscatter (1) particles
                mask = ((particles_with_hits["created_in_simulation"] == origin_val) & 
                        (particles_with_hits["created_inside_tracker"] == loc_val))
                
                no_bs = ((mask) & (particles_with_hits["backscatter"] == False)).sum()
                yes_bs = ((mask) & (particles_with_hits["backscatter"] == True)).sum()
                
                data[(origin_name, loc_name)] = [no_bs, yes_bs]
    
        # Create figure
        fig, ax = plt.figure(figsize=(7, 6)), plt.gca()
    
        # Set up the grid
        rows = ["Simulation", "Generator"]
        cols = ["Inside", "Outside"]
        cell_width, cell_height = 1.0, 1.0
    
        # Create the grid with diagonally split cells
        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                # Cell position
                x, y = j * cell_width, (len(rows) - 1 - i) * cell_height
                
                # Get data
                no_bs, yes_bs = data[(row, col)]
                
                # Create the cell
                rect = patches.Rectangle((x, y), cell_width, cell_height, 
                                        linewidth=2, edgecolor='black', facecolor='none')
                ax.add_patch(rect)
                
                # Draw diagonal
                ax.plot([x, x + cell_width], [y, y + cell_height], 'k-', linewidth=1)
                
                # Add counts
                # Non-backscatter in bottom-left triangle
                ax.text(x + 0.25, y + 0.70, f"{no_bs}", 
                        ha='center', va='center', fontsize=12, fontweight='bold')
                
                # Backscatter in top-right triangle  
                ax.text(x + 0.75, y + 0.40, f"{yes_bs}", 
                        ha='center', va='center', fontsize=12, fontweight='bold')
                
                # Add labels for the diagonals (smaller text)
                ax.text(x + 0.15, y + 0.4, "Not BS", ha='center', va='center', fontsize=8, rotation=45)
                ax.text(x + 0.85, y + 0.6, "Is BS", ha='center', va='center', fontsize=8, rotation=45)
    
        # Add row and column labels
        for i, row in enumerate(rows):
            ax.text(-0.2, (len(rows) - 1 - i) * cell_height + 0.5, row, 
                    ha='right', va='center', fontsize=12)
    
        for j, col in enumerate(cols):
            ax.text(j * cell_width + 0.5, len(rows) * cell_height + 0.1, col, 
                    ha='center', va='bottom', fontsize=12)
    
        # Title
        ax.text(len(cols) * cell_width / 2, len(rows) * cell_height + 0.3, 
                "Particle Distribution by Origin, Location, and Backscatter",
                ha='center', fontsize=14)
    
        # Set limits and remove axes
        ax.set_xlim(-0.5, len(cols) * cell_width + 0.5)
        ax.set_ylim(-0.5, len(rows) * cell_height + 0.5)
        ax.axis('off')
    
        plt.tight_layout()
        plt.show()
    
        # Also print the raw data as a reference
        print("Data Summary:")
        print("Location  | Origin     | No Backscatter | Backscatter")
        print("-" * 55)
        for (origin, location), (no_bs, yes_bs) in sorted(data.items()):
            print(f"{location:9} | {origin:10} | {no_bs:13} | {yes_bs:10}")
    
        # Print totals across different dimensions
        print("\nTotals by Category:")
        print("PARTICLES WITH HITS")
        print(f"Total particles: {len(particles_with_hits)}")
        print(f"From Generator: {(particles_with_hits['created_in_simulation'] == 0).sum()}")
        print(f"From Simulation: {(particles_with_hits['created_in_simulation'] == 1).sum()}")
        print(f"Inside Tracker: {(particles_with_hits['created_inside_tracker'] == 1).sum()}")
        print(f"Outside Tracker: {(particles_with_hits['created_inside_tracker'] == 0).sum()}")
        print(f"Backscatter: {(particles_with_hits['backscatter'] == True).sum()}")
    
        print("\nALL PARTICLES")
        print(f"Total particles: {len(particles_df)}")
        print(f"From Generator: {(particles_df['created_in_simulation'] == 0).sum()}")
        print(f"From Simulation: {(particles_df['created_in_simulation'] == 1).sum()}")
        print(f"Inside Tracker: {(particles_df['created_inside_tracker'] == 1).sum()}")
        print(f"Outside Tracker: {(particles_df['created_inside_tracker'] == 0).sum()}")
    
    def visualize_decay_tree(self, particle_id: Optional[int] = None, 
                          highlight_collapsed: Optional[int] = None,
                          show_tracking_cylinder: bool = True,
                          figsize: Tuple[int, int] = (14, 12)):
        """
        Create a 3D visualization of the particle decay tree.
        
        Requires that detector_params were provided during EDM4hepEvent initialization if show_tracking_cylinder=True.

        Args:
            particle_id: Optional particle ID to show only its decay tree.
               If None, shows the full decay tree.
            highlight_collapsed: Optional ID to highlight all particles with this collapsedParticleID.
            show_tracking_cylinder: Whether to show the tracking volume outline.
            figsize: Size of the figure as (width, height).
            
        Returns:
            matplotlib.figure.Figure: The figure object containing the 3D plot.
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        detector_params = self._event.detector_params # Get params from event
        
        if detector_params is None and show_tracking_cylinder:
            print("Warning: detector_params not set in event, cannot show tracking cylinder.")
            show_tracking_cylinder = False
            
        # Make sure the decay graph is built
        G = self._event.get_decay_tree()
        
        # If particle_id is specified, extract subgraph for that particle and its descendants
        if particle_id is not None:
            if particle_id not in G:
                print(f"Error: Particle {particle_id} not found in the decay graph.")
                return
            
            descendants = nx.descendants(G, particle_id)
            nodes_to_include = {particle_id} | descendants
            G = G.subgraph(nodes_to_include)
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Define edge colors based on detector regions
        edge_colors = {
            'tracker->tracker': 'blue',
            'tracker->calo': 'green',
            'calo->tracker': 'orange',
            'calo->calo': 'red',
        }
        
        # Helper function to check if a point is in tracking cylinder
        def in_tracking_cylinder(x, y, z):
            if detector_params is None:
                return True  # Default to True if no detector params
            r = np.sqrt(x**2 + y**2)
            return r < detector_params['tracking_radius'] and abs(z) < detector_params['tracking_z_max']
        
        # Draw edges (decay relationships)
        for parent, child, data in G.edges(data=True):
            # Parent node position
            parent_x = G.nodes[parent]['decay_vertex_x']
            parent_y = G.nodes[parent]['decay_vertex_y']
            parent_z = G.nodes[parent]['decay_vertex_z']
            
            # Child node position
            child_x = G.nodes[child]['decay_vertex_x']
            child_y = G.nodes[child]['decay_vertex_y']
            child_z = G.nodes[child]['decay_vertex_z']
            
            # Determine detector regions
            parent_in_tracking = in_tracking_cylinder(parent_x, parent_y, parent_z)
            child_in_tracking = in_tracking_cylinder(child_x, child_y, child_z)
            
            # Determine edge type and color
            if parent_in_tracking and child_in_tracking:
                edge_type = 'tracker->tracker'
            elif parent_in_tracking and not child_in_tracking:
                edge_type = 'tracker->calo'
            elif not parent_in_tracking and child_in_tracking:
                edge_type = 'calo->tracker'
            else:
                edge_type = 'calo->calo'
            
            edge_color = edge_colors[edge_type]
            
            # If highlighting by collapsedParticleID, override color if needed
            if highlight_collapsed is not None:
                collapsed_id = G.nodes[child].get('collapsedParticleID', -1)
                if collapsed_id == highlight_collapsed:
                    linewidth = 2.5
                    alpha = 1.0
                    # Use a distinctive color that's different from the edge type colors
                    edge_color = 'magenta'
                else:
                    linewidth = 1.0
                    alpha = 0.6
            else:
                linewidth = 1.0
                alpha = 0.6
            
            # Draw line for the decay relationship
            ax.plot([parent_x, child_x], [parent_y, child_y], [parent_z, child_z], 
                    color=edge_color, lw=linewidth, alpha=alpha, 
                    label=edge_type if parent == list(G.nodes())[0] else "")
        
        # Draw nodes (particles)
        for node, data in G.nodes(data=True):
            x, y, z = data['decay_vertex_x'], data['decay_vertex_y'], data['decay_vertex_z']
            is_in_tracking = in_tracking_cylinder(x, y, z)
            
            # Color based on tracking cylinder location
            color = 'blue' if is_in_tracking else 'red'
            
            # Size based on energy (optional)
            energy = data.get('energy', 0)
            size = min(30, max(10, energy / 5)) if energy else 20
            
            # Highlight collapsed ID if specified
            if highlight_collapsed is not None and data.get('collapsedParticleID', -1) == highlight_collapsed:
                color = 'magenta'
                size *= 1.5
            
            ax.scatter(x, y, z, c=color, s=size, alpha=0.7)
            
            # Add labels for particleID, collapsedParticleID, incidentParentID, and energy
            particle_id = data.get('particleID', 'N/A')
            collapsed_id = data.get('collapsedParticleID', -1)
            incident_id = data.get('incidentParentID', -1)
            
            # Format the label text with energy
            label_text = f"ID: {particle_id}\nC: {collapsed_id}\nI: {incident_id}\nE: {energy:.2f} GeV"
            
            # Position the label slightly offset from the node
            # Use a small offset in 3D space
            offset_x, offset_y, offset_z = 5, 5, 5
            
            # Add the text with a small background box
            ax.text(x + offset_x, y + offset_y, z + offset_z, label_text, 
                    fontsize=8, color='black', 
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
        
        # Draw the tracking cylinder if requested
        if show_tracking_cylinder and detector_params is not None:
            tracking_radius = detector_params['tracking_radius']
            tracking_z_max = detector_params['tracking_z_max']
            
            # Create points for cylinder
            theta = np.linspace(0, 2*np.pi, 100)
            z = np.linspace(-tracking_z_max, tracking_z_max, 2)
            theta_grid, z_grid = np.meshgrid(theta, z)
            x_grid = tracking_radius * np.cos(theta_grid)
            y_grid = tracking_radius * np.sin(theta_grid)
            
            # Draw cylinder surface
            ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.1, color='lightblue')
            
            # Draw end caps
            r = np.linspace(0, tracking_radius, 10)
            theta_grid, r_grid = np.meshgrid(theta, r)
            x_grid = r_grid * np.cos(theta_grid)
            y_grid = r_grid * np.sin(theta_grid)
            
            # Top cap
            z_grid = np.ones_like(x_grid) * tracking_z_max
            ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.1, color='lightblue')
            
            # Bottom cap
            z_grid = np.ones_like(x_grid) * (-tracking_z_max)
            ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.1, color='lightblue')
        
        # Add legend items
        legend_items = []
        
        # For particles
        legend_items.append(ax.scatter([], [], c='blue', s=20, label='Particle in Tracking'))
        legend_items.append(ax.scatter([], [], c='red', s=20, label='Particle in Calorimeter'))
        
        # For edge types (create unique labels)
        edge_handles = []
        edge_labels = []
        for edge_type, color in edge_colors.items():
            line = ax.plot([], [], color=color, linewidth=2)[0]
            edge_handles.append(line)
            edge_labels.append(edge_type)
        
        # If highlighting
        if highlight_collapsed is not None:
            legend_items.append(ax.plot([], [], color='magenta', linewidth=2.5, 
                                       label=f'Highlighted (ID: {highlight_collapsed})')[0])
        
        ax.legend(handles=legend_items + edge_handles, labels=[item.get_label() for item in legend_items] + edge_labels,
                 loc='upper right', bbox_to_anchor=(1.1, 1))
        
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        
        if particle_id is not None:
            ax.set_title(f'Decay Tree for Particle {particle_id}')
        else:
            ax.set_title(f'Full Decay Tree for Event {self._event.event_index}')
        
        # Set axis limits based on particle positions
        all_coords = []
        for _, data in G.nodes(data=True):
            all_coords.append((abs(data['decay_vertex_x']), 
                              abs(data['decay_vertex_y']), 
                              abs(data['decay_vertex_z'])))
        
        if all_coords:
            max_coord = max([max(x, y, z) for x, y, z in all_coords])
            ax.set_xlim(-max_coord*1.1, max_coord*1.1)
            ax.set_ylim(-max_coord*1.1, max_coord*1.1)
            ax.set_zlim(-max_coord*1.1, max_coord*1.1)
        
        plt.tight_layout()
        # plt.show()
        
        return fig
    
    def highlight_ancestors_descendants(self, particle_id: int, figsize: Tuple[int, int] = (10, 10)):
        """
        Plot the vertex positions of ancestors and descendants of a particle.
        
        Args:
            particle_id: ID of the particle to analyze.
            figsize: Size of the figure as (width, height).
            
        Returns:
            matplotlib.figure.Figure: The figure object containing the plot.
        """
        # Make sure particle exists
        if particle_id not in self._event._particles_df.index:
            print(f"Error: Particle {particle_id} not found in the event.")
            return
        
        # Get particle
        particle = self._event.get_particle(particle_id)
        
        # Get ancestors and descendants
        ancestors = particle.get_ancestors()
        descendants = particle.get_descendants()
        
        # If no decay tree built yet
        if not ancestors and not descendants:
            print(f"Warning: No ancestors or descendants found. The decay tree might not be built yet.")
            self._event.get_decay_tree()
            ancestors = particle.get_ancestors()
            descendants = particle.get_descendants()
        
        # Get ancestor and descendant IDs
        ancestor_ids = [p.id for p in ancestors]
        descendant_ids = [p.id for p in descendants]
        
        # Create a DataFrame with vertex positions
        particles_df = self._event._particles_df
        
        # Calculate vertex r coordinate
        particles_df_copy = particles_df.copy()
        particles_df_copy["vr"] = np.sqrt(particles_df.vx**2 + particles_df.vy**2)
        
        fig = plt.figure(figsize=figsize)
        
        # Plot vertices in r-z space
        plt.scatter(particles_df_copy.loc[ancestor_ids].vz, 
                  particles_df_copy.loc[ancestor_ids].vr, 
                  c="red", s=50, marker="x", label="Ancestor vertices")
        
        plt.scatter(particles_df_copy.loc[descendant_ids].vz, 
                  particles_df_copy.loc[descendant_ids].vr, 
                  c="blue", s=50, marker="x", label="Descendant vertices")
        
        plt.scatter(particles_df_copy.loc[particle_id].vz, 
                  particles_df_copy.loc[particle_id].vr, 
                  c="green", s=100, marker="*", label="Selected particle")
        
        # Add particle IDs as labels
        for ancestor_id in ancestor_ids:
            plt.text(particles_df_copy.loc[ancestor_id].vz, 
                   particles_df_copy.loc[ancestor_id].vr, 
                   f"{ancestor_id}", fontsize=10, ha='right')
        
        for descendant_id in descendant_ids:
            plt.text(particles_df_copy.loc[descendant_id].vz, 
                   particles_df_copy.loc[descendant_id].vr, 
                   f"{descendant_id}", fontsize=10, ha='left')
        
        plt.text(particles_df_copy.loc[particle_id].vz, 
               particles_df_copy.loc[particle_id].vr, 
               f"{particle_id}", fontsize=12, ha='center', va='bottom')
        
        plt.xlabel("z [mm]")
        plt.ylabel("r [mm]")
        plt.title(f"Ancestors and Descendants of Particle {particle_id} (PDG: {particle.pdg})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_particle_kinematics(self, particle_selection: Optional[Union[List[int], str]] = None,
                              figsize: Tuple[int, int] = (15, 10)):
        """
        Plot kinematic distributions for selected particles.
        
        Args:
            particle_selection: Either a list of particle IDs, or one of:
               - 'all': All particles
               - 'with_hits': Only particles with tracker or calo hits
               - 'primary': Only primary particles (no parents)
               - 'stable': Only stable particles (no daughters)
            figsize: Size of the figure as (width, height).
            
        Returns:
            matplotlib.figure.Figure: The figure object containing the plots.
        """
        particles_df = self._event._particles_df
        
        # Handle particle selection
        if particle_selection is None or particle_selection == 'all':
            selected_df = particles_df
            selection_name = "All Particles"
        elif particle_selection == 'with_hits':
            # Particles with tracker hits
            tracker_particle_ids = set()
            if not self._event._tracker_links_df.empty:
                tracker_particle_ids = set(self._event._tracker_links_df['particle_id'].unique())
            
            # Particles with calo hits
            calo_particle_ids = set()
            if not self._event._calo_links_df.empty:
                calo_particle_ids = set(self._event._calo_links_df['particle_id'].unique())
            
            # Combine
            particle_ids = tracker_particle_ids | calo_particle_ids
            selected_df = particles_df.loc[particles_df.index.isin(particle_ids)]
            selection_name = "Particles with Hits"
        elif particle_selection == 'primary':
            # No parents or empty parents range
            selected_df = particles_df[
                (particles_df['parents_begin'] >= particles_df['parents_end']) |
                (particles_df['parents_begin'] == 0 and particles_df['parents_end'] == 0)
            ]
            selection_name = "Primary Particles"
        elif particle_selection == 'stable':
            # No daughters or empty daughters range
            selected_df = particles_df[
                (particles_df['daughters_begin'] >= particles_df['daughters_end']) |
                (particles_df['daughters_begin'] == 0 and particles_df['daughters_end'] == 0)
            ]
            selection_name = "Stable Particles"
        elif isinstance(particle_selection, list):
            # Specific particle IDs
            selected_df = particles_df.loc[particles_df.index.isin(particle_selection)]
            selection_name = f"Selected Particles ({len(selected_df)} of {len(particles_df)})"
        else:
            raise ValueError(f"Invalid particle_selection: {particle_selection}")
        
        if selected_df.empty:
            print(f"Warning: No particles match the selection criteria.")
            return
        
        fig = plt.figure(figsize=figsize)
        
        # Create a 2x2 grid of plots
        plt.subplot(2, 2, 1)
        plt.hist(selected_df['pt'], bins=50, alpha=0.7)
        plt.xlabel('pT [GeV]')
        plt.ylabel('Count')
        plt.title('Transverse Momentum')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.hist(selected_df['eta'], bins=50, alpha=0.7)
        plt.xlabel('η (Pseudorapidity)')
        plt.ylabel('Count')
        plt.title('Pseudorapidity')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.hist(selected_df['phi'], bins=50, alpha=0.7)
        plt.xlabel('φ (Azimuthal Angle)')
        plt.ylabel('Count')
        plt.title('Azimuthal Angle')
        plt.grid(True, alpha=0.3)
        
        # Get mass for particles with mass > 0
        mass_df = selected_df[selected_df['mass'] > 0]
        plt.subplot(2, 2, 4)
        plt.hist(mass_df['mass'], bins=50, alpha=0.7)
        plt.xlabel('Mass [GeV]')
        plt.ylabel('Count')
        plt.title('Mass (m > 0)')
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f"Kinematics for {selection_name} in Event {self._event.event_index}", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Make room for the suptitle
        plt.show()
        
        return fig
