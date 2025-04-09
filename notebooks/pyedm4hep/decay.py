import networkx as nx
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any, Set

if TYPE_CHECKING:
    from .event import EDM4hepEvent
    from .particle import Particle

class DecayGraphHandler:
    """
    Handles the creation and analysis of the particle decay graph.
    
    This class adapts functionality from the original particle_decay_tree.py 
    to work with the EDM4hepEvent object model, providing methods to build
    the decay graph, process particle relationships, and analyze the decay tree.
    """
    
    def __init__(self, event: 'EDM4hepEvent'):
        """
        Initialize the DecayGraphHandler.
        
        Args:
            event (EDM4hepEvent): The event containing particle data.
        """
        self._event = event
        self._graph: Optional[nx.DiGraph] = None
        self._collapsed_info = None  # Will hold dataframe with info about "collapsed" particles
    
    def get_graph(self) -> nx.DiGraph:
        """
        Get the decay graph, building it if necessary.
        
        Returns:
            nx.DiGraph: Directed graph representing the decay tree.
        """
        if self._graph is None:
            self.build_graph()
        return self._graph
    
    def build_graph(self) -> nx.DiGraph:
        """
        Build a decay tree graph where:
        - Nodes are particles
        - Edges represent parent-child relationships
        
        Returns:
            nx.DiGraph: The constructed graph.
        """
        # Get the relevant dataframes
        particles_df = self._event._particles_df
        daughters_df = self._event._daughters_df
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add all particles as nodes
        print(f"Creating nodes for {len(particles_df)} particles...")
        
        nodes_df = particles_df.copy()
        # Add any extra attributes you want in the graph nodes
        nodes_df["particleID"] = nodes_df.index
        nodes_df["collapsedParticleID"] = nodes_df.index
        nodes_df["incidentParentID"] = nodes_df.index
        
        # Rename DataFrame columns to more descriptive names for node attributes
        nodes_df.rename(columns={
            "p": "energy", 
            "vx": "decay_vertex_x", 
            "vy": "decay_vertex_y", 
            "vz": "decay_vertex_z"
        }, inplace=True)
        
        # Create a dictionary of node attributes for efficient graph creation
        node_attrs = nodes_df[['energy', 'decay_vertex_x', 'decay_vertex_y', 'decay_vertex_z',
                               'particleID', 'collapsedParticleID', 'incidentParentID',
                               'PDG', 'pt', 'eta', 'phi']].to_dict('index')
        
        # Add all nodes at once
        G.add_nodes_from(node_attrs.items())
        
        # Add edges based on parent-child relationships if daughters_df is available and non-empty
        if not daughters_df.empty:
            print("Creating edges based on parent-child relationships...")
            
            # Calculate how many daughters each particle has
            num_daughters = (particles_df["daughters_end"] - particles_df["daughters_begin"]).values
            
            # Create source array: repeat each parent ID for its number of daughters
            parent_ids = np.repeat(nodes_df.particleID.values, num_daughters)
            
            # Create DataFrame that maps each daughter to its parent
            parent_daughter_map = pd.DataFrame({
                'parent_id': parent_ids,
                'daughter_id': daughters_df['particle_id'].values
            })
            
            # Add all edges at once (from parent to child)
            G.add_edges_from(parent_daughter_map[['parent_id', 'daughter_id']].values)
            print(f"Created {G.number_of_edges()} edges")
        else:
            print("No daughter information available. Graph will only contain nodes.")
        
        # Store and return the graph
        self._graph = G
        return G
    
    def process_decay_tree(self, energy_threshold: Optional[float] = None) -> nx.DiGraph:
        """
        Process the decay tree to assign collapsedParticleID and incidentParentID.
        
        Requires detector_params to have been set during EDM4hepEvent initialization.
        Optionally uses an energy threshold for collapsing decisions.

        Args:
            energy_threshold (Optional[float]): Energy threshold for considering separate 
                                                particles during collapsing. If None, uses 
                                                value from event's detector_params or defaults to 0.

        Returns:
            nx.DiGraph: The processed graph.
        """
        # Make sure the graph is built
        G = self.get_graph()
        detector_params = self._event.detector_params
        
        if detector_params is None:
            print("Error: process_decay_tree requires detector_params to be set during EDM4hepEvent initialization.")
            return G # Return unprocessed graph

        # Determine energy threshold
        if energy_threshold is None:
            energy_threshold = detector_params.get('energy_threshold', 0.0)
        
        # Get geometry params from event
        tracking_radius = detector_params.get('tracking_radius')
        tracking_z_max = detector_params.get('tracking_z_max')
        
        if tracking_radius is None or tracking_z_max is None:
            print("Error: process_decay_tree requires 'tracking_radius' and 'tracking_z_max' in detector_params.")
            return G # Return unprocessed graph
            
        print("Finding root nodes...")
        root_nodes = [node for node, in_degree in G.in_degree() if in_degree == 0]
        print(f"Found {len(root_nodes)} root nodes")
        
        # Queue for breadth-first traversal
        queue = root_nodes.copy()
        visited_nodes = set()
        
        def in_tracking_cylinder(x, y, z):
            """Check if a point is inside the tracking cylinder."""
            r = np.sqrt(x**2 + y**2)
            return r < tracking_radius and abs(z) < tracking_z_max
        
        print("Processing nodes in topological order...")
        # Process the graph breadth-first
        while queue:
            current_node = queue.pop(0)
            
            # Skip if already visited
            if current_node in visited_nodes:
                continue
            
            visited_nodes.add(current_node)
            
            # Get node position
            x = G.nodes[current_node]['decay_vertex_x']
            y = G.nodes[current_node]['decay_vertex_y']
            z = G.nodes[current_node]['decay_vertex_z']
            
            # Get particle ID
            parent_particle_id = G.nodes[current_node]['particleID']
            # Check if particle is in tracking cylinder
            parent_in_tracking = in_tracking_cylinder(x, y, z)
            
            # Process each outgoing edge
            for _, child_node in G.out_edges(current_node, data=False):
                child_x = G.nodes[child_node]['decay_vertex_x']
                child_y = G.nodes[child_node]['decay_vertex_y']
                child_z = G.nodes[child_node]['decay_vertex_z']
                
                # Get child particle ID
                child_particle_id = child_node
                child_in_tracking = in_tracking_cylinder(child_x, child_y, child_z)
                
                # Case a: Parent and child in tracking cylinder
                if parent_in_tracking and child_in_tracking:
                    G.nodes[child_node]['collapsedParticleID'] = child_particle_id
                # Case b: Parent in tracking, child in calo
                elif parent_in_tracking and not child_in_tracking:
                    energy = G.nodes[child_node]['energy']
                    # Check energy threshold
                    if energy > energy_threshold:
                        G.nodes[child_node]['collapsedParticleID'] = child_particle_id
                        G.nodes[child_node]['incidentParentID'] = parent_particle_id
                    else:
                        G.nodes[child_node]['collapsedParticleID'] = parent_particle_id
                        G.nodes[child_node]['incidentParentID'] = parent_particle_id
                # Case c: Parent in calo, child in calo
                elif not parent_in_tracking and not child_in_tracking:
                    energy = G.nodes[child_node]['energy']
                    # Check energy threshold
                    if energy > energy_threshold:
                        # Case c-i: Energy above threshold
                        G.nodes[child_node]['collapsedParticleID'] = child_particle_id
                        G.nodes[child_node]['incidentParentID'] = G.nodes[current_node]['incidentParentID']
                    else:
                        # Case c-ii: Energy below threshold
                        G.nodes[child_node]['collapsedParticleID'] = G.nodes[current_node]['collapsedParticleID']
                        G.nodes[child_node]['incidentParentID'] = G.nodes[current_node]['incidentParentID']
                # Case d: Parent in calo, child in tracking
                elif not parent_in_tracking and child_in_tracking:
                    G.nodes[child_node]['collapsedParticleID'] = child_particle_id
                    G.nodes[child_node]['incidentParentID'] = G.nodes[current_node]['incidentParentID']
                
                # Add child to the queue to continue traversal
                queue.append(child_node)
        
        # Check if we processed all nodes
        if len(visited_nodes) < G.number_of_nodes():
            print(f"Warning: Processed only {len(visited_nodes)} out of {G.number_of_nodes()} nodes")
        
        # Update the stored graph
        self._graph = G
        return G
    
    def analyze_particles(self) -> pd.DataFrame:
        """
        Generate statistics about collapsed particles.
        
        Returns:
            pd.DataFrame: DataFrame with information about each collapsed particle.
        """
        # Get the processed graph
        G = self.get_graph()
        particles_df = self._event._particles_df
        
        # Get all unique collapsed particle IDs
        collapsed_ids = set()
        for _, data in G.nodes(data=True):
            if data.get('collapsedParticleID', -1) != -1:
                collapsed_ids.add(data['collapsedParticleID'])
        
        # Prepare a list to store information about each collapsed particle
        collapsed_info = []
        
        # For each collapsed particle ID, collect information
        for c_id in collapsed_ids:
            # Find all nodes that belong to this collapsed particle
            nodes_with_id = [(n, d) for n, d in G.nodes(data=True) 
                             if d.get('collapsedParticleID', -1) == c_id]
            
            if c_id in particles_df.index:
                particle = particles_df.loc[c_id]
                pdg_id = particle['PDG']
                
                # Calculate total energy
                total_energy = sum(d.get('energy', 0) for _, d in nodes_with_id)
                
                # Count number of segments
                num_segments = len(nodes_with_id)
                
                # Calculate path length (approximate)
                path_length = 0
                # Find the primary particle node
                if c_id in G.nodes:
                    # Start position
                    start_x = G.nodes[c_id]['decay_vertex_x']
                    start_y = G.nodes[c_id]['decay_vertex_y']
                    start_z = G.nodes[c_id]['decay_vertex_z']
                    
                    # Sum distances to all child particles
                    for node, _ in nodes_with_id:
                        if node != c_id:
                            end_x = G.nodes[node]['decay_vertex_x']
                            end_y = G.nodes[node]['decay_vertex_y']
                            end_z = G.nodes[node]['decay_vertex_z']
                            
                            dx = end_x - start_x
                            dy = end_y - start_y
                            dz = end_z - start_z
                            segment_length = np.sqrt(dx**2 + dy**2 + dz**2)
                            path_length += segment_length
                
                # Add to the list
                collapsed_info.append({
                    'collapsed_id': c_id,
                    'pdg_id': pdg_id,
                    'total_energy': total_energy,
                    'num_segments': num_segments,
                    'path_length': path_length
                })
        
        # Convert to DataFrame
        self._collapsed_info = pd.DataFrame(collapsed_info)
        return self._collapsed_info
    
    def get_collapsed_info(self) -> pd.DataFrame:
        """
        Get the DataFrame with collapsed particle information, computing it if necessary.
        
        Returns:
            pd.DataFrame: DataFrame with information about collapsed particles.
        """
        if self._collapsed_info is None:
            return self.analyze_particles()
        return self._collapsed_info
    
    def get_particle_descendants(self, particle_id: int) -> List[int]:
        """
        Get all descendant particle IDs for a given particle.
        
        Args:
            particle_id: The particle ID to find descendants for.
            
        Returns:
            List[int]: List of descendant particle IDs.
        """
        G = self.get_graph()
        if particle_id not in G:
            print(f"Warning: Particle {particle_id} not found in decay graph.")
            return []
        
        try:
            return list(nx.descendants(G, particle_id))
        except nx.NetworkXError as e:
            print(f"Error getting descendants for particle {particle_id}: {e}")
            return []
    
    def get_particle_ancestors(self, particle_id: int) -> List[int]:
        """
        Get all ancestor particle IDs for a given particle.
        
        Args:
            particle_id: The particle ID to find ancestors for.
            
        Returns:
            List[int]: List of ancestor particle IDs.
        """
        G = self.get_graph()
        if particle_id not in G:
            print(f"Warning: Particle {particle_id} not found in decay graph.")
            return []
        
        try:
            return list(nx.ancestors(G, particle_id))
        except nx.NetworkXError as e:
            print(f"Error getting ancestors for particle {particle_id}: {e}")
            return []
    
    def get_particles_by_collapsed_id(self, collapsed_id: int) -> List[int]:
        """
        Get all particle IDs that are collapsed into a specific ID.
        
        This is useful for finding all the segments/particles that should
        be treated as a single physical particle in analysis.
        
        Args:
            collapsed_id: The collapsed particle ID to search for.
            
        Returns:
            List[int]: List of particle IDs with this collapsed ID.
        """
        G = self.get_graph()
        return [n for n, d in G.nodes(data=True) 
                if d.get('collapsedParticleID', None) == collapsed_id]
                
    def get_particle_decay_path(self, start_id: int, end_id: int) -> List[int]:
        """
        Get the path of particles from start_id to end_id.
        
        Useful for tracing how a particle decays into another.
        
        Args:
            start_id: The starting particle ID.
            end_id: The ending particle ID.
            
        Returns:
            List[int]: The path of particle IDs, or empty list if no path exists.
        """
        G = self.get_graph()
        if start_id not in G or end_id not in G:
            print(f"Warning: One or both particles ({start_id}, {end_id}) not found in decay graph.")
            return []
        
        try:
            # Check if end_id is a descendant of start_id
            if end_id not in nx.descendants(G, start_id):
                print(f"Warning: Particle {end_id} is not a descendant of {start_id}.")
                return []
                
            # Find shortest path (in a decay tree, there should be only one path)
            return nx.shortest_path(G, start_id, end_id)
        except (nx.NetworkXNoPath, nx.NetworkXError) as e:
            print(f"Error finding path from {start_id} to {end_id}: {e}")
            return []
