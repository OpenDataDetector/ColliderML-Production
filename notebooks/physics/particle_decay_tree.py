import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import plotly.graph_objects as go

def build_decay_tree(particles_df, daughters_df):
    """
    Build a decay tree graph where:
    - Nodes are particles
    - Edges represent parent-child relationships
    
    Parameters:
    - particles_df: DataFrame containing particle information
    - daughters_df: DataFrame containing daughter-parent relationships
    
    Returns:
    - G: NetworkX DiGraph representing the decay tree
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add all particles as nodes first - vectorized approach
    print("Creating nodes...")

    nodes_df = particles_df.copy()
    # Convert DataFrame to dict of dicts for faster node creation
    nodes_df["particleID"] = nodes_df.index
    nodes_df["collapsedParticleID"] = nodes_df.index
    nodes_df["incidentParentID"] = nodes_df.index
    nodes_df.rename(columns={"p": "energy", "vx": "decay_vertex_x", "vy": "decay_vertex_y", "vz": "decay_vertex_z"}, inplace=True)
    node_attrs = nodes_df[['energy', 'decay_vertex_x', 'decay_vertex_y', 'decay_vertex_z', 'particleID', 'collapsedParticleID', 'incidentParentID']].to_dict('index')
    
    # Add all nodes at once
    G.add_nodes_from(node_attrs.items())
    
    num_daughters = (particles_df["daughters_end"] - particles_df["daughters_begin"]).values
    daughters_df["parent_id"] = np.repeat(nodes_df.particleID.values, num_daughters)
    
    # Add all edges at once
    G.add_edges_from(daughters_df[["parent_id", "particle_id"]].values)
    print(f"Created {G.number_of_edges()} edges")
    
    return G

def in_tracking_cylinder(x, y, z, params):
    """
    Check if a point is inside the tracking cylinder
    
    Parameters:
    - x, y, z: Coordinates of the point
    - params: Dictionary with tracking_radius and tracking_z_max
    
    Returns:
    - Boolean: True if point is inside tracking cylinder
    """
    r = np.sqrt(x**2 + y**2)
    return r < params['tracking_radius'] and abs(z) < params['tracking_z_max']

def process_decay_tree(G, detector_params):
    """
    Walk through the decay tree and assign collapsedParticleID and incidentParentID
    
    Parameters:
    - G: NetworkX DiGraph representing the decay tree (particles as nodes)
    - detector_params: Dictionary with detector dimensions
    
    Returns:
    - G: Updated NetworkX DiGraph
    """
    print("Finding root nodes...")
    root_nodes = [node for node, in_degree in G.in_degree() if in_degree == 0]
    print(f"Found {len(root_nodes)} root nodes")
    
    # Queue for breadth-first traversal
    queue = root_nodes.copy()
    visited_nodes = set()

    print("Processing nodes in topological order...")
    # Process the graph breadth-first
    with tqdm(total=G.number_of_nodes()) as pbar:
        while queue:
            current_node = queue.pop(0)
            # print("Current node: ", current_node)

            # Skip if already visited
            if current_node in visited_nodes:
                continue
                
            visited_nodes.add(current_node)
            pbar.update(1)
            
            # Get node position
            x = G.nodes[current_node]['decay_vertex_x']
            y = G.nodes[current_node]['decay_vertex_y']
            z = G.nodes[current_node]['decay_vertex_z']

            # Get particle ID
            parent_particle_id = G.nodes[current_node]['particleID']
            # print("Parent particle ID: ", parent_particle_id)
            # Check if particle is in tracking cylinder
            parent_in_tracking = in_tracking_cylinder(x, y, z, detector_params)
            # print("Parent in tracking: ", parent_in_tracking)
            
            # Process each outgoing edge
            for _, child_node in G.out_edges(current_node, data=False):
                child_x = G.nodes[child_node]['decay_vertex_x']
                child_y = G.nodes[child_node]['decay_vertex_y']
                child_z = G.nodes[child_node]['decay_vertex_z']

                # Get child particle ID
                child_particle_id = child_node
                # print("Child particle ID: ", child_particle_id)
                child_in_tracking = in_tracking_cylinder(child_x, child_y, child_z, detector_params)
                # print("Child in tracking: ", child_in_tracking)
                
                # Case a: Parent and child in tracking cylinder
                if parent_in_tracking and child_in_tracking:
                    G.nodes[child_node]['collapsedParticleID'] = child_particle_id
                    # print("Child collapsed particle ID: ", child_particle_id)
                # Case b: Parent in tracking, child in calo
                elif parent_in_tracking and not child_in_tracking:
                    energy = G.nodes[child_node]['energy']
                    # Check energy threshold
                    if energy > detector_params['energy_threshold']:
                        # print("Above energy threshold")
                        G.nodes[child_node]['collapsedParticleID'] = child_particle_id
                        G.nodes[child_node]['incidentParentID'] = parent_particle_id
                        # print("Child collapsed particle ID: ", child_particle_id)
                        # print("Child incident parent ID: ", parent_particle_id)
                    else:
                        # print("Below energy threshold")
                        G.nodes[child_node]['collapsedParticleID'] = parent_particle_id
                        G.nodes[child_node]['incidentParentID'] = parent_particle_id
                        # print("Child collapsed particle ID: ", parent_particle_id)
                        # print("Child incident parent ID: ", parent_particle_id)
                # Case c: Parent in calo, child in calo
                elif not parent_in_tracking and not child_in_tracking:
                    energy = G.nodes[child_node]['energy']
                    # Check energy threshold
                    if energy > detector_params['energy_threshold']:
                        # print("Above energy threshold")
                        # Case c-i: Energy above threshold
                        G.nodes[child_node]['collapsedParticleID'] = child_particle_id
                        G.nodes[child_node]['incidentParentID'] = G.nodes[current_node]['incidentParentID']
                        # print("Child collapsed particle ID: ", child_particle_id)
                        # print("Child incident parent ID: ", G.nodes[current_node]['incidentParentID'])
                    else:
                        # print("Below energy threshold")
                        # Case c-ii: Energy below threshold
                        G.nodes[child_node]['collapsedParticleID'] = G.nodes[current_node]['collapsedParticleID']
                        G.nodes[child_node]['incidentParentID'] = G.nodes[current_node]['incidentParentID']
                        # print("Child collapsed particle ID: ", G.nodes[current_node]['collapsedParticleID'])
                # Case d: Parent in calo, child in tracking
                elif not parent_in_tracking and child_in_tracking:
                    G.nodes[child_node]['collapsedParticleID'] = child_particle_id
                    G.nodes[child_node]['incidentParentID'] = G.nodes[current_node]['incidentParentID']
                    # print("Child collapsed particle ID: ", child_particle_id)
                    # print("Child incident parent ID: ", G.nodes[current_node]['incidentParentID'])
                
                # Add child to the queue to continue traversal
                queue.append(child_node)
    
    # Check if we processed all nodes
    if len(visited_nodes) < G.number_of_nodes():
        print(f"Warning: Processed only {len(visited_nodes)} out of {G.number_of_nodes()} nodes")
    
    return G

def visualize_decay_tree(G, detector_params, highlight_collapsed=None, show_tracking_cylinder=False):
    """
    Create a 3D visualization of the particle decay tree
    
    Parameters:
    - G: NetworkX DiGraph representing the decay tree (particles as nodes)
    - detector_params: Dictionary with detector dimensions
    - highlight_collapsed: Optional ID to highlight all particles with this collapsedParticleID
    
    Returns:
    - ax: Matplotlib axis object with the plot
    """
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define edge colors based on detector regions
    edge_colors = {
        'tracker->tracker': 'blue',
        'tracker->calo': 'green',
        'calo->tracker': 'orange',
        'calo->calo': 'red',
    }
    
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
        parent_in_tracking = in_tracking_cylinder(parent_x, parent_y, parent_z, detector_params)
        child_in_tracking = in_tracking_cylinder(child_x, child_y, child_z, detector_params)
        
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
                color=edge_color, lw=linewidth, alpha=alpha, label=edge_type if parent == list(G.nodes())[0] else "")
    
    # Draw nodes (particles)
    for node, data in G.nodes(data=True):
        x, y, z = data['decay_vertex_x'], data['decay_vertex_y'], data['decay_vertex_z']
        is_in_tracking = in_tracking_cylinder(x, y, z, detector_params)
        
        # Color based on tracking cylinder location
        color = 'blue' if is_in_tracking else 'red'
        
        # Size based on energy (optional)
        energy = data.get('energy', 0)
        size = min(30, max(10, energy / 5)) if energy else 20
        
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
    
    # Optional: Draw decay vertices as small points
    decay_vertices = set()
    for _, _, data in G.edges(data=True):
        if 'decay_vertex_x' in data and 'decay_vertex_y' in data and 'decay_vertex_z' in data:
            decay_vertices.add((data['decay_vertex_x'], data['decay_vertex_y'], data['decay_vertex_z']))
    
    for vx, vy, vz in decay_vertices:
        is_in_tracking = in_tracking_cylinder(vx, vy, vz, detector_params)
        color = 'cyan' if is_in_tracking else 'orange'
        ax.scatter(vx, vy, vz, c=color, marker='x', s=15, alpha=0.5)
    
    # Draw the tracking cylinder
    if show_tracking_cylinder:
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
    
    # Add legend
    # For particles
    ax.scatter([], [], c='blue', s=20, label='Particle in Tracking')
    ax.scatter([], [], c='red', s=20, label='Particle in Calorimeter')
    
    # For decay vertices
    ax.scatter([], [], c='cyan', marker='x', s=15, label='Decay vertex in Tracking')
    ax.scatter([], [], c='orange', marker='x', s=15, label='Decay vertex in Calorimeter')
    
    # For edge types
    for edge_type, color in edge_colors.items():
        ax.plot([], [], color=color, linewidth=2, label=edge_type)
    
    # If highlighting
    if highlight_collapsed is not None:
        ax.plot([], [], color='magenta', linewidth=2.5, label=f'Highlighted (ID: {highlight_collapsed})')
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    ax.set_title('Particle Decay Tree in Detector')
    
    # Set axis limits based on particle positions
    all_coords = []
    for _, data in G.nodes(data=True):
        all_coords.append((abs(data['decay_vertex_x']), abs(data['decay_vertex_y']), abs(data['decay_vertex_z'])))
    if all_coords:
        max_coord = max([max(x, y, z) for x, y, z in all_coords])
        ax.set_xlim(-max_coord*1.1, max_coord*1.1)
        ax.set_ylim(-max_coord*1.1, max_coord*1.1)
        ax.set_zlim(-max_coord*1.1, max_coord*1.1)
    
    plt.tight_layout()
    
    return ax

def analyze_particle_flow(G, particles_df):
    """
    Generate statistics about collapsed particles
    
    Parameters:
    - G: NetworkX DiGraph after processing (particles as nodes)
    - particles_df: DataFrame with particle information
    
    Returns:
    - collapsed_info: DataFrame with information about each collapsed particle
    """
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
    return pd.DataFrame(collapsed_info)

# Main function to tie everything together
def run_decay_tree_analysis(particles_df, daughters_df, tracking_radius=1200, tracking_z_max=3100, energy_threshold=0.0):
    """
    Complete pipeline to build and analyze a particle decay tree
    
    Parameters:
    - particles_df: DataFrame containing particle information
    - daughters_df: DataFrame containing daughter-parent relationships
    - tracking_radius: Radius of tracking cylinder in mm
    - tracking_z_max: Z-extent of tracking cylinder in mm
    - energy_threshold: Energy threshold in GeV for case c
    
    Returns:
    - G: Processed NetworkX DiGraph 
    - collapsed_info: DataFrame with statistics about collapsed particles
    - vertex_map: Dictionary mapping vertex coordinates to vertex IDs
    - detector_params: Dictionary with detector settings
    """
    # Define simplified detector parameters
    detector_params = {
        'tracking_radius': tracking_radius,    # in mm
        'tracking_z_max': tracking_z_max,      # in mm
        'energy_threshold': energy_threshold   # in GeV
    }
    
    # Build the decay tree
    G = build_decay_tree(particles_df, daughters_df)
    print(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Process the decay tree to assign collapsed particle IDs
    G = process_decay_tree(G, detector_params)
    
    # Analyze the particle flow
    collapsed_info = analyze_particle_flow(G, particles_df)
    print(f"Found {len(collapsed_info)} collapsed particles")
    
    return G, collapsed_info, detector_params

def visualize_highlights(G, detector_params, collapsed_info, n_particles=3):
    """
    Create individual visualizations highlighting specific particles
    
    Parameters:
    - G: NetworkX DiGraph after processing
    - detector_params: Dictionary with detector dimensions
    - collapsed_info: DataFrame with collapsed particle information
    - n_particles: Number of top particles to highlight
    
    Returns:
    - None, displays the plots
    """
    if len(collapsed_info) == 0:
        print("No collapsed particles to highlight")
        return
    
    # Sort by total energy
    top_particles = collapsed_info.sort_values('total_energy', ascending=False).head(n_particles)
    
    for _, row in top_particles.iterrows():
        c_id = row['collapsed_id']
        pdg_id = row['pdg_id']
        energy = row['total_energy']
        
        print(f"Highlighting particle ID {c_id} (PDG: {pdg_id}, Energy: {energy:.2f} GeV)")
        ax = visualize_decay_tree(G, detector_params, highlight_collapsed=c_id)
        plt.title(f"Particle ID {c_id} (PDG: {pdg_id}, Energy: {energy:.2f} GeV)")
        plt.show()

def interactive_decay_tree(G, detector_params, highlight_collapsed=None, show_tracking_cylinder=False, height=1000, width=1000):
    """
    Create an interactive 3D visualization of the particle decay tree using Plotly
    
    Parameters:
    - G: NetworkX DiGraph representing the decay tree (particles as nodes)
    - detector_params: Dictionary with detector dimensions
    - highlight_collapsed: Optional ID to highlight all particles with this collapsedParticleID
    
    Returns:
    - fig: Plotly figure object for the interactive plot
    """
    # Define edge colors based on detector regions
    edge_colors = {
        'tracker->tracker': 'blue',
        'tracker->calo': 'green',
        'calo->tracker': 'orange',
        'calo->calo': 'red',
    }
    
    # Create a figure
    fig = go.Figure()
    
    # Add edges (decay relationships)
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
        parent_in_tracking = in_tracking_cylinder(parent_x, parent_y, parent_z, detector_params)
        child_in_tracking = in_tracking_cylinder(child_x, child_y, child_z, detector_params)
        
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
                linewidth = 4
                opacity = 1.0
                # Use a distinctive color that's different from the edge type colors
                edge_color = 'magenta'
            else:
                linewidth = 2
                opacity = 0.6
        else:
            linewidth = 2
            opacity = 0.6
        
        # Create line for the decay relationship
        fig.add_trace(go.Scatter3d(
            x=[parent_x, child_x],
            y=[parent_y, child_y],
            z=[parent_z, child_z],
            mode='lines',
            line=dict(color=edge_color, width=linewidth),
            opacity=opacity,
            hoverinfo='text',
            hovertext=f'Edge Type: {edge_type}<br>Parent ID: {parent}<br>Child ID: {child}',
            showlegend=False
        ))
    
    # Add nodes (particles) - track nodes
    tracking_nodes_x = []
    tracking_nodes_y = []
    tracking_nodes_z = []
    tracking_nodes_size = []
    tracking_nodes_text = []
    
    # Calo nodes
    calo_nodes_x = []
    calo_nodes_y = []
    calo_nodes_z = []
    calo_nodes_size = []
    calo_nodes_text = []
    
    # Group nodes by detector region for cleaner plotting
    for node, data in G.nodes(data=True):
        x, y, z = data['decay_vertex_x'], data['decay_vertex_y'], data['decay_vertex_z']
        is_in_tracking = in_tracking_cylinder(x, y, z, detector_params)
        
        # Size based on energy (optional)
        energy = data.get('energy', 0)
        size = min(10, max(5, energy / 10)) if energy else 5
        
        # Text for hover information
        particle_id = data.get('particleID', 'N/A')
        collapsed_id = data.get('collapsedParticleID', -1)
        incident_id = data.get('incidentParentID', -1)
        
        hover_text = (
            f"ID: {particle_id}<br>"
            f"CollapsedID: {collapsed_id}<br>"
            f"IncidentID: {incident_id}<br>"
            f"Energy: {energy:.2f} GeV<br>"
            f"Position: ({x:.1f}, {y:.1f}, {z:.1f})"
        )
        
        if is_in_tracking:
            tracking_nodes_x.append(x)
            tracking_nodes_y.append(y)
            tracking_nodes_z.append(z)
            tracking_nodes_size.append(size)
            tracking_nodes_text.append(hover_text)
        else:
            calo_nodes_x.append(x)
            calo_nodes_y.append(y)
            calo_nodes_z.append(z)
            calo_nodes_size.append(size)
            calo_nodes_text.append(hover_text)
    
    # Add tracking nodes
    if tracking_nodes_x:
        fig.add_trace(go.Scatter3d(
            x=tracking_nodes_x,
            y=tracking_nodes_y,
            z=tracking_nodes_z,
            mode='markers',
            marker=dict(
                size=tracking_nodes_size,
                color='blue',
                opacity=0.8
            ),
            text=tracking_nodes_text,
            hoverinfo='text',
            name='Particles in Tracking'
        ))
    
    # Add calorimeter nodes
    if calo_nodes_x:
        fig.add_trace(go.Scatter3d(
            x=calo_nodes_x,
            y=calo_nodes_y,
            z=calo_nodes_z,
            mode='markers',
            marker=dict(
                size=calo_nodes_size,
                color='red',
                opacity=0.8
            ),
            text=calo_nodes_text,
            hoverinfo='text',
            name='Particles in Calorimeter'
        ))
    
    # Draw the tracking cylinder
    if show_tracking_cylinder:
        tracking_radius = detector_params['tracking_radius']
        tracking_z_max = detector_params['tracking_z_max']
        
        # Create points for cylinder
        theta = np.linspace(0, 2*np.pi, 50)
        z = np.linspace(-tracking_z_max, tracking_z_max, 20)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = tracking_radius * np.cos(theta_grid)
        y_grid = tracking_radius * np.sin(theta_grid)
        
        # Create the cylindrical surface
        fig.add_trace(go.Surface(
            x=x_grid,
            y=y_grid,
            z=z_grid,
            colorscale=[[0, 'lightblue'], [1, 'lightblue']],
            opacity=0.2,
            showscale=False,
            name='Tracking Cylinder'
        ))
    
    # Add edge type examples to legend
    for edge_type, color in edge_colors.items():
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='lines',
            line=dict(color=color, width=3),
            name=edge_type
        ))
    
    # If highlighting, add to legend
    if highlight_collapsed is not None:
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='lines',
            line=dict(color='magenta', width=4),
            name=f'Highlighted (ID: {highlight_collapsed})'
        ))
    
    # Set layout
    max_coord = 0
    for _, data in G.nodes(data=True):
        max_coord = max(max_coord, abs(data['decay_vertex_x']), abs(data['decay_vertex_y']), abs(data['decay_vertex_z']))
    
    max_coord = max_coord * 1.2  # Add some margin
    
    fig.update_layout(
        title='Interactive Particle Decay Tree',
        scene=dict(
            xaxis=dict(title='X [mm]', range=[-max_coord, max_coord]),
            yaxis=dict(title='Y [mm]', range=[-max_coord, max_coord]),
            zaxis=dict(title='Z [mm]', range=[-max_coord, max_coord]),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=height,  # Make the plot taller
        width=width,  # Make the plot wider
        legend=dict(
            x=0.01,
            y=0.99,
            traceorder='normal',
            font=dict(size=10),
            bgcolor='rgba(255, 255, 255, 0.7)'
        ),
        hovermode='closest'
    )
    
    return fig 