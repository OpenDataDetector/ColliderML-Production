import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import particle_decay_tree as pdt

class TestParticleDecayTree(unittest.TestCase):
    """Test suite for the particle decay tree implementation"""
    
    def setUp(self):
        """Set up common detector parameters for all tests"""
        self.detector_params = {
            'tracking_radius': 1200,    # in mm
            'tracking_z_max': 3100,     # in mm
            'energy_threshold': 5.0     # in GeV (using a higher value for clearer test cases)
        }
    
    def test_linear_track(self):
        """Test a simple linear track going from tracker to calorimeter"""
        # Create a single particle that travels from origin into calorimeter
        particles_df = pd.DataFrame({
            'PDG': [211],               # Pion
            'generatorStatus': [1],
            'simulatorStatus': [1],
            'charge': [1.0],
            'time': [0.0],
            'mass': [0.13957],
            'vx': [0.0],                # Starting at origin
            'vy': [0.0],
            'vz': [0.0],
            'px': [10.0],               # Momentum in x direction
            'py': [0.0],
            'pz': [0.0],
            'p': [10.0],                # Total momentum
            'endpoint_x': [2000.0],     # Endpoint in calorimeter
            'endpoint_y': [0.0],
            'endpoint_z': [0.0],
            'daughters_begin': [np.nan],
            'daughters_end': [np.nan],
        }, index=[0])                   # Particle ID 0
        
        daughters_df = pd.DataFrame({
            'particle_id': [],
            'collectionID': []
        })
        
        # Process the decay tree
        G, collapsed_info, _, _ = pdt.run_decay_tree_analysis(
            particles_df, daughters_df, 
            tracking_radius=self.detector_params['tracking_radius'],
            tracking_z_max=self.detector_params['tracking_z_max'],
            energy_threshold=self.detector_params['energy_threshold']
        )
        
        # Visualize the result
        self.visualize_test_case(G, self.detector_params, "Linear Track Test")
        
        # Assertions
        self.assertEqual(G.number_of_nodes(), 2)  # Start and end vertices
        self.assertEqual(G.number_of_edges(), 1)  # One track
        
        # Get the edge and check its properties
        edge = list(G.edges(data=True))[0]
        self.assertEqual(edge[2]['collapsedParticleID'], 0)  # Should be the same as particle ID
        self.assertEqual(edge[2]['incidentParentID'], 0)  # For case b: Start in tracking, end in calo
    
    def test_decay_in_tracker(self):
        """Test particle decaying inside the tracking volume"""
        # Parent particle
        parent = pd.DataFrame({
            'PDG': [211],               # Pion
            'generatorStatus': [1],
            'simulatorStatus': [1],
            'charge': [1.0],
            'time': [0.0],
            'mass': [0.13957],
            'vx': [0.0],                # Starting at origin
            'vy': [0.0],
            'vz': [0.0],
            'px': [10.0],
            'py': [0.0],
            'pz': [0.0],
            'p': [10.0],                # Total momentum
            'endpoint_x': [800.0],      # Decay point in tracker
            'endpoint_y': [0.0],
            'endpoint_z': [0.0],
            'daughters_begin': [0],
            'daughters_end': [2],
        }, index=[0])                   # Particle ID 0
        
        # Two daughter particles
        daughter1 = pd.DataFrame({
            'PDG': [211],               # Pion
            'generatorStatus': [1],
            'simulatorStatus': [1],
            'charge': [1.0],
            'time': [0.0],
            'mass': [0.13957],
            'vx': [800.0],              # Starting at decay point
            'vy': [0.0],
            'vz': [0.0],
            'px': [5.0],
            'py': [5.0],
            'pz': [0.0],
            'p': [7.07],                # Total momentum
            'endpoint_x': [1500.0],     # Endpoint in calorimeter
            'endpoint_y': [700.0],
            'endpoint_z': [0.0],
            'daughters_begin': [np.nan],
            'daughters_end': [np.nan],
        }, index=[1])                   # Particle ID 1
        
        daughter2 = pd.DataFrame({
            'PDG': [211],               # Pion
            'generatorStatus': [1],
            'simulatorStatus': [1],
            'charge': [1.0],
            'time': [0.0],
            'mass': [0.13957],
            'vx': [800.0],              # Starting at decay point
            'vy': [0.0],
            'vz': [0.0],
            'px': [5.0],
            'py': [-5.0],
            'pz': [0.0],
            'p': [7.07],                # Total momentum
            'endpoint_x': [1500.0],     # Endpoint in calorimeter
            'endpoint_y': [-700.0],
            'endpoint_z': [0.0],
            'daughters_begin': [np.nan],
            'daughters_end': [np.nan],
        }, index=[2])                   # Particle ID 2
        
        # Combine into a single DataFrame
        particles_df = pd.concat([parent, daughter1, daughter2])
        
        # Create daughters dataframe
        daughters_df = pd.DataFrame({
            'particle_id': [1, 2],      # References to particle IDs 1 and 2
            'collectionID': [0, 0]
        })
        
        # Process the decay tree
        G, collapsed_info, _, _ = pdt.run_decay_tree_analysis(
            particles_df, daughters_df, 
            tracking_radius=self.detector_params['tracking_radius'],
            tracking_z_max=self.detector_params['tracking_z_max'],
            energy_threshold=self.detector_params['energy_threshold']
        )
        
        # Visualize the result
        self.visualize_test_case(G, self.detector_params, "Decay in Tracker Test")
        
        # Assertions
        self.assertEqual(G.number_of_nodes(), 4)  # Origin, decay point, 2 endpoints
        self.assertEqual(G.number_of_edges(), 3)  # Parent track and 2 daughter tracks
        
        # Check parent edge
        parent_edge = None
        for u, v, d in G.edges(data=True):
            if d['particle_id'] == 0:
                parent_edge = (u, v, d)
                break
        
        self.assertIsNotNone(parent_edge)
        self.assertEqual(parent_edge[2]['collapsedParticleID'], 0)
        
        # Check daughter edges
        daughter_edges = []
        for u, v, d in G.edges(data=True):
            if d['particle_id'] in [1, 2]:
                daughter_edges.append((u, v, d))
        
        self.assertEqual(len(daughter_edges), 2)
        
        # Both daughters should maintain their own IDs
        for _, _, d in daughter_edges:
            self.assertIn(d['collapsedParticleID'], [1, 2])
            # For daughters that end in calorimeter
            if d['particle_id'] in [1, 2]:
                self.assertEqual(d['incidentParentID'], d['particle_id'])
    
    def test_decay_in_calo(self):
        """Test particle decaying inside the calorimeter"""
        # Parent particle from origin to calorimeter
        parent = pd.DataFrame({
            'PDG': [211],               # Pion
            'generatorStatus': [1],
            'simulatorStatus': [1],
            'charge': [1.0],
            'time': [0.0],
            'mass': [0.13957],
            'vx': [0.0],                # Starting at origin
            'vy': [0.0],
            'vz': [0.0],
            'px': [15.0],
            'py': [0.0],
            'pz': [0.0],
            'p': [15.0],                # Total momentum
            'endpoint_x': [1500.0],     # Decay point in calorimeter
            'endpoint_y': [0.0],
            'endpoint_z': [0.0],
            'daughters_begin': [0],
            'daughters_end': [2],
        }, index=[0])                   # Particle ID 0
        
        # Two daughter particles - both high energy
        daughter1 = pd.DataFrame({
            'PDG': [22],                # Photon
            'generatorStatus': [1],
            'simulatorStatus': [1],
            'charge': [0.0],
            'time': [0.0],
            'mass': [0.0],
            'vx': [1500.0],             # Starting at decay point in calo
            'vy': [0.0],
            'vz': [0.0],
            'px': [7.0],
            'py': [1.0],
            'pz': [0.0],
            'p': [7.07],                # Total momentum (above threshold)
            'endpoint_x': [2500.0],     # Further in calorimeter
            'endpoint_y': [1000.0],
            'endpoint_z': [0.0],
            'daughters_begin': [np.nan],
            'daughters_end': [np.nan],
        }, index=[1])                   # Particle ID 1
        
        daughter2 = pd.DataFrame({
            'PDG': [22],                # Photon
            'generatorStatus': [1],
            'simulatorStatus': [1],
            'charge': [0.0],
            'time': [0.0],
            'mass': [0.0],
            'vx': [1500.0],             # Starting at decay point in calo
            'vy': [0.0],
            'vz': [0.0],
            'px': [7.0],
            'py': [-1.0],
            'pz': [0.0],
            'p': [7.07],                # Total momentum (above threshold)
            'endpoint_x': [2500.0],     # Further in calorimeter
            'endpoint_y': [-1000.0],
            'endpoint_z': [0.0],
            'daughters_begin': [np.nan],
            'daughters_end': [np.nan],
        }, index=[2])                   # Particle ID 2
        
        # Combine into a single DataFrame
        particles_df = pd.concat([parent, daughter1, daughter2])
        
        # Create daughters dataframe
        daughters_df = pd.DataFrame({
            'particle_id': [1, 2],      # References to particle IDs 1 and 2
            'collectionID': [0, 0]
        })
        
        # Process the decay tree
        G, collapsed_info, _, _ = pdt.run_decay_tree_analysis(
            particles_df, daughters_df, 
            tracking_radius=self.detector_params['tracking_radius'],
            tracking_z_max=self.detector_params['tracking_z_max'],
            energy_threshold=self.detector_params['energy_threshold']
        )
        
        # Visualize the result
        self.visualize_test_case(G, self.detector_params, "Decay in Calorimeter Test")
        
        # Assertions
        self.assertEqual(G.number_of_nodes(), 4)  # Origin, decay point, 2 endpoints
        self.assertEqual(G.number_of_edges(), 3)  # Parent track and 2 daughter tracks
        
        # Check parent edge (should have its own ID)
        parent_edge = None
        for u, v, d in G.edges(data=True):
            if d['particle_id'] == 0:
                parent_edge = (u, v, d)
                break
        
        self.assertIsNotNone(parent_edge)
        self.assertEqual(parent_edge[2]['collapsedParticleID'], 0)
        self.assertEqual(parent_edge[2]['incidentParentID'], 0)  # For case b: Start in tracking, end in calo
        
        # Check daughter edges (both high energy, should have unique IDs)
        daughter_edges = []
        for u, v, d in G.edges(data=True):
            if d['particle_id'] in [1, 2]:
                daughter_edges.append((u, v, d))
        
        self.assertEqual(len(daughter_edges), 2)
        
        # Both daughters are high energy (above threshold) so should maintain unique IDs
        for _, _, d in daughter_edges:
            self.assertEqual(d['collapsedParticleID'], d['particle_id'])
            self.assertEqual(d['incidentParentID'], 0)  # Should point to parent
    
    def test_low_energy_calo(self):
        """Test low energy decay in calorimeter (below threshold)"""
        # Parent particle from origin to calorimeter
        parent = pd.DataFrame({
            'PDG': [211],               # Pion
            'generatorStatus': [1],
            'simulatorStatus': [1],
            'charge': [1.0],
            'time': [0.0],
            'mass': [0.13957],
            'vx': [0.0],                # Starting at origin
            'vy': [0.0],
            'vz': [0.0],
            'px': [15.0],
            'py': [0.0],
            'pz': [0.0],
            'p': [15.0],                # Total momentum
            'endpoint_x': [1500.0],     # Decay point in calorimeter
            'endpoint_y': [0.0],
            'endpoint_z': [0.0],
            'daughters_begin': [0],
            'daughters_end': [2],
        }, index=[0])                   # Particle ID 0
        
        # Two daughter particles - both low energy (below threshold)
        daughter1 = pd.DataFrame({
            'PDG': [22],                # Photon
            'generatorStatus': [1],
            'simulatorStatus': [1],
            'charge': [0.0],
            'time': [0.0],
            'mass': [0.0],
            'vx': [1500.0],             # Starting at decay point in calo
            'vy': [0.0],
            'vz': [0.0],
            'px': [2.0],
            'py': [1.0],
            'pz': [0.0],
            'p': [2.24],                # Total momentum (below threshold)
            'endpoint_x': [2500.0],     # Further in calorimeter
            'endpoint_y': [1000.0],
            'endpoint_z': [0.0],
            'daughters_begin': [np.nan],
            'daughters_end': [np.nan],
        }, index=[1])                   # Particle ID 1
        
        daughter2 = pd.DataFrame({
            'PDG': [22],                # Photon
            'generatorStatus': [1],
            'simulatorStatus': [1],
            'charge': [0.0],
            'time': [0.0],
            'mass': [0.0],
            'vx': [1500.0],             # Starting at decay point in calo
            'vy': [0.0],
            'vz': [0.0],
            'px': [2.0],
            'py': [-1.0],
            'pz': [0.0],
            'p': [2.24],                # Total momentum (below threshold)
            'endpoint_x': [2500.0],     # Further in calorimeter
            'endpoint_y': [-1000.0],
            'endpoint_z': [0.0],
            'daughters_begin': [np.nan],
            'daughters_end': [np.nan],
        }, index=[2])                   # Particle ID 2
        
        # Combine into a single DataFrame
        particles_df = pd.concat([parent, daughter1, daughter2])
        
        # Create daughters dataframe
        daughters_df = pd.DataFrame({
            'particle_id': [1, 2],      # References to particle IDs 1 and 2
            'collectionID': [0, 0]
        })
        
        # Process the decay tree
        G, collapsed_info, _, _ = pdt.run_decay_tree_analysis(
            particles_df, daughters_df, 
            tracking_radius=self.detector_params['tracking_radius'],
            tracking_z_max=self.detector_params['tracking_z_max'],
            energy_threshold=self.detector_params['energy_threshold']
        )
        
        # Visualize the result
        self.visualize_test_case(G, self.detector_params, "Low Energy Calorimeter Test")
        
        # Assertions
        self.assertEqual(G.number_of_nodes(), 4)  # Origin, decay point, 2 endpoints
        self.assertEqual(G.number_of_edges(), 3)  # Parent track and 2 daughter tracks
        
        # Check parent edge
        parent_edge = None
        for u, v, d in G.edges(data=True):
            if d['particle_id'] == 0:
                parent_edge = (u, v, d)
                break
        
        self.assertIsNotNone(parent_edge)
        self.assertEqual(parent_edge[2]['collapsedParticleID'], 0)
        
        # Check daughter edges (both low energy, should inherit parent ID)
        daughter_edges = []
        for u, v, d in G.edges(data=True):
            if d['particle_id'] in [1, 2]:
                daughter_edges.append((u, v, d))
        
        self.assertEqual(len(daughter_edges), 2)
        
        # Both daughters should inherit the parent's collapsed ID
        for _, _, d in daughter_edges:
            self.assertEqual(d['collapsedParticleID'], 0)  # Inherits parent ID
            self.assertEqual(d['incidentParentID'], d['particle_id'])  # Points to itself
    
    def test_cascade_decay(self):
        """Test a cascade decay with multiple generations"""
        # First generation particle from origin to tracker decay point
        particle1 = pd.DataFrame({
            'PDG': [211],               # Pion
            'generatorStatus': [1],
            'simulatorStatus': [1],
            'charge': [1.0],
            'time': [0.0],
            'mass': [0.13957],
            'vx': [0.0],                # Starting at origin
            'vy': [0.0],
            'vz': [0.0],
            'px': [10.0],
            'py': [0.0],
            'pz': [0.0],
            'p': [10.0],                # Total momentum
            'endpoint_x': [500.0],      # Decay point in tracker
            'endpoint_y': [0.0],
            'endpoint_z': [0.0],
            'daughters_begin': [0],
            'daughters_end': [1],
        }, index=[0])                   # Particle ID 0
        
        # Second generation particle from tracker to calorimeter
        particle2 = pd.DataFrame({
            'PDG': [211],               # Pion
            'generatorStatus': [1],
            'simulatorStatus': [1],
            'charge': [1.0],
            'time': [0.0],
            'mass': [0.13957],
            'vx': [500.0],              # Starting at first decay point
            'vy': [0.0],
            'vz': [0.0],
            'px': [10.0],
            'py': [1.0],
            'pz': [0.0],
            'p': [10.05],               # Total momentum
            'endpoint_x': [1500.0],     # Decay point in calorimeter
            'endpoint_y': [100.0],
            'endpoint_z': [0.0],
            'daughters_begin': [1],
            'daughters_end': [3],
        }, index=[1])                   # Particle ID 1
        
        # Third generation particles from calo decay point - one high energy, one low
        particle3 = pd.DataFrame({
            'PDG': [22],                # Photon
            'generatorStatus': [1],
            'simulatorStatus': [1],
            'charge': [0.0],
            'time': [0.0],
            'mass': [0.0],
            'vx': [1500.0],             # Starting at calo decay point
            'vy': [100.0],
            'vz': [0.0],
            'px': [8.0],
            'py': [0.0],
            'pz': [0.0],
            'p': [8.0],                 # Total momentum (high energy)
            'endpoint_x': [2500.0],     # Further in calorimeter
            'endpoint_y': [100.0],
            'endpoint_z': [0.0],
            'daughters_begin': [np.nan],
            'daughters_end': [np.nan],
        }, index=[2])                   # Particle ID 2
        
        particle4 = pd.DataFrame({
            'PDG': [22],                # Photon
            'generatorStatus': [1],
            'simulatorStatus': [1],
            'charge': [0.0],
            'time': [0.0],
            'mass': [0.0],
            'vx': [1500.0],             # Starting at calo decay point
            'vy': [100.0],
            'vz': [0.0],
            'px': [2.0],
            'py': [0.0],
            'pz': [0.0],
            'p': [2.0],                 # Total momentum (low energy)
            'endpoint_x': [1800.0],     # Further in calorimeter
            'endpoint_y': [100.0],
            'endpoint_z': [0.0],
            'daughters_begin': [np.nan],
            'daughters_end': [np.nan],
        }, index=[3])                   # Particle ID 3
        
        # Combine into a single DataFrame
        particles_df = pd.concat([particle1, particle2, particle3, particle4])
        
        # Create daughters dataframe
        daughters_df = pd.DataFrame({
            'particle_id': [1, 2, 3],  # References to particle IDs
            'collectionID': [0, 0, 0]
        })
        
        # Process the decay tree
        G, collapsed_info, _, _ = pdt.run_decay_tree_analysis(
            particles_df, daughters_df, 
            tracking_radius=self.detector_params['tracking_radius'],
            tracking_z_max=self.detector_params['tracking_z_max'],
            energy_threshold=self.detector_params['energy_threshold']
        )
        
        # Visualize the result
        self.visualize_test_case(G, self.detector_params, "Cascade Decay Test")
        
        # Assertions
        self.assertEqual(G.number_of_nodes(), 5)  # Origin, 2 decay points, 2 endpoints
        self.assertEqual(G.number_of_edges(), 4)  # 4 particle tracks
        
        # First and second gen particles should have their own IDs
        for u, v, d in G.edges(data=True):
            if d['particle_id'] in [0, 1]:
                self.assertEqual(d['collapsedParticleID'], d['particle_id'])
        
        # Third gen - high energy should have its own ID
        for u, v, d in G.edges(data=True):
            if d['particle_id'] == 2:  # High energy photon
                self.assertEqual(d['collapsedParticleID'], 2)
                self.assertEqual(d['incidentParentID'], 1)  # Points to parent
        
        # Third gen - low energy should inherit parent ID
        for u, v, d in G.edges(data=True):
            if d['particle_id'] == 3:  # Low energy photon
                self.assertEqual(d['collapsedParticleID'], 1)  # Inherits parent ID
                self.assertEqual(d['incidentParentID'], 3)  # Points to itself
    
    def visualize_test_case(self, G, detector_params, title, show_particle_ids=True):
        """Enhanced visualization of test cases with particle IDs on edges"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw edges first (particles)
        for start, end, data in G.edges(data=True):
            start_x = G.nodes[start]['x']
            start_y = G.nodes[start]['y']
            start_z = G.nodes[start]['z']
            
            end_x = G.nodes[end]['x']
            end_y = G.nodes[end]['y']
            end_z = G.nodes[end]['z']
            
            # Color by collapsed particle ID
            collapsed_id = data['collapsedParticleID']
            # Default color by collapsed ID (modulo to keep the color range manageable)
            color = f"C{abs(collapsed_id) % 10}" if collapsed_id != -1 else 'gray'
            
            # Draw the edge
            ax.plot([start_x, end_x], [start_y, end_y], [start_z, end_z], color=color, lw=2.0, alpha=0.7)
            
            # Add particle ID label near the middle of the edge
            if show_particle_ids:
                mid_x = (start_x + end_x) / 2
                mid_y = (start_y + end_y) / 2
                mid_z = (start_z + end_z) / 2
                
                # Create text labels
                particle_id = data['particle_id']
                collapsed_id = data['collapsedParticleID']
                label = f"P:{particle_id}\nC:{collapsed_id}"
                
                # Position the label with a small offset
                offset = 50  # offset in mm
                ax.text(mid_x + offset, mid_y + offset, mid_z, label, fontsize=9)
        
        # Draw nodes (vertices)
        for node, data in G.nodes(data=True):
            x, y, z = data['x'], data['y'], data['z']
            is_in_tracking = pdt.in_tracking_cylinder(x, y, z, detector_params)
            color = 'blue' if is_in_tracking else 'red'
            ax.scatter(x, y, z, c=color, s=100, alpha=0.7, edgecolors='black')
            
            # Add vertex info
            if show_particle_ids:
                collapsed_id = data['collapsedParticleID']
                v_label = f"C:{collapsed_id}" if collapsed_id != -1 else "ROOT"
                ax.text(x + 50, y, z + 50, v_label, fontsize=8)
        
        # Draw the tracking cylinder
        tracking_radius = detector_params['tracking_radius']
        tracking_z_max = detector_params['tracking_z_max']
        
        # Create points for cylinder
        theta = np.linspace(0, 2*np.pi, 50)
        z = np.linspace(-tracking_z_max, tracking_z_max, 2)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = tracking_radius * np.cos(theta_grid)
        y_grid = tracking_radius * np.sin(theta_grid)
        
        # Draw cylinder surface
        ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.1, color='blue')
        
        # Set axis limits to match the scale of the test data
        max_x = max([abs(data['x']) for _, data in G.nodes(data=True)]) * 1.2
        max_y = max([abs(data['y']) for _, data in G.nodes(data=True)]) * 1.2
        max_z = max([abs(data['z']) for _, data in G.nodes(data=True)]) * 1.2 or tracking_z_max * 0.5
        
        ax.set_xlim(-max_x, max_x)
        ax.set_ylim(-max_y, max_y)
        ax.set_zlim(-max_z, max_z)
        
        # Labels and title
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        ax.set_title(title)
        
        # Add legend for tracking and calorimeter
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], color='blue', lw=4, linestyle='none', marker='o'),
            Line2D([0], [0], color='red', lw=4, linestyle='none', marker='o'),
            Line2D([0], [0], color='gray', lw=2),
        ]
        
        ax.legend(custom_lines, ['Tracking Region', 'Calorimeter Region', 'Particle Path'], 
                  loc='upper left', frameon=True, framealpha=0.9)
        
        plt.tight_layout()
        plt.show()
        
        return ax

if __name__ == "__main__":
    # Run all tests
    unittest.main()
    
    # Alternatively, uncomment to run specific tests:
    # test = TestParticleDecayTree()
    # test.setUp()
    # test.test_linear_track()
    # test.test_decay_in_tracker()
    # test.test_decay_in_calo()
    # test.test_low_energy_calo()
    # test.test_cascade_decay() 