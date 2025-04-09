import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns

from edm4hep_utils import load_edm4hep_file, get_simulator_status_bits

def event_diagnostics(file, event_num=0, label=None, detector_params=None):
    event = load_edm4hep_file(file, event_num=event_num)
    tracker_df = event["tracker_df"]
    parents_df = event["parents_df"]
    daughters_df = event["daughters_df"]
    particles_df = event["particles_df"]
    hits_df = event["calo_hits_df"]
    contrib_df = event["calo_contrib_df"]

    particles_df["vr"] = np.sqrt(particles_df.vx**2 + particles_df.vy**2)
    particles_df["endpoint_r"] = np.sqrt(particles_df.endpoint_x**2 + particles_df.endpoint_y**2)
    
    particles_df["created_in_simulation"] = particles_df.simulatorStatus.apply(lambda x: get_simulator_status_bits(x)["created_in_simulation"]).astype(int)
    particles_df["created_inside_tracker"] = ((particles_df.vr < detector_params['tracking_radius']) & 
                                           (particles_df.vz.abs() < detector_params['tracking_z_max'])).astype(int)
    particles_df["ended_inside_tracker"] = ((particles_df.endpoint_r < detector_params['tracking_radius']) & 
                                         (particles_df.endpoint_z.abs() < detector_params['tracking_z_max'])).astype(int)
    # particles_df["backscatter"] = particles_df.simulatorStatus.apply(lambda x: get_simulator_status_bits(x)["backscatter"]).astype(int)
    particles_df["backscatter"] = (particles_df.created_inside_tracker == 0) & (particles_df.ended_inside_tracker == 1)

    particles_with_hits = np.concatenate([contrib_df.particle_id.unique(), tracker_df.particle_id.unique()])
    particles_with_hits = np.unique(particles_with_hits)
    particles_with_hits = particles_df.loc[particles_with_hits]

    # Call the function with the current DataFrame
    plot_particle_distribution(particles_df, particles_with_hits)

    # Check if any particles have vx, vy, vz that are outside the tracking cylinder
    plt.scatter(hits_df.z, hits_df.r, c="red", s=1, label="Calo hits")
    plt.scatter(tracker_df.z, tracker_df.r, c="blue", s=2, label="Tracker hits")

    tracking_radius = detector_params['tracking_radius']
    tracking_z_max = detector_params['tracking_z_max']

    plt.plot([-tracking_z_max, -tracking_z_max, tracking_z_max, tracking_z_max, -tracking_z_max], [0, tracking_radius, tracking_radius, 0, 0], c="black", lw=1, label="Tracking cylinder (conservative outer edge)")
    plt.gca().set_aspect('equal', 'box')
    plt.title(f"Event {event_num}")
    plt.legend()
    plt.show()

    sns.histplot(particles_df.vz, bins = 30, binrange=(-7000, 7000), label="All particles")
    sns.histplot(particles_with_hits.vz, bins = 30, binrange=(-7000, 7000), label="Particles with hits")
    plt.axvline(detector_params['tracking_z_max'], color="black", linestyle="--", lw=1)
    plt.axvline(-detector_params['tracking_z_max'], color="black", linestyle="--", lw=1)
    plt.yscale('log')
    plt.title(f"Production vertex z coordinate for {label}")
    plt.legend()
    plt.show()

    sns.histplot(particles_df.vr, bins = 30, binrange=(0, 2200), label="All particles")
    sns.histplot(particles_with_hits.vr, bins = 30, binrange=(0, 2200), label="Particles with hits")
    plt.axvline(detector_params['tracking_radius'], color="black", linestyle="--", lw=1)
    plt.yscale('log')
    plt.title(f"Production vertex r coordinate for {label}")
    plt.legend()
    plt.show()

    return event

def plot_particle_distribution(particles_df, particles_with_hits):
    """
    Creates a visualization of particle distribution by origin, location, and backscatter status.
    
    Parameters:
    -----------
    particles_with_hits : pandas.DataFrame
        DataFrame containing particle information with columns:
        - created_in_simulation: 0 for generator, 1 for simulation
        - created_outside_tracker: 0 for inside tracker, 1 for outside
        - backscatter: 0 for no backscatter, 1 for backscatter
        
    Returns:
    --------
    None: Displays the plot and prints summary statistics
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np

    # Generate the counts for each combination
    data = {}
    for origin_val, origin_name in [(0, "Generator"), (1, "Simulation")]:
        for loc_val, loc_name in [(1, "Inside"), (0, "Outside")]:
            # Count non-backscatter (0) and backscatter (1) particles
            mask = ((particles_with_hits["created_in_simulation"] == origin_val) & 
                    (particles_with_hits["created_inside_tracker"] == loc_val))
            
            no_bs = ((mask) & (particles_with_hits["backscatter"] == 0)).sum()
            yes_bs = ((mask) & (particles_with_hits["backscatter"] == 1)).sum()
            
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
            ax.text(x + 0.25, y + 0.1, "Is BS", ha='center', va='center', fontsize=8, rotation=45)

            # Put a dot at the center of x,y
            ax.plot(x, y, 'k.', markersize=10)

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
    print(f"Backscatter: {(particles_with_hits['backscatter'] == 1).sum()}")

    print("\nALL PARTICLES")
    print(f"Total particles: {len(particles_df)}")
    print(f"From Generator: {(particles_df['created_in_simulation'] == 0).sum()}")
    print(f"From Simulation: {(particles_df['created_in_simulation'] == 1).sum()}")
    print(f"Inside Tracker: {(particles_df['created_inside_tracker'] == 1).sum()}")
    print(f"Outside Tracker: {(particles_df['created_inside_tracker'] == 0).sum()}")