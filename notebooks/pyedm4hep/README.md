# PyEDM4hep Library Plan

This document outlines the plan for a Python library, `pyedm4hep`, designed to provide an object-oriented interface to EDM4hep data while leveraging the performance of Pandas DataFrames for bulk operations.

## Goal

Provide a user-friendly and efficient way to interact with EDM4hep data (typically stored in ROOT files) within Python environments like Jupyter notebooks. The library should offer the convenience of object-oriented navigation (e.g., accessing particle daughters, hit positions) without sacrificing the speed of vectorized operations on large datasets provided by libraries like Pandas and NumPy.

## Usage Examples

```python
import matplotlib.pyplot as plt
from pyedm4hep import EDM4hepEvent

# --- Define Detector Geometry ---
detector_params = {
    'tracking_radius': 1100.0, # Example value in mm
    'tracking_z_max': 2500.0,  # Example value in mm
    'energy_threshold': 0.05   # Example value in GeV (for decay tree processing)
}

# --- Loading an Event (with geometry) ---
file_path = 'path/to/your/events.root'
try:
    # Pass detector_params during initialization
    event = EDM4hepEvent(file_path, event_index=0, detector_params=detector_params)
    print(f"Loaded event: {event}")
    print(f"Detector params stored: {event.detector_params}")
except ValueError as e:
    print(e)
    # Handle error, e.g., skip this file/event

# --- Accessing Event DataFrames (for bulk operations) ---
if event:
    particles_df = event.get_particles_df()
    print(f"\nTotal particles: {len(particles_df)}")
    print("Particle DataFrame head:\n", particles_df.head())

    tracker_hits_df = event.get_tracker_hits_df()
    if not tracker_hits_df.empty:
        print(f"\nTotal tracker hits: {len(tracker_hits_df)}")
        print("Tracker Hits DataFrame head:\n", tracker_hits_df.head())

# --- Working with Individual Particles (Object-Oriented Interface) ---
if event and not particles_df.empty:
    # Get a specific particle by its ID (index in the DataFrame)
    try:
        particle_id = 5 # Example particle ID
        particle = event.get_particle(particle_id)
        print(f"\nParticle {particle_id}: {particle}")

        # Access particle properties
        print(f"  PDG: {particle.pdg}")
        print(f"  Charge: {particle.charge}")
        print(f"  pT: {particle.pt:.3f} GeV")
        print(f"  Vertex: {particle.vertex}")
        print(f"  Is backscatter? {particle.is_backscatter}")
        print(f"  Created inside tracker? {particle.created_inside_tracker}")

        # Navigate relationships
        daughters = particle.get_daughters()
        print(f"  Number of daughters: {len(daughters)}")
        if daughters:
            print(f"    Daughter 0: {daughters[0]}")

        parents = particle.get_parents()
        print(f"  Number of parents: {len(parents)}")

        # Find associated hits
        trkr_hits = particle.get_tracker_hits()
        print(f"  Associated tracker hits: {len(trkr_hits)}")
        if trkr_hits:
            print(f"    Tracker Hit 0: {trkr_hits[0]}")
            print(f"      Hit position: {trkr_hits[0].position}")
            print(f"      Hit detector: {trkr_hits[0].detector}")
            # Navigate back to particle from hit
            originating_particle = trkr_hits[0].get_particle()
            print(f"      Originating particle (from hit): {originating_particle.id}")

        calo_contribs = particle.get_calo_contributions()
        print(f"  Associated calo contributions: {len(calo_contribs)}")
        if calo_contribs:
            print(f"    Calo Contribution 0: {calo_contribs[0]}")
            print(f"      Contribution energy: {calo_contribs[0].energy:.3f} GeV")
            # Get the hit this contribution belongs to
            calo_hit = calo_contribs[0].get_hit()
            if calo_hit:
                print(f"      Associated Calo Hit: {calo_hit.hit_index} (Energy: {calo_hit.energy:.3f} GeV)")

    except IndexError as e:
        print(f"Error accessing particle: {e}")

# --- Using the Decay Tree ---
if event:
    try:
        # Get the decay graph (built automatically on first access)
        decay_graph = event.get_decay_tree()
        print(f"\nDecay graph built with {decay_graph.number_of_nodes()} nodes and {decay_graph.number_of_edges()} edges.")

        # Get ancestors/descendants of a particle
        particle_id = 10 # Example particle ID
        particle = event.get_particle(particle_id)
        ancestors = particle.get_ancestors()
        descendants = particle.get_descendants()
        print(f"Particle {particle_id} ancestors: {[p.id for p in ancestors]}")
        print(f"Particle {particle_id} descendants: {[p.id for p in descendants]}")

    except (IndexError, nx.NetworkXError) as e:
        print(f"Error working with decay tree for particle {particle_id}: {e}")

# --- Plotting --- 
if event:
    print("\nGenerating plots...")
    # Use the plotting handler attached to the event
    plotter = event.plot 
    
    # Example: Event overview plot (no longer needs detector_params)
    fig_overview = plotter.event_overview()
    if fig_overview:
        fig_overview.suptitle("Event Overview")
        plt.show()
    
    # Example: Decay tree visualization for a specific particle (no longer needs detector_params)
    try:
        particle_to_plot = event.get_particle(5) # Example
        fig_decay = plotter.visualize_decay_tree(
            particle_id=particle_to_plot.id, # Pass the particle ID
            show_tracking_cylinder=True, # Uses params stored in event
        )
        if fig_decay:
            fig_decay.suptitle(f"Decay Tree starting from Particle {particle_to_plot.id}")
    except IndexError:
        print("Particle for decay tree plotting not found.")

    # Example: Kinematic distribution
    if not particles_df.empty:
        # Plotting kinematics doesn't depend on detector geometry
        fig_kinematics = plotter.plot_particle_kinematics('all') # Example: plot all particles
        if fig_kinematics:
            fig_kinematics.suptitle("Kinematics (All Particles)")
            # plt.show() # Let environment display

# --- Finding Matching Particles Across Events ---
# Suppose we have loaded two events: event0 and event1
# And we want to find a particle in event1 that matches particle 5 from event0

# Example: Load two events (assuming event loading works for index 1)
try:
    event0 = EDM4hepEvent(file_path, event_index=0, detector_params=detector_params)
    event1 = EDM4hepEvent(file_path, event_index=1, detector_params=detector_params)
except ValueError as e:
    print(f"Error loading events: {e}")
    event0, event1 = None, None

if event0 and event1:
    try:
        source_particle_id = 5
        source_particle = event0.get_particle(source_particle_id)
        
        # Define the kinematics to match (optional, defaults work too)
        # kinematics_to_match = {
        #     'p': source_particle.p,
        #     'time': source_particle.time,
        #     'PDG': source_particle.pdg
        # }
        # match_vars = ['p', 'time', 'PDG'] 
        
        # Call the method on the target event object, passing the source particle
        matching_particle_in_event1 = event1.find_matching_particle(
            source_particle=source_particle, 
            # match_vars=match_vars, # Optional: Use defaults: ['time', 'px', 'py', 'pz', 'PDG']
            tolerance=1e-5 # Optional: Adjust tolerance
        )
        
        if matching_particle_in_event1 is not None:
            print(f"\nFound particle in event 1 matching particle {source_particle_id} from event 0:")
            print(f"  Event 0 Particle {source_particle_id}: {source_particle}")
            print(f"  Event 1 Particle {matching_particle_in_event1.id}: {matching_particle_in_event1}")
        else:
            print(f"\nNo particle found in event 1 matching particle {source_particle_id} from event 0 with specified kinematics.")

    except IndexError as e:
        print(f"Error finding matching particle: {e}")

## Core Design: Hybrid OOP + Pandas Backend

The central idea is to define Python classes representing EDM4hep entities (`Event`, `Particle`, `TrackerHit`, `CaloHit`, etc.) but have these classes act as interfaces or views to underlying Pandas DataFrames that hold the actual data.

-   **Data Loading:** Use `uproot` to read EDM4hep ROOT files efficiently into Pandas DataFrames.
-   **Central `EDM4hepEvent` Object:** Each event will be represented by an `EDM4hepEvent` object. This object will load and hold all the necessary DataFrames for that event (particles, hits, relationships, etc.).
-   **Entity Objects (`Particle`, `Hit`, etc.):** These objects will be lightweight. They will primarily store an identifier (like a `particle_id` or DataFrame index) and a reference back to their parent `EDM4hepEvent` object.
-   **Attribute Access:** Accessing attributes of an entity object (e.g., `particle.pt`) will trigger a lookup in the corresponding DataFrame within the `EDM4hepEvent` object using the entity's identifier. This will be implemented using Python `@property` decorators.
-   **Relationship Access:** Accessing related objects (e.g., `particle.get_daughters()`) will perform lookups using the relationship DataFrames (e.g., `_daughters_df`) and return lists of newly instantiated entity objects (created on demand).
-   **Bulk Operations:** Methods requiring operations across many entities (e.g., plotting distributions, calculating global event properties, building the full decay graph) will directly use the Pandas DataFrames stored within the `EDM4hepEvent` object for optimal performance.

## Proposed Class Structure

1.  **`EDM4hepEvent`**
    *   **Attributes:**
        *   `file_path`: Path to the source ROOT file.
        *   `event_number`: The specific event index loaded.
        *   `_particles_df`: DataFrame holding `MCParticles` data.
        *   `_parents_df`: DataFrame holding `_MCParticles_parents` relationship data.
        *   `_daughters_df`: DataFrame holding `_MCParticles_daughters` relationship data.
        *   `_tracker_hits_df`: Combined DataFrame for all tracker hits (from various `*Readout` collections).
        *   `_tracker_particle_links_df`: Combined DataFrame linking tracker hits to particles.
        *   `_calo_hits_df`: Combined DataFrame for all calorimeter hits (e.g., `ECalBarrelCollection`, `HCalEndcapCollection`).
        *   `_calo_contributions_df`: Combined DataFrame for calorimeter hit contributions.
        *   `_calo_particle_links_df`: Combined DataFrame linking calorimeter contributions to particles.
        *   `_decay_graph`: (Optional, lazy-loaded) `networkx.DiGraph` representing the decay tree.
        *   `event_header`: Dictionary or object holding event header info.
        *   `detector_params`: Dictionary holding detector geometry parameters.
    *   **Methods:**
        *   `__init__(file_path, event_number, detector_params)`: Loads data using helper functions (adapted from `edm4hep_utils.py`).
        *   `get_particles_df()`: Returns `_particles_df`.
        *   `get_tracker_hits_df()`: Returns `_tracker_hits_df`.
        *   `get_calo_hits_df()`: Returns `_calo_hits_df`.
        *   `get_calo_contributions_df()`: Returns `_calo_contributions_df`.
        *   `get_particle(particle_id)`: Returns a `Particle` object for the given ID.
        *   `get_tracker_hit(hit_index)`: Returns a `TrackerHit` object.
        *   `get_calo_hit(hit_index)`: Returns a `CaloHit` object.
        *   `get_calo_contribution(contrib_index)`: Returns a `CaloContribution` object.
        *   `build_decay_tree()`: Builds the `networkx.DiGraph` using `_particles_df` and `_daughters_df` (adapted from `particle_decay_tree.py`) and stores it in `_decay_graph`. Creates `Particle` nodes with references.
        *   `get_decay_tree()`: Returns the `_decay_graph`, building it if necessary.
        *   `run_diagnostics(detector_params)`: Performs event-level diagnostic plots (adapted from `viz_utils.py`).
        *   Potentially other bulk analysis/plotting methods.

2.  **`Particle`**
    *   **Attributes:**
        *   `_event`: Reference to the parent `EDM4hepEvent`.
        *   `particle_id`: The index/ID of this particle in `_particles_df`.
    *   **Methods:**
        *   `__init__(particle_id, event)`
    *   **Properties (`@property`)**:
        *   `pdg`: Looks up `PDG` in `_particles_df`.
        *   `charge`: Looks up `charge`.
        *   `mass`: Looks up `mass`.
        *   `pt`, `eta`, `phi`, `p`, `energy`: Looks up calculated or stored kinematic values.
        *   `vertex`: Returns `(vx, vy, vz)`.
        *   `endpoint`: Returns `(endpoint_x, endpoint_y, endpoint_z)`.
        *   `generator_status`, `simulator_status`: Looks up status flags.
        *   `is_created_in_simulation`: Decodes `simulator_status`. (Helper needed)
        *   `is_backscatter`: Decodes `simulator_status`. (Helper needed)
        *   `created_inside_tracker`: Decodes `simulator_status`. (Helper needed)
        *   *... other particle attributes ...*
    *   **Relationship Methods:**
        *   `get_parents()`: Uses `_particles_df` (`parents_begin`/`end`) and `_parents_df` to find parent IDs, returns list of `Particle` objects.
        *   `get_daughters()`: Uses `_particles_df` (`daughters_begin`/`end`) and `_daughters_df` to find daughter IDs, returns list of `Particle` objects.
        *   `get_tracker_hits()`: Queries `_tracker_particle_links_df` and `_tracker_hits_df`, returns list of `TrackerHit` objects.
        *   `get_calo_contributions()`: Queries `_calo_particle_links_df` and `_calo_contributions_df`, returns list of `CaloContribution` objects.
        *   `get_ancestors()`: Requires the decay graph. Uses `nx.ancestors(event.get_decay_tree(), self.particle_id)`. Returns list of `Particle` objects.
        *   `get_descendants()`: Requires the decay graph. Uses `nx.descendants(event.get_decay_tree(), self.particle_id)`. Returns list of `Particle` objects.

3.  **`TrackerHit`**
    *   **Attributes:**
        *   `_event`: Reference to `EDM4hepEvent`.
        *   `hit_index`: Index in `_tracker_hits_df`.
    *   **Methods:**
        *   `__init__(hit_index, event)`
    *   **Properties:**
        *   `position`: Returns `(x, y, z)`.
        *   `r`, `R`, `phi`, `theta`, `eta`: Calculated geometric properties.
        *   `time`, `edep`, `path_length`, `quality`, `cell_id`, `detector`: Looks up values.
        *   `momentum`: Returns `(px, py, pz)`.
        *   `pt`: Calculated transverse momentum.
    *   **Relationship Methods:**
        *   `get_particle()`: Uses `_tracker_particle_links_df` to find the associated `particle_id`, returns `Particle` object.

4.  **`CaloHit`** (Represents a cell hit)
    *   **Attributes:**
        *   `_event`: Reference to `EDM4hepEvent`.
        *   `hit_index`: Index in `_calo_hits_df`.
    *   **Methods:**
        *   `__init__(hit_index, event)`
    *   **Properties:**
        *   `position`: Returns `(x, y, z)`.
        *   `r`, `R`, `phi`, `theta`, `eta`: Calculated geometric properties.
        *   `energy`: Looks up total cell energy.
        *   `cell_id`, `detector`: Looks up values.
    *   **Relationship Methods:**
        *   `get_contributions()`: Uses `contribution_begin`/`end` range to find contribution indices in `_calo_contributions_df`, returns list of `CaloContribution` objects.

5.  **`CaloContribution`** (Represents a single particle's contribution to a `CaloHit`)
    *   **Attributes:**
        *   `_event`: Reference to `EDM4hepEvent`.
        *   `contrib_index`: Index in `_calo_contributions_df`.
    *   **Methods:**
        *   `__init__(contrib_index, event)`
    *   **Properties:**
        *   `pdg`, `energy`, `time`: Looks up values.
        *   `step_position`: Returns `(step_x, step_y, step_z)`.
    *   **Relationship Methods:**
        *   `get_particle()`: Uses `_calo_particle_links_df` to find the associated `particle_id`, returns `Particle` object.
        *   `get_hit()`: Determines the parent `CaloHit` based on the contribution index range in `_calo_hits_df`, returns `CaloHit` object.

## Implementation Steps

1.  **Create Directory Structure:**
    ```
    pyedm4hep/
        __init__.py
        event.py        # EDM4hepEvent class
        particle.py     # Particle class
        hits.py         # TrackerHit, CaloHit, CaloContribution classes
        utils.py        # Data loading helpers (adapted from edm4hep_utils.py)
        plotting.py     # Plotting helpers (adapted from viz_utils.py)
        decay.py        # Decay tree functions (adapted from particle_decay_tree.py)
        README.md       # This file
    ```
2.  **Adapt `edm4hep_utils.py`:** Move data loading and DataFrame building functions into `utils.py`. Ensure they return the necessary set of DataFrames. Modify `_process_event` logic.
3.  **Implement `EDM4hepEvent`:** Create the class in `event.py`. Implement `__init__` to call loading functions from `utils.py`. Implement basic DataFrame accessors.
4.  **Implement `Particle`, `TrackerHit`, `CaloHit`, `CaloContribution`:** Create classes in `particle.py` and `hits.py`. Implement `__init__` and `@property` methods for attribute access.
5.  **Implement Relationship Methods:** Add methods like `get_parents`, `get_daughters`, `get_particle`, `get_contributions` using DataFrame lookups.
6.  **Adapt `particle_decay_tree.py`:** Move functions to `decay.py`. Adapt `build_decay_tree` to be a method of `EDM4hepEvent` and work with its internal DataFrames. Ensure node attributes include the necessary info and potentially store references to `Particle` objects. Add `get_ancestors`/`get_descendants` to the `Particle` class.
7.  **Adapt `viz_utils.py`:** Move functions to `plotting.py`. Adapt `event_diagnostics` to be a method of `EDM4hepEvent` or take an `EDM4hepEvent` object.
8.  **Testing:** Create notebooks or scripts to test:
    *   Loading events.
    *   Accessing attributes of individual particles and hits.
    *   Navigating relationships (parents, daughters, hits <-> particles).
    *   Building and accessing the decay tree (ancestors/descendants).
    *   Running bulk operations and comparing performance/results to the pure DataFrame approach.

## Future Considerations

-   Error handling (missing files, incorrect event numbers, missing branches).
-   More sophisticated caching (e.g., for relationship lookups).
-   Integration with other analysis frameworks.
-   Packaging for distribution (`setup.py` or `pyproject.toml`). 