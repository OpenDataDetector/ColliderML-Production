import pandas as pd
import numpy as np
import networkx as nx

# Import loading functions from utils
from .utils import load_event_data, get_simulator_status_bits
# Import placeholder classes (will be defined in particle.py, hits.py)
# We use string forward references or import them conditionally for type hinting
from typing import TYPE_CHECKING, List, Dict, Optional
if TYPE_CHECKING:
    from .particle import Particle
    from .hits import TrackerHit, CaloHit, CaloContribution
    from .decay import DecayGraphHandler # Assuming decay logic moves here
    from .plotting import PlottingHandler # Assuming plotting logic moves here

class EDM4hepEvent:
    """Represents a single event from an EDM4hep file,
    providing access to particles, hits, and their relationships
    via both OOP interfaces and underlying Pandas DataFrames.
    """

    def __init__(self, file_path: str, event_index: int, detector_params: Optional[Dict[str, float]] = None):
        """Loads data for the specified event index from the file.

        Args:
            file_path (str): Path to the EDM4hep ROOT file.
            event_index (int): The 0-based index of the event to load.
            detector_params (Optional[Dict[str, float]]): Dictionary with detector geometry
                 (e.g., {'tracking_radius': float, 'tracking_z_max': float}).
                 If provided, geometry-dependent flags will be calculated. 

        Raises:
            ValueError: If the event index is invalid or the file/event cannot be loaded.
        """
        self.file_path = file_path
        self.event_index = event_index
        self.detector_params = detector_params

        # Load data and assign to internal attributes
        self._load_and_assign_data()

        # Pre-calculate derived properties if particles exist
        if not self._particles_df.empty:
            self._calculate_derived_particle_properties()
            self._calculate_geometry_flags()

        # Placeholder for lazily-loaded handlers and graph
        self._decay_graph: Optional[nx.DiGraph] = None
        self._decay_handler: Optional['DecayGraphHandler'] = None
        self._plotting_handler: Optional['PlottingHandler'] = None

    def _load_and_assign_data(self):
        """Loads event data and assigns it to internal DataFrame attributes."""
        self._data = load_event_data(self.file_path, self.event_index)
        if self._data is None:
            raise ValueError(f"Failed to load event {self.event_index} from {self.file_path}.")

        # Store dataframes directly as attributes
        self._particles_df: pd.DataFrame = self._data['particles']
        self._parents_df: pd.DataFrame = self._data['parents']
        self._daughters_df: pd.DataFrame = self._data['daughters']
        self._tracker_hits_df: pd.DataFrame = self._data['tracker_hits']
        self._calo_hits_df: pd.DataFrame = self._data['calo_hits']
        self._calo_contributions_df: pd.DataFrame = self._data['calo_contributions']
        self.event_header: dict = self._data['event_header']

    def _calculate_derived_particle_properties(self):
        """Calculates basic derived properties and simulation status flags for particles."""
        # Calculate vertex and endpoint r coordinates
        self._particles_df["vr"] = np.sqrt(self._particles_df.vx**2 + self._particles_df.vy**2)
        self._particles_df["endpoint_r"] = np.sqrt(self._particles_df.endpoint_x**2 + self._particles_df.endpoint_y**2)
        self._particles_df["energy"] = np.sqrt(self._particles_df.px**2 + self._particles_df.py**2 + self._particles_df.pz**2 + self._particles_df.mass**2)
        self._particles_df["kinetic_energy"] = self._particles_df["energy"] - self._particles_df["mass"]

        # Calculate number of hits per particle
        all_hits = pd.concat([self._tracker_hits_df[["particle_id"]], self._calo_contributions_df[["particle_id"]]])
        self._particles_df["num_hits"] = all_hits.groupby("particle_id").size().reindex(self._particles_df.index, fill_value=0)

        # Calculate simulation status flags
        status_bits_series = self._particles_df['simulatorStatus'].apply(get_simulator_status_bits)
        self._particles_df['created_in_simulation'] = status_bits_series.apply(lambda bits: bits.get('created_in_simulation', False))

    def _calculate_geometry_flags(self):
        """Calculates geometry-dependent flags based on detector_params."""
        if self.detector_params:
            tracking_radius = self.detector_params.get('tracking_radius')
            tracking_z_max = self.detector_params.get('tracking_z_max')

            if tracking_radius is not None and tracking_z_max is not None:
                self._particles_df["created_inside_tracker"] = (
                    (self._particles_df.vr < tracking_radius) &
                    (self._particles_df.vz.abs() < tracking_z_max)
                ).astype(bool)

                self._particles_df["ended_inside_tracker"] = (
                    (self._particles_df.endpoint_r < tracking_radius) &
                    (self._particles_df.endpoint_z.abs() < tracking_z_max)
                ).astype(bool)

                # Update backscatter based on these flags
                # Overwrites the one from simulatorStatus if geometry is defined
                self._particles_df["backscatter"] = (
                    (self._particles_df["created_inside_tracker"] == False) &
                    (self._particles_df["ended_inside_tracker"] == True)
                )
            else:
                print("Warning: detector_params provided but missing 'tracking_radius' or 'tracking_z_max'. Geometry flags not calculated.")
                # Add placeholder columns if calculation failed due to missing params
                self._particles_df["created_inside_tracker"] = False
                self._particles_df["ended_inside_tracker"] = False
                # Keep the backscatter flag derived from simulatorStatus (calculated in _calculate_derived_particle_properties)
        else:
             # Add placeholder columns if no geometry defined
             self._particles_df["created_inside_tracker"] = False
             self._particles_df["ended_inside_tracker"] = False
             # Keep the backscatter flag derived from simulatorStatus

    # --- DataFrame Accessors ---

    def get_particles_df(self) -> pd.DataFrame:
        """Returns the DataFrame containing all MCParticle data for this event."""
        return self._particles_df.copy() # Return copy to prevent modification

    def get_parents_df(self) -> pd.DataFrame:
        """Returns the DataFrame containing parent link data."""
        return self._parents_df.copy()

    def get_daughters_df(self) -> pd.DataFrame:
        """Returns the DataFrame containing daughter link data."""
        return self._daughters_df.copy()

    def get_tracker_hits_df(self) -> pd.DataFrame:
        """Returns the combined DataFrame for all tracker hits."""
        return self._tracker_hits_df.copy()

    def get_calo_hits_df(self) -> pd.DataFrame:
        """Returns the combined DataFrame for all calorimeter hits."""
        return self._calo_hits_df.copy()

    def get_calo_contributions_df(self) -> pd.DataFrame:
        """Returns the combined DataFrame for all calorimeter contributions."""
        return self._calo_contributions_df.copy()

    # --- Object Accessors ---

    def get_particle(self, particle_id: int) -> 'Particle':
        """Returns a Particle object for the given particle ID (index)."""
        # Import locally to avoid circular dependency issues at load time
        from .particle import Particle
        if particle_id not in self._particles_df.index:
            raise IndexError(f"Particle ID {particle_id} not found in event {self.event_index}.")
        return Particle(particle_id, self)

    def get_tracker_hit(self, global_hit_index: int) -> 'TrackerHit':
        """Returns a TrackerHit object for the given global hit index."""
        from .hits import TrackerHit
        if global_hit_index not in self._tracker_hits_df.index:
            raise IndexError(f"Tracker hit index {global_hit_index} not found in event {self.event_index}.")
        return TrackerHit(global_hit_index, self)

    def get_calo_hit(self, global_hit_index: int) -> 'CaloHit':
        """Returns a CaloHit object for the given global hit index."""
        from .hits import CaloHit
        if global_hit_index not in self._calo_hits_df.index:
            raise IndexError(f"Calo hit index {global_hit_index} not found in event {self.event_index}.")
        return CaloHit(global_hit_index, self)

    def get_calo_contribution(self, global_contrib_index: int) -> 'CaloContribution':
        """Returns a CaloContribution object for the given global contribution index."""
        from .hits import CaloContribution
        if global_contrib_index not in self._calo_contributions_df.index:
            raise IndexError(f"Calo contribution index {global_contrib_index} not found in event {self.event_index}.")
        return CaloContribution(global_contrib_index, self)

    # --- Decay Tree Access ---

    @property
    def decay(self) -> 'DecayGraphHandler':
        """Provides access to decay graph building and analysis methods."""
        if self._decay_handler is None:
            # Lazily import and instantiate the handler
            from .decay import DecayGraphHandler
            self._decay_handler = DecayGraphHandler(self)
        return self._decay_handler

    def get_decay_tree(self) -> nx.DiGraph:
        """Builds (if necessary) and returns the NetworkX decay graph."""
        return self.decay.get_graph() # Delegate to handler

    # --- Plotting Access ---
    @property
    def plot(self) -> 'PlottingHandler':
        """Provides access to plotting methods for this event."""
        if self._plotting_handler is None:
            # Lazily import and instantiate the handler
            from .plotting import PlottingHandler
            self._plotting_handler = PlottingHandler(self)
        return self._plotting_handler

    # --- Other Potential Methods ---

    def __repr__(self) -> str:
        return f"<EDM4hepEvent file='{self.file_path}' index={self.event_index} particles={len(self._particles_df)} track_hits={len(self._tracker_hits_df)} calo_hits={len(self._calo_hits_df)}>"

    def get_particles(self) -> List['Particle']:
        """Returns a list of all Particle objects in the event."""
        # Import locally
        from .particle import Particle
        return [Particle(pid, self) for pid in self._particles_df.index]

    def find_matching_particle(self, 
                                 source_particle: 'Particle', 
                                 match_vars: Optional[List[str]] = None, 
                                 tolerance: float = 1e-6) -> Optional['Particle']:
        """Finds the first particle in this event that matches the given source particle's kinematics.

        Args:
            source_particle (Particle): The particle object from another event (or this one) 
                                        whose kinematics should be matched.
            match_vars (Optional[List[str]]): List of attribute/property names (kinematic variables) 
                                              on the Particle object to match on. If None, 
                                              defaults to ['time', 'px', 'py', 'pz', 'pdg']. 
                                              Note: 'pdg' will be mapped to the 'PDG' DataFrame column.
            tolerance (float): Absolute tolerance used for comparing floating-point values 
                               using numpy.isclose(). Defaults to 1e-6.

        Returns:
            Optional[Particle]: The Particle object of the first matching particle found, 
                                or None if no match is found.
        """
        target_particles_df = self._particles_df # Use internal DataFrame
        
        if target_particles_df is None or target_particles_df.empty:
            return None
            
        if source_particle is None:
            print("Error: source_particle cannot be None.")
            return None
            
        # Define default variables if not provided (use lowercase property names)
        if match_vars is None:
            match_vars = ['time', 'px', 'py', 'pz', 'pdg']
            
        # Extract source kinematics from the source_particle object using getattr
        source_kinematics = {}
        for var in match_vars:
            try:
                source_kinematics[var] = getattr(source_particle, var)
            except AttributeError:
                print(f"Warning: Attribute '{var}' not found on source_particle. Skipping this variable.")
                # Remove var from matching list if it can't be retrieved
                # match_vars.remove(var) # Careful modifying list during iteration, better to just skip
                continue 

        # Filter out variables we couldn't get from source
        valid_match_vars = [var for var in match_vars if var in source_kinematics]
        if not valid_match_vars:
             print("Error: No valid variables to match on.")
             return None
             
        # Start with a boolean mask selecting all rows
        match_mask = pd.Series(True, index=target_particles_df.index)
        
        for var in valid_match_vars: # Use only vars successfully retrieved
            # Map property name to DataFrame column name if necessary
            target_column = 'PDG' if var == 'pdg' else var
            
            if target_column not in target_particles_df.columns:
                print(f"Warning: Target column '{target_column}' (derived from '{var}') not found in target_particles_df. Skipping.")
                continue
                
            source_value = source_kinematics[var]
            target_values = target_particles_df[target_column] # Use mapped column name
            
            # Determine comparison method based on data type
            # Check source type first, as target might be object dtype initially
            if isinstance(source_value, float):
                # Use np.isclose for floating-point comparison
                try:
                    # Ensure target can be converted to float for comparison
                    target_float = target_values.astype(float)
                    match_mask &= np.isclose(target_float, source_value, atol=tolerance)
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert target column '{target_column}' to float for comparison with '{var}'. Skipping.")
                    continue
            elif isinstance(source_value, (int, np.integer)):
                # Use exact matching for integers
                try:
                    # Ensure target can be converted to compatible int type if needed
                    target_int = target_values.astype(np.int64) # Or choose appropriate int type
                    match_mask &= (target_int == source_value)
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert target column '{target_column}' to int for comparison with '{var}'. Skipping.")
                    continue
            elif isinstance(source_value, (bool, np.bool_)):
                 # Use exact matching for booleans
                 try:
                     target_bool = target_values.astype(bool)
                     match_mask &= (target_bool == source_value)
                 except (ValueError, TypeError):
                     print(f"Warning: Could not convert target column '{target_column}' to bool for comparison with '{var}'. Skipping.")
                     continue
            else:
                # Attempt exact matching for other types (e.g., strings, if any)
                try:
                    match_mask &= (target_values == source_value)
                except TypeError:
                     print(f"Warning: Could not compare source value type {type(source_value)} with target column type {target_values.dtype} for variable '{var}'. Skipping.")
                     continue
                
        # Find indices where the mask is True
        matching_indices = target_particles_df.index[match_mask]
        
        if not matching_indices.empty:
            matching_id = matching_indices[0] # Get the first matching ID
            # Import locally
            from .particle import Particle 
            return Particle(matching_id, self) # Return the Particle object
        else:
            return None

    # ... potentially add methods for getting lists of hits etc. ...
