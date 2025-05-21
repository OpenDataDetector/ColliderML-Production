import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, List, Tuple, Optional, Dict
import networkx as nx

# Import utility functions
from .utils import get_simulator_status_bits

# Type hinting for Event object to avoid circular imports
if TYPE_CHECKING:
    from .event import EDM4hepEvent
    from .hits import TrackerHit, CaloContribution

class Particle:
    """Represents a single MCParticle within an EDM4hepEvent.

    Provides access to particle properties and relationships through
    methods and properties that query the parent EDM4hepEvent's dataframes.
    """
    def __init__(self, particle_id: int, event: 'EDM4hepEvent'):
        """Initializes the Particle object.

        Args:
            particle_id (int): The index (ID) of this particle in the event's particles_df.
            event (EDM4hepEvent): The parent event object containing the dataframes.
        """
        # Check if particle_id exists at initialization
        if particle_id not in event._particles_df.index:
            raise IndexError(f"Particle ID {particle_id} not found in event {event.event_index}. Cannot create Particle object.")
        self._id = particle_id
        self._event = event
        # Store the particle's data row for quick access (optional optimization)
        # self._data = self._event._particles_df.loc[self._id]

    @property
    def id(self) -> int:
        """The unique ID (index) of this particle within the event."""
        return self._id

    def _get_data(self, column: str = None):
        """Helper to safely get data from the particles dataframe."""
        if column is None:
            return self._event._particles_df.loc[self._id]
        try:
            return self._event._particles_df.loc[self._id, column]
        except KeyError:
            # This might happen if the column doesn't exist or particle ID became invalid
            # Re-check existence might be needed if DataFrames could change, but they shouldn't after load.
            print(f"Warning: Column '{column}' not found for particle {self._id}.")
            return None # Or raise an error
        except IndexError:
            print(f"Warning: Particle ID {self._id} seems invalid when accessing '{column}'.")
            return None # Or raise an error

    # --- Basic Properties ---
    @property
    def pdg(self) -> int:
        """Particle Data Group (PDG) ID."""
        return int(self._get_data('PDG'))

    @property
    def charge(self) -> float:
        """Electric charge."""
        return float(self._get_data('charge'))

    @property
    def mass(self) -> float:
        """Mass of the particle."""
        return float(self._get_data('mass'))

    @property
    def time(self) -> float:
        """Creation time."""
        return float(self._get_data('time'))

    @property
    def generator_status(self) -> int:
        """Status code from the event generator."""
        return int(self._get_data('generatorStatus'))

    @property
    def simulator_status(self) -> int:
        """Status code from the simulation (contains status bits)."""
        return int(self._get_data('simulatorStatus'))

    # --- Kinematic Properties ---
    @property
    def px(self) -> float:
        """Momentum x-component."""
        return float(self._get_data('px'))

    @property
    def py(self) -> float:
        """Momentum y-component."""
        return float(self._get_data('py'))

    @property
    def pz(self) -> float:
        """Momentum z-component."""
        return float(self._get_data('pz'))

    @property
    def momentum(self) -> Tuple[float, float, float]:
        """Momentum vector (px, py, pz)."""
        return (self.px, self.py, self.pz)

    @property
    def kinetic_energy(self) -> float:
        """Kinetic energy."""
        return float(self._get_data('kinetic_energy'))

    @property
    def pt(self) -> float:
        """Transverse momentum."""
        return float(self._get_data('pt'))

    @property
    def p(self) -> float:
        """Magnitude of momentum."""
        return float(self._get_data('p'))

    @property
    def energy(self) -> float:
        """Energy (calculated as sqrt(p^2 + m^2)).
        """
        return float(self._get_data('energy'))

    @property
    def eta(self) -> float:
        """Pseudorapidity."""
        return float(self._get_data('eta'))

    @property
    def phi(self) -> float:
        """Azimuthal angle."""
        return float(self._get_data('phi'))

    # --- Vertex and Endpoint Properties ---
    @property
    def vx(self) -> float:
        """Production vertex x-coordinate."""
        return float(self._get_data('vx'))

    @property
    def vy(self) -> float:
        """Production vertex y-coordinate."""
        return float(self._get_data('vy'))

    @property
    def vz(self) -> float:
        """Production vertex z-coordinate."""
        return float(self._get_data('vz'))

    @property
    def vertex(self) -> Tuple[float, float, float]:
        """Production vertex (vx, vy, vz)."""
        return (self.vx, self.vy, self.vz)

    @property
    def endpoint_x(self) -> float:
        """Decay/interaction endpoint x-coordinate."""
        return float(self._get_data('endpoint_x'))

    @property
    def endpoint_y(self) -> float:
        """Decay/interaction endpoint y-coordinate."""
        return float(self._get_data('endpoint_y'))

    @property
    def endpoint_z(self) -> float:
        """Decay/interaction endpoint z-coordinate."""
        return float(self._get_data('endpoint_z'))

    @property
    def endpoint(self) -> Tuple[float, float, float]:
        """Decay/interaction endpoint (x, y, z)."""
        return (self.endpoint_x, self.endpoint_y, self.endpoint_z)

    @property
    def vr(self) -> Optional[float]:
        """Production vertex radius in x-y plane. Returns None if not pre-calculated."""
        val = self._get_data('vr')
        return float(val) if val is not None else None

    @property
    def endpoint_r(self) -> Optional[float]:
        """Endpoint radius in x-y plane. Returns None if not pre-calculated."""
        val = self._get_data('endpoint_r')
        return float(val) if val is not None else None

    # --- Simulation Status Properties ---
    @property
    def status_bits(self) -> Dict[str, bool]:
         """Decoded simulation status bits."""
         # Cache the result as it involves computation
         if not hasattr(self, '_status_bits_cache'):
             self._status_bits_cache = get_simulator_status_bits(self.simulator_status)
         return self._status_bits_cache

    @property
    def is_created_in_simulation(self) -> Optional[bool]:
         """Whether the particle was created in simulation (Geant4). Reads pre-calculated flag."""
         # Reads pre-calculated column from event init
         val = self._get_data('created_in_simulation')
         # Check if the value exists and is boolean-like (should be bool or np.bool_)
         if val is not None and isinstance(val, (bool, np.bool_)):
             return bool(val)
         # Fallback to decoding status bits if column is missing (optional, adds complexity)
         # print("Warning: 'created_in_simulation' column missing, falling back to status_bits.")
         # return self.status_bits.get('created_in_simulation', None)
         return None

    @property
    def is_backscatter(self) -> Optional[bool]:
         """Whether the particle is a backscatter. Reads pre-calculated flag based on detector geometry."""
         val = self._get_data('backscatter')
         if val is not None and isinstance(val, (bool, np.bool_)):
            return bool(val)
         return None

    @property
    def created_inside_tracker(self) -> Optional[bool]:
        """Whether the particle was created inside the tracker volume (requires detector_params during event init)."""
        val = self._get_data('created_inside_tracker')
        if val is not None and isinstance(val, (bool, np.bool_)):
            return bool(val)
        return None

    @property
    def ended_inside_tracker(self) -> Optional[bool]:
        """Whether the particle endpoint is inside the tracker volume (requires detector_params during event init)."""
        val = self._get_data('ended_inside_tracker')
        if val is not None and isinstance(val, (bool, np.bool_)):
            return bool(val)
        return None

    @property
    def vertex_is_not_endpoint_of_parent(self) -> bool:
         """Decodes from simulator status bits."""
         return self.status_bits.get('vertex_not_endpoint', False)

    @property
    def decayed_in_tracker(self) -> bool:
        """Decodes from simulator status bits."""
        return self.status_bits.get('decayed_in_tracker', False)

    @property
    def decayed_in_calorimeter(self) -> bool:
        """Decodes from simulator status bits."""
        return self.status_bits.get('decayed_in_calorimeter', False)

    @property
    def has_left_detector(self) -> bool:
        """Decodes from simulator status bits."""
        return self.status_bits.get('has_left_detector', False)

    @property
    def stopped(self) -> bool:
        """Decodes from simulator status bits."""
        return self.status_bits.get('stopped', False)

    @property
    def is_overlay(self) -> bool:
        """Decodes from simulator status bits."""
        return self.status_bits.get('overlay', False)

    # --- Relationship Methods ---

    def get_parents(self) -> List['Particle']:
        """Returns a list of parent Particle objects."""
        particles_df = self._event._particles_df
        parents_df = self._event._parents_df

        if 'parents_begin' not in particles_df.columns or 'parents_end' not in particles_df.columns:
            return []
        if parents_df.empty:
             return []

        start = int(particles_df.loc[self._id, 'parents_begin'])
        end = int(particles_df.loc[self._id, 'parents_end'])

        parent_ids = parents_df.iloc[start:end]['particle_id'].tolist()
        return [self._event.get_particle(pid) for pid in parent_ids if pid in particles_df.index]

    def get_daughters(self) -> List['Particle']:
        """Returns a list of daughter Particle objects."""
        particles_df = self._event._particles_df
        daughters_df = self._event._daughters_df

        if 'daughters_begin' not in particles_df.columns or 'daughters_end' not in particles_df.columns:
            return []
        if daughters_df.empty:
            return []

        start = int(particles_df.loc[self._id, 'daughters_begin'])
        end = int(particles_df.loc[self._id, 'daughters_end'])

        daughter_ids = daughters_df.iloc[start:end]['particle_id'].tolist()
        return [self._event.get_particle(pid) for pid in daughter_ids if pid in particles_df.index]

    def get_tracker_hits(self) -> List['TrackerHit']:
        """Returns a list of associated TrackerHit objects."""
        tracker_hits_df = self._event._tracker_hits_df
        if tracker_hits_df.empty or 'particle_id' not in tracker_hits_df.columns:
            return []

        hit_indices = tracker_hits_df[tracker_hits_df['particle_id'] == self._id]['global_hit_index'].tolist()
        # Need to import here to avoid circular dependency at module level
        from .hits import TrackerHit
        return [TrackerHit(idx, self._event) for idx in hit_indices if idx in self._event._tracker_hits_df.index]

    def get_tracker_hits_df(self) -> pd.DataFrame:
        """Returns a dataframe of associated TrackerHit objects."""
        tracker_hits_df = self._event._tracker_hits_df
        if tracker_hits_df.empty or 'particle_id' not in tracker_hits_df.columns:
            return pd.DataFrame()

        return tracker_hits_df[tracker_hits_df['particle_id'] == self._id]

    def get_num_hits(self) -> int:
        """Returns the number of tracker AND calorimeter hits associated with the particle."""
        tracker_hits = self.get_tracker_hits_df()
        calo_contributions = self.get_calo_contributions_df()
        return len(tracker_hits) + len(calo_contributions)


    def get_num_tracker_hits(self) -> int:
        """Returns the number of tracker hits associated with the particle."""
        tracker_hits = self.get_tracker_hits_df()
        return len(tracker_hits)

    def get_num_calo_hits(self) -> int:
        """Returns the number of calorimeter hits associated with the particle."""
        calo_hits = self.get_calo_hits_df()
        return len(calo_hits)

    def get_calo_hits(self) -> List['CaloContribution']:
        """Returns a list of associated CaloContribution objects."""
        calo_contributions_df = self._event._calo_contributions_df
        if calo_contributions_df.empty or 'particle_id' not in calo_contributions_df.columns:
            return []

        contrib_indices = calo_contributions_df[calo_contributions_df['particle_id'] == self._id]['global_contrib_index'].tolist()
        # Need to import here to avoid circular dependency at module level
        from .hits import CaloContribution
        return [CaloContribution(idx, self._event) for idx in contrib_indices if idx in self._event._calo_contributions_df.index]

    def get_calo_hits_df(self) -> pd.DataFrame:
        """Returns a dataframe of associated CaloContribution objects."""
        calo_contributions_df = self._event._calo_contributions_df
        if calo_contributions_df.empty or 'particle_id' not in calo_contributions_df.columns:
            return pd.DataFrame()

        return calo_contributions_df[calo_contributions_df['particle_id'] == self._id]

    # --- Decay Tree Methods (require graph to be built) ---

    def get_ancestors(self) -> List['Particle']:
        """Returns a list of ancestor Particle objects (requires decay tree)."""
        decay_graph = self._event.get_decay_tree() # Ensures graph is built
        if self._id not in decay_graph:
            print(f"Warning: Particle {self._id} not found in decay graph.")
            return []
        try:
            ancestor_ids = nx.ancestors(decay_graph, self._id)
            return [self._event.get_particle(pid) for pid in ancestor_ids if pid in self._event._particles_df.index]
        except nx.NetworkXError as e:
             print(f"Error getting ancestors for particle {self._id}: {e}")
             return []

    def get_descendants(self) -> List['Particle']:
        """Returns a list of descendant Particle objects (requires decay tree)."""
        decay_graph = self._event.get_decay_tree()
        if self._id not in decay_graph:
            print(f"Warning: Particle {self._id} not found in decay graph.")
            return []
        try:
            descendant_ids = nx.descendants(decay_graph, self._id)
            return [self._event.get_particle(pid) for pid in descendant_ids if pid in self._event._particles_df.index]
        except nx.NetworkXError as e:
             print(f"Error getting descendants for particle {self._id}: {e}")
             return []

    # --- Representation ---

    def __repr__(self) -> str:
        return f"<Particle id={self._id} pdg={self.pdg} p={self.p:.2f} pT={self.pt:.2f} eta={self.eta:.2f}>"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Particle):
            return NotImplemented
        # Two particles are the same if they belong to the same event and have the same ID
        return self._id == other._id and self._event is other._event

    def __hash__(self) -> int:
        # Hash based on event instance and particle ID for uniqueness within/across events
        return hash((id(self._event), self._id))
