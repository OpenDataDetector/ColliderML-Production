import numpy as np
from typing import TYPE_CHECKING, List, Tuple, Optional, Dict

# Type hinting for Event and Particle objects to avoid circular imports
if TYPE_CHECKING:
    from .event import EDM4hepEvent
    from .particle import Particle


class TrackerHit:
    """Represents a single hit in the tracking detector."""
    
    def __init__(self, global_hit_index: int, event: 'EDM4hepEvent'):
        """Initialize a TrackerHit object.
        
        Args:
            global_hit_index (int): The global index of this hit in the tracker_hits_df.
            event (EDM4hepEvent): The parent event object.
        """
        # Check if hit index exists at initialization
        if global_hit_index not in event._tracker_hits_df.index:
            raise IndexError(f"Tracker hit index {global_hit_index} not found in event {event.event_index}.")
        
        self._hit_index = global_hit_index
        self._event = event
    
    @property
    def hit_index(self) -> int:
        """The global hit index (within the combined tracker hits)."""
        return self._hit_index
    
    def _get_data(self, column: str):
        """Helper to safely get data from the tracker hits dataframe."""
        try:
            return self._event._tracker_hits_df.loc[self._hit_index, column]
        except KeyError:
            print(f"Warning: Column '{column}' not found for tracker hit {self._hit_index}.")
            return None
        except IndexError:
            print(f"Warning: Hit index {self._hit_index} seems invalid when accessing '{column}'.")
            return None
    
    # --- Basic Properties ---
    
    @property
    def detector(self) -> str:
        """The detector component that registered this hit."""
        return str(self._get_data('detector'))
    
    @property
    def cell_id(self) -> int:
        """Cell ID within the detector."""
        return int(self._get_data('cellID'))
    
    @property
    def time(self) -> float:
        """Time of the hit."""
        return float(self._get_data('time'))
    
    @property
    def path_length(self) -> float:
        """Path length."""
        return float(self._get_data('pathLength'))
    
    @property
    def quality(self) -> int:
        """Hit quality flag."""
        return int(self._get_data('quality'))
    
    @property
    def edep(self) -> float:
        """Energy deposited in the hit."""
        return float(self._get_data('EDep'))
    
    # --- Position and Momentum Properties ---
    
    @property
    def x(self) -> float:
        """X-coordinate."""
        return float(self._get_data('x'))
    
    @property
    def y(self) -> float:
        """Y-coordinate."""
        return float(self._get_data('y'))
    
    @property
    def z(self) -> float:
        """Z-coordinate."""
        return float(self._get_data('z'))
    
    @property
    def position(self) -> Tuple[float, float, float]:
        """Position vector (x, y, z)."""
        return (self.x, self.y, self.z)
    
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
    
    # --- Derived Geometric Properties ---
    
    @property
    def r(self) -> float:
        """Radial distance from the beam axis (x-y plane)."""
        return float(self._get_data('r'))
    
    @property
    def R(self) -> float:
        """Radial distance from the origin (3D)."""
        return float(self._get_data('R'))
    
    @property
    def phi(self) -> float:
        """Azimuthal angle."""
        return float(self._get_data('phi'))
    
    @property
    def theta(self) -> float:
        """Polar angle."""
        return float(self._get_data('theta'))
    
    @property
    def eta(self) -> float:
        """Pseudorapidity."""
        return float(self._get_data('eta'))
    
    @property
    def pt(self) -> float:
        """Transverse momentum."""
        return float(self._get_data('pt'))
    
    # --- Relationships ---
    
    def get_particle(self) -> Optional['Particle']:
        """Returns the particle associated with this hit, or None if not found."""
        links_df = self._event._tracker_links_df
        
        if links_df.empty or 'global_hit_index' not in links_df.columns:
            return None
        
        # Find in the links DataFrame
        matches = links_df[links_df['global_hit_index'] == self._hit_index]
        if matches.empty:
            return None
        
        # Get the particle ID(s)
        particle_id = matches.iloc[0]['particle_id']
        
        # Import locally to avoid circular dependency
        from .particle import Particle
        
        try:
            return self._event.get_particle(particle_id)
        except (KeyError, IndexError):
            print(f"Warning: Linked particle ID {particle_id} not found for hit {self._hit_index}.")
            return None
    
    # --- Representation ---
    
    def __repr__(self) -> str:
        return f"<TrackerHit idx={self._hit_index} det='{self.detector}' pos=({self.x:.1f}, {self.y:.1f}, {self.z:.1f})>"


class CaloHit:
    """Represents a single hit in a calorimeter cell."""
    
    def __init__(self, global_hit_index: int, event: 'EDM4hepEvent'):
        """Initialize a CaloHit object.
        
        Args:
            global_hit_index (int): The global index of this hit in the calo_hits_df.
            event (EDM4hepEvent): The parent event object.
        """
        # Check if hit index exists at initialization
        if global_hit_index not in event._calo_hits_df.index:
            raise IndexError(f"Calo hit index {global_hit_index} not found in event {event.event_index}.")
        
        self._hit_index = global_hit_index
        self._event = event
    
    @property
    def hit_index(self) -> int:
        """The global hit index (within the combined calo hits)."""
        return self._hit_index
    
    def _get_data(self, column: str):
        """Helper to safely get data from the calo hits dataframe."""
        try:
            return self._event._calo_hits_df.loc[self._hit_index, column]
        except KeyError:
            print(f"Warning: Column '{column}' not found for calo hit {self._hit_index}.")
            return None
        except IndexError:
            print(f"Warning: Hit index {self._hit_index} seems invalid when accessing '{column}'.")
            return None
    
    # --- Basic Properties ---
    
    @property
    def detector(self) -> str:
        """The detector component that registered this hit."""
        return str(self._get_data('detector'))
    
    @property
    def cell_id(self) -> int:
        """Cell ID within the detector."""
        return int(self._get_data('cellID'))
    
    @property
    def energy(self) -> float:
        """Total energy deposited in the hit."""
        return float(self._get_data('energy'))
    
    # --- Position Properties ---
    
    @property
    def x(self) -> float:
        """X-coordinate."""
        return float(self._get_data('x'))
    
    @property
    def y(self) -> float:
        """Y-coordinate."""
        return float(self._get_data('y'))
    
    @property
    def z(self) -> float:
        """Z-coordinate."""
        return float(self._get_data('z'))
    
    @property
    def position(self) -> Tuple[float, float, float]:
        """Position vector (x, y, z)."""
        return (self.x, self.y, self.z)
    
    # --- Derived Geometric Properties ---
    
    @property
    def r(self) -> float:
        """Radial distance from the beam axis (x-y plane)."""
        return float(self._get_data('r'))
    
    @property
    def R(self) -> float:
        """Radial distance from the origin (3D)."""
        return float(self._get_data('R'))
    
    @property
    def phi(self) -> float:
        """Azimuthal angle."""
        return float(self._get_data('phi'))
    
    @property
    def theta(self) -> float:
        """Polar angle."""
        return float(self._get_data('theta'))
    
    @property
    def eta(self) -> float:
        """Pseudorapidity."""
        return float(self._get_data('eta'))
    
    # --- Contribution Management ---
    
    @property
    def contribution_begin(self) -> int:
        """Starting index in the global contribution array."""
        return int(self._get_data('global_contribution_begin'))
    
    @property
    def contribution_end(self) -> int:
        """Ending index in the global contribution array."""
        return int(self._get_data('global_contribution_end'))
    
    def get_contributions(self) -> List['CaloContribution']:
        """Returns a list of calorimeter contributions for this hit."""
        begin_idx = self.contribution_begin
        end_idx = self.contribution_end
        contrib_df = self._event._calo_contributions_df
        
        if begin_idx >= end_idx or begin_idx < 0:
            return []
        
        # Find valid contribution indices
        contrib_indices = contrib_df.index[begin_idx:end_idx]
        
        # Create CaloContribution objects for each valid index
        contributions = []
        for idx in contrib_indices:
            try:
                contributions.append(CaloContribution(idx, self._event))
            except (KeyError, IndexError):
                print(f"Warning: Invalid contribution index {idx} for hit {self._hit_index}.")
        
        return contributions
    
    # --- Representation ---
    
    def __repr__(self) -> str:
        return f"<CaloHit idx={self._hit_index} det='{self.detector}' E={self.energy:.3f} pos=({self.x:.1f}, {self.y:.1f}, {self.z:.1f})>"


class CaloContribution:
    """Represents a single particle's contribution to a calorimeter hit."""
    
    def __init__(self, global_contrib_index: int, event: 'EDM4hepEvent'):
        """Initialize a CaloContribution object.
        
        Args:
            global_contrib_index (int): The global index of this contribution in the calo_contributions_df.
            event (EDM4hepEvent): The parent event object.
        """
        # Check if contribution index exists at initialization
        if global_contrib_index not in event._calo_contributions_df.index:
            raise IndexError(f"Calo contribution index {global_contrib_index} not found in event {event.event_index}.")
        
        self._contrib_index = global_contrib_index
        self._event = event
    
    @property
    def contrib_index(self) -> int:
        """The global contribution index."""
        return self._contrib_index
    
    def _get_data(self, column: str):
        """Helper to safely get data from the calo contributions dataframe."""
        try:
            return self._event._calo_contributions_df.loc[self._contrib_index, column]
        except KeyError:
            print(f"Warning: Column '{column}' not found for calo contribution {self._contrib_index}.")
            return None
        except IndexError:
            print(f"Warning: Contribution index {self._contrib_index} seems invalid when accessing '{column}'.")
            return None
    
    # --- Basic Properties ---
    
    @property
    def detector(self) -> str:
        """The detector component for this contribution."""
        return str(self._get_data('detector'))
    
    @property
    def pdg(self) -> int:
        """PDG code of the contributing particle."""
        return int(self._get_data('PDG'))
    
    @property
    def energy(self) -> float:
        """Energy of this contribution."""
        return float(self._get_data('energy'))
    
    @property
    def time(self) -> float:
        """Time of this contribution."""
        return float(self._get_data('time'))
    
    # --- Step Position Properties ---
    
    @property
    def step_x(self) -> float:
        """X-coordinate of the energy deposition step."""
        return float(self._get_data('step_x'))
    
    @property
    def step_y(self) -> float:
        """Y-coordinate of the energy deposition step."""
        return float(self._get_data('step_y'))
    
    @property
    def step_z(self) -> float:
        """Z-coordinate of the energy deposition step."""
        return float(self._get_data('step_z'))
    
    @property
    def step_position(self) -> Tuple[float, float, float]:
        """Position vector of the energy deposition step (x, y, z)."""
        return (self.step_x, self.step_y, self.step_z)
    
    # --- Hit Position Properties (if added during processing) ---
    
    @property
    def x(self) -> float:
        """X-coordinate of the associated hit."""
        return float(self._get_data('x'))
    
    @property
    def y(self) -> float:
        """Y-coordinate of the associated hit."""
        return float(self._get_data('y'))
    
    @property
    def z(self) -> float:
        """Z-coordinate of the associated hit."""
        return float(self._get_data('z'))
    
    @property
    def position(self) -> Tuple[float, float, float]:
        """Position vector of the associated hit (x, y, z)."""
        return (self.x, self.y, self.z)
    
    # --- Relationships ---
    
    @property
    def global_hit_index(self) -> int:
        """Global index of the associated CaloHit."""
        return int(self._get_data('global_hit_index'))
    
    def get_hit(self) -> Optional[CaloHit]:
        """Returns the CaloHit associated with this contribution."""
        hit_idx = self.global_hit_index
        try:
            return self._event.get_calo_hit(hit_idx)
        except (KeyError, IndexError):
            print(f"Warning: Associated hit index {hit_idx} not valid for contribution {self._contrib_index}.")
            return None
    
    def get_particle(self) -> Optional['Particle']:
        """Returns the Particle associated with this contribution, or None if not found."""
        links_df = self._event._calo_links_df
        
        if links_df.empty or 'global_contrib_index' not in links_df.columns:
            return None
        
        # Find in the links DataFrame
        matches = links_df[links_df['global_contrib_index'] == self._contrib_index]
        if matches.empty:
            return None
        
        # Get the particle ID(s)
        particle_id = matches.iloc[0]['particle_id']
        
        # Import locally to avoid circular dependency
        from .particle import Particle
        
        try:
            return self._event.get_particle(particle_id)
        except (KeyError, IndexError):
            print(f"Warning: Linked particle ID {particle_id} not found for contribution {self._contrib_index}.")
            return None
    
    # --- Representation ---
    
    def __repr__(self) -> str:
        return f"<CaloContribution idx={self._contrib_index} pdg={self.pdg} E={self.energy:.3f}>"
