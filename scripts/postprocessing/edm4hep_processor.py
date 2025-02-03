"""
EDM4HEP data processing utilities for converting to pandas DataFrames.
"""

import numpy as np
import pandas as pd
import uproot
from typing import List, Dict, Any, Tuple, Optional

def process_hits_data(
    event_id: int,
    tracker_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Process hit data for a single event using EDM4HEP tracker hits.
    
    Args:
        event_id: Event number
        tracker_df: DataFrame containing EDM4HEP tracker hits
        
    Returns:
        DataFrame containing hit data for this event
    """
    # Select relevant columns
    hit_columns = [
        "cellID",
        "EDep",
        "time",
        "pathLength",
        "quality",
        "x",
        "y",
        "z",
        "px",
        "py",
        "pz",
        "particle_id",
        "detector",
    ]
    
    event_hits = tracker_df[hit_columns].copy()
    event_hits["event_id"] = event_id
    
    return event_hits

def process_track_data(
    event_id: int,
    tracks_df: pd.DataFrame,
    track_states_df: Optional[pd.DataFrame] = None,
    track_hits_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Process track data for a single event.
    
    Args:
        event_id: Event number
        tracks_df: DataFrame containing track parameters
        track_states_df: Optional DataFrame containing track states
        track_hits_df: Optional DataFrame containing track-hit associations
        
    Returns:
        DataFrame containing processed track data
    """
    event_tracks = tracks_df.copy()
    event_tracks["event_id"] = event_id
    
    # Add track states if available
    if track_states_df is not None:
        # Merge track states (IP, first hit, last hit, etc.)
        event_tracks = pd.merge(
            event_tracks,
            track_states_df,
            left_index=True,
            right_index=True,
            suffixes=("", "_state")
        )
    
    # Add hit associations if available
    if track_hits_df is not None:
        # Group hits by track and create hit ID lists
        hit_groups = track_hits_df.groupby("track_id")["hit_id"].agg(list)
        event_tracks["hit_ids"] = event_tracks.index.map(hit_groups)
    
    return event_tracks

def process_particles_data(
    event_id: int,
    particles_df: pd.DataFrame,
    parents_df: Optional[pd.DataFrame] = None,
    daughters_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Process particle data for a single event.
    
    Args:
        event_id: Event number
        particles_df: DataFrame containing particle data
        parents_df: Optional DataFrame containing parent relationships
        daughters_df: Optional DataFrame containing daughter relationships
        
    Returns:
        DataFrame containing processed particle data
    """
    event_particles = particles_df.copy()
    event_particles["event_id"] = event_id
    
    # Add parent/daughter information if available
    if parents_df is not None:
        event_particles = pd.merge(
            event_particles,
            parents_df,
            left_index=True,
            right_index=True,
            suffixes=("", "_parent")
        )
    
    if daughters_df is not None:
        event_particles = pd.merge(
            event_particles,
            daughters_df,
            left_index=True,
            right_index=True,
            suffixes=("", "_daughter")
        )
    
    return event_particles

def process_calorimeter_data(
    event_id: int,
    calo_hits_df: pd.DataFrame,
    calo_contrib_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Process calorimeter data for a single event.
    
    Args:
        event_id: Event number
        calo_hits_df: DataFrame containing calorimeter hits
        calo_contrib_df: Optional DataFrame containing hit contributions
        
    Returns:
        DataFrame containing processed calorimeter data
    """
    event_calo = calo_hits_df.copy()
    event_calo["event_id"] = event_id
    
    # Add contribution information if available
    if calo_contrib_df is not None:
        event_calo = pd.merge(
            event_calo,
            calo_contrib_df,
            left_index=True,
            right_index=True,
            suffixes=("", "_contrib")
        )
    
    return event_calo

def load_edm4hep_event(
    file_path: str,
    event_num: int = 0
) -> Dict[str, pd.DataFrame]:
    """
    Load a single event from an EDM4HEP file.
    
    Args:
        file_path: Path to EDM4HEP root file
        event_num: Event number to load
        
    Returns:
        Dictionary containing DataFrames for different components
    """
    import uproot
    
    # Open the file
    events = uproot.open(file_path)
    
    # Process event components
    event_data = {}
    
    try:
        # Load tracker hits
        tracker = events["events/TrackerHits"].arrays()
        event_data["tracker_df"] = pd.DataFrame({
            "cellID": tracker["TrackerHits.cellID"][event_num],
            "EDep": tracker["TrackerHits.EDep"][event_num],
            "time": tracker["TrackerHits.time"][event_num],
            "pathLength": tracker["TrackerHits.pathLength"][event_num],
            "quality": tracker["TrackerHits.quality"][event_num],
            "x": tracker["TrackerHits.position.x"][event_num],
            "y": tracker["TrackerHits.position.y"][event_num],
            "z": tracker["TrackerHits.position.z"][event_num],
            "px": tracker["TrackerHits.momentum.x"][event_num],
            "py": tracker["TrackerHits.momentum.y"][event_num],
            "pz": tracker["TrackerHits.momentum.z"][event_num],
        })
    except KeyError:
        print("No tracker hits found")
        
    try:
        # Load tracks
        tracks = events["events/Tracks"].arrays()
        event_data["tracks_df"] = pd.DataFrame({
            "type": tracks["Tracks.type"][event_num],
            "chi2": tracks["Tracks.chi2"][event_num],
            "ndf": tracks["Tracks.ndf"][event_num],
            "dEdx": tracks["Tracks.dEdx"][event_num],
            "dEdxError": tracks["Tracks.dEdxError"][event_num],
            "radiusOfInnermostHit": tracks["Tracks.radiusOfInnermostHit"][event_num],
        })
        
        # Load track states
        track_states = events["events/TrackStates"].arrays()
        event_data["track_states_df"] = pd.DataFrame({
            "location": track_states["TrackStates.location"][event_num],
            "d0": track_states["TrackStates.D0"][event_num],
            "phi": track_states["TrackStates.phi"][event_num],
            "omega": track_states["TrackStates.omega"][event_num],
            "z0": track_states["TrackStates.Z0"][event_num],
            "tanLambda": track_states["TrackStates.tanLambda"][event_num],
            "time": track_states["TrackStates.time"][event_num],
        })
        
        # Load track-hit associations
        track_hits = events["events/TrackerHitRelations"].arrays()
        event_data["track_hits_df"] = pd.DataFrame({
            "track_id": track_hits["TrackerHitRelations.track"][event_num],
            "hit_id": track_hits["TrackerHitRelations.hit"][event_num],
            "weight": track_hits["TrackerHitRelations.weight"][event_num],
        })
    except KeyError:
        print("No track data found")
        
    try:
        # Load particles
        particles = events["events/MCParticles"].arrays()
        event_data["particles_df"] = pd.DataFrame({
            "PDG": particles["MCParticles.PDG"][event_num],
            "generatorStatus": particles["MCParticles.generatorStatus"][event_num],
            "simulatorStatus": particles["MCParticles.simulatorStatus"][event_num],
            "charge": particles["MCParticles.charge"][event_num],
            "time": particles["MCParticles.time"][event_num],
            "mass": particles["MCParticles.mass"][event_num],
            "vx": particles["MCParticles.vertex.x"][event_num],
            "vy": particles["MCParticles.vertex.y"][event_num],
            "vz": particles["MCParticles.vertex.z"][event_num],
            "px": particles["MCParticles.momentum.x"][event_num],
            "py": particles["MCParticles.momentum.y"][event_num],
            "pz": particles["MCParticles.momentum.z"][event_num],
            "endpoint_x": particles["MCParticles.endpoint.x"][event_num],
            "endpoint_y": particles["MCParticles.endpoint.y"][event_num],
            "endpoint_z": particles["MCParticles.endpoint.z"][event_num],
        })
    except KeyError:
        print("No particle data found")
        
    try:
        # Load calorimeter hits
        calo = events["events/CalorimeterHits"].arrays()
        event_data["calo_hits_df"] = pd.DataFrame({
            "cellID": calo["CalorimeterHits.cellID"][event_num],
            "energy": calo["CalorimeterHits.energy"][event_num],
            "time": calo["CalorimeterHits.time"][event_num],
            "x": calo["CalorimeterHits.position.x"][event_num],
            "y": calo["CalorimeterHits.position.y"][event_num],
            "z": calo["CalorimeterHits.position.z"][event_num],
        })
    except KeyError:
        print("No calorimeter hits found")
    
    return event_data 