import pandas as pd
import numpy as np


# Basic event metadata
event_info = [
    'EventHeader',
    'EventHeader/EventHeader.eventNumber',
    'EventHeader/EventHeader.runNumber',
    'EventHeader/EventHeader.timeStamp',
    'EventHeader/EventHeader.weight'
]

# Parameters (configuration/metadata)
parameters = [
    'PARAMETERS/_intMap',
    'PARAMETERS/_floatMap', 
    'PARAMETERS/_stringMap',
    'PARAMETERS/_doubleMap'
]

# Pixel Detectors
pixel_readouts = [
    'PixelBarrelReadout',  # Inner tracking
    'PixelEndcapReadout',
]

# Strip Detectors
strip_readouts = [
    'ShortStripBarrelReadout',  # Middle tracking
    'ShortStripEndcapReadout',
    'LongStripBarrelReadout',   # Outer tracking
    'LongStripEndcapReadout'
]

# Electromagnetic Calorimeter
ecal = [
    'ECalBarrelCollection',
    'ECalEndcapCollection',
    # Fields: .cellID, .energy, .position.(x,y,z)
    # Contributions tracking energy deposits
]

# Hadronic Calorimeter  
hcal = [
    'HCalBarrelCollection',
    'HCalEndcapCollection',
    # Similar structure to ECAL
]

# Monte Carlo Particles
mc_particles = [
    'MCParticles',
]


def build_tracker_df(event, detector_name):
    """Build dataframe from tracker hits and their particle links"""
    hits = event[detector_name].arrays()
    particle_links = event[f"_{detector_name}_MCParticle"].arrays()
    
    hit_dict = {
        'cellID': hits[f"{detector_name}.cellID"][0],
        'EDep': hits[f"{detector_name}.EDep"][0],
        'time': hits[f"{detector_name}.time"][0],
        'pathLength': hits[f"{detector_name}.pathLength"][0],
        'quality': hits[f"{detector_name}.quality"][0],
        'x': hits[f"{detector_name}.position.x"][0],
        'y': hits[f"{detector_name}.position.y"][0],
        'z': hits[f"{detector_name}.position.z"][0],
        'px': hits[f"{detector_name}.momentum.x"][0],
        'py': hits[f"{detector_name}.momentum.y"][0],
        'pz': hits[f"{detector_name}.momentum.z"][0],
        'particle_id': particle_links[f"_{detector_name}_MCParticle.index"][0]
    }
    
    df = pd.DataFrame(hit_dict)
    
    # Add derived quantities
    df['r'] = np.sqrt(df['x']**2 + df['y']**2)
    df['phi'] = np.arctan2(df['y'], df['x'])
    df['pt'] = np.sqrt(df['px']**2 + df['py']**2)
    
    return df

def build_calo_df(event, detector_name):
    """Build dataframe from calorimeter hits and their particle links"""
    hits = event[detector_name].arrays()
    contributions = event[f"{detector_name}Contributions"].arrays()
    particle_links = event[f"_{detector_name}Contributions_particle"].arrays()
    
    # Build hits dataframe
    hit_dict = {
        'cellID': hits[f"{detector_name}.cellID"][0],
        'energy': hits[f"{detector_name}.energy"][0],
        'x': hits[f"{detector_name}.position.x"][0],
        'y': hits[f"{detector_name}.position.y"][0],
        'z': hits[f"{detector_name}.position.z"][0],
        'contribution_begin': hits[f"{detector_name}.contributions_begin"][0],
        'contribution_end': hits[f"{detector_name}.contributions_end"][0]
    }
    
    hits_df = pd.DataFrame(hit_dict)
    
    # Add derived quantities
    hits_df['r'] = np.sqrt(hits_df['x']**2 + hits_df['y']**2)
    hits_df['phi'] = np.arctan2(hits_df['y'], hits_df['x'])
    
    # Build contributions dataframe
    contrib_dict = {
        'PDG': contributions[f"{detector_name}Contributions.PDG"][0],
        'energy': contributions[f"{detector_name}Contributions.energy"][0],
        'time': contributions[f"{detector_name}Contributions.time"][0],
        'x': contributions[f"{detector_name}Contributions.stepPosition.x"][0],
        'y': contributions[f"{detector_name}Contributions.stepPosition.y"][0],
        'z': contributions[f"{detector_name}Contributions.stepPosition.z"][0],
        'particle_id': particle_links[f"_{detector_name}Contributions_particle.index"][0]
    }
    
    contrib_df = pd.DataFrame(contrib_dict)
    
    return hits_df, contrib_df

def build_particle_df(event):
    """Build dataframe from MCParticles collection and return separate parent/daughter dataframes"""
    # Main particle properties
    particles = event["MCParticles"].arrays()
    particle_dict = {
        'PDG': particles["MCParticles.PDG"][0],
        'generatorStatus': particles["MCParticles.generatorStatus"][0],
        'simulatorStatus': particles["MCParticles.simulatorStatus"][0],
        'charge': particles["MCParticles.charge"][0],
        'time': particles["MCParticles.time"][0],
        'mass': particles["MCParticles.mass"][0],
        'vx': particles["MCParticles.vertex.x"][0],
        'vy': particles["MCParticles.vertex.y"][0],
        'vz': particles["MCParticles.vertex.z"][0],
        'px': particles["MCParticles.momentum.x"][0],
        'py': particles["MCParticles.momentum.y"][0],
        'pz': particles["MCParticles.momentum.z"][0],
        'endpoint_x': particles["MCParticles.endpoint.x"][0],
        'endpoint_y': particles["MCParticles.endpoint.y"][0],
        'endpoint_z': particles["MCParticles.endpoint.z"][0],
    }
    
    # Parent relationships
    parents = event["_MCParticles_parents"].arrays()
    parent_dict = {
        'index': parents["_MCParticles_parents.index"][0],
        'collectionID': parents["_MCParticles_parents.collectionID"][0]
    }
    
    # Daughter relationships
    daughters = event["_MCParticles_daughters"].arrays()
    daughter_dict = {
        'index': daughters["_MCParticles_daughters.index"][0],
        'collectionID': daughters["_MCParticles_daughters.collectionID"][0]
    }
    
    # Create dataframes
    particles_df = pd.DataFrame(particle_dict)
    parents_df = pd.DataFrame(parent_dict)
    daughters_df = pd.DataFrame(daughter_dict)
    
    # Add derived quantities to main particle dataframe
    particles_df['pt'] = np.sqrt(particles_df['px']**2 + particles_df['py']**2)
    particles_df['p'] = np.sqrt(particles_df['px']**2 + particles_df['py']**2 + particles_df['pz']**2)
    particles_df['eta'] = np.arcsinh(particles_df['pz']/particles_df['pt'])
    particles_df['phi'] = np.arctan2(particles_df['py'], particles_df['px'])
    
    return particles_df, parents_df, daughters_df