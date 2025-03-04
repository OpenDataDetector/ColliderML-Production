import pandas as pd
import numpy as np
import uproot


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


def calculate_R(x, y, z=None):
    """Calculate R (radial distance) from x,y,z coordinates"""
    if z is None:
        return np.sqrt(x**2 + y**2)
    return np.sqrt(x**2 + y**2 + z**2)

def calculate_theta(r, z):
    """Calculate theta (polar angle) from r,z coordinates"""
    return np.arctan2(r, z)

def calculate_eta(theta):
    """Calculate pseudorapidity from theta"""
    return -np.log(np.tan(theta/2))


def load_edm4hep_file(file_path, event_num=None):
    """Load EDM4hep file and return event data.
    
    Args:
        file_path: Path to EDM4hep ROOT file
        event_num: Optional specific event number to load. If None, loads all events.
        
    Returns:
        If event_num is specified: dict of DataFrames for that event
        If event_num is None: list of dicts, where each dict contains DataFrames for one event
    """
    events = uproot.open(file_path)["events"]
    num_events = len(events["MCParticles.PDG"].array())
    
    if event_num is not None:
        return _process_event(events, event_num)
    else:
        return [_process_event(events, i) for i in range(num_events)]

def _process_event(events, event_idx):
    """Process a single event and return its dictionary of DataFrames."""
    return {
        "tracker_df": build_tracker_df(events, event_idx),
        "calo_hits_df": build_calo_df(events, event_idx)[0],
        "calo_contrib_df": build_calo_df(events, event_idx)[1],
        "particles_df": build_particle_df(events, event_idx)[0],
        "parents_df": build_particle_df(events, event_idx)[1],
        "daughters_df": build_particle_df(events, event_idx)[2]
    }

def build_tracker_df(events, event_idx, detector_name=None):
    """Build dataframe from tracker hits for a single event"""
    if detector_name is not None:
        hits = events[detector_name].arrays()
        particle_links = events[f"_{detector_name}_MCParticle"].arrays()
        return _process_tracker_hits(hits, particle_links, detector_name, event_idx)
    
    all_trackers = pixel_readouts + strip_readouts
    dfs = []
    for det in all_trackers:
        if det not in events:
            continue
        hits = events[det].arrays()
        particle_links = events[f"_{det}_MCParticle"].arrays()
        df = _process_tracker_hits(hits, particle_links, det, event_idx)
        df['detector'] = det
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def _process_tracker_hits(hits, particle_links, detector_name, event_idx):
    """Process tracker hits for a single event"""
    hit_dict = {
        'cellID': hits[f"{detector_name}.cellID"][event_idx],
        'EDep': hits[f"{detector_name}.EDep"][event_idx],
        'time': hits[f"{detector_name}.time"][event_idx],
        'pathLength': hits[f"{detector_name}.pathLength"][event_idx],
        'quality': hits[f"{detector_name}.quality"][event_idx],
        'x': hits[f"{detector_name}.position.x"][event_idx],
        'y': hits[f"{detector_name}.position.y"][event_idx],
        'z': hits[f"{detector_name}.position.z"][event_idx],
        'px': hits[f"{detector_name}.momentum.x"][event_idx],
        'py': hits[f"{detector_name}.momentum.y"][event_idx],
        'pz': hits[f"{detector_name}.momentum.z"][event_idx],
        'particle_id': particle_links[f"_{detector_name}_MCParticle.index"][event_idx]
    }
    
    df = pd.DataFrame(hit_dict)
    df['r'] = calculate_R(df['x'], df['y'])
    df['R'] = calculate_R(df['x'], df['y'], df['z'])
    df['phi'] = np.arctan2(df['y'], df['x'])
    df['theta'] = calculate_theta(df['r'], df['z'])
    df['eta'] = calculate_eta(df['theta'])
    df['pt'] = np.sqrt(df['px']**2 + df['py']**2)
    
    return df

def build_calo_df(events, event_idx, detector_name=None):
    """Build dataframe from calorimeter hits for a single event"""
    if detector_name is not None:
        hits = events[detector_name].arrays()
        contributions = events[f"{detector_name}Contributions"].arrays()
        particle_links = events[f"_{detector_name}Contributions_particle"].arrays()
        return _process_calo_hits(hits, contributions, particle_links, detector_name, event_idx)
    
    all_calos = ecal + hcal
    hits_dfs = []
    contrib_dfs = []
    contrib_offset = 0
    
    for det in all_calos:
        if det not in events:
            continue
        hits = events[det].arrays()
        contributions = events[f"{det}Contributions"].arrays()
        particle_links = events[f"_{det}Contributions_particle"].arrays()
        hits_df, contrib_df = _process_calo_hits(hits, contributions, particle_links, det, event_idx, contrib_offset)
        
        hits_df['detector'] = det
        contrib_df['detector'] = det
        
        hits_dfs.append(hits_df)
        contrib_dfs.append(contrib_df)
        contrib_offset += len(contrib_df)
    return (pd.concat(hits_dfs, ignore_index=True) if hits_dfs else pd.DataFrame(),
            pd.concat(contrib_dfs, ignore_index=True) if contrib_dfs else pd.DataFrame())

def _process_calo_hits(hits, contributions, particle_links, detector_name, event_idx, contrib_offset=None):
    """Process calorimeter hits for a single event"""

    hit_dict = {
        'cellID': hits[f"{detector_name}.cellID"][event_idx],
        'energy': hits[f"{detector_name}.energy"][event_idx],
        'x': hits[f"{detector_name}.position.x"][event_idx],
        'y': hits[f"{detector_name}.position.y"][event_idx],
        'z': hits[f"{detector_name}.position.z"][event_idx],
        'contribution_begin': hits[f"{detector_name}.contributions_begin"][event_idx],
        'contribution_end': hits[f"{detector_name}.contributions_end"][event_idx]
    }
    
    hits_df = pd.DataFrame(hit_dict)
    hits_df['r'] = calculate_R(hits_df['x'], hits_df['y'])
    hits_df['R'] = calculate_R(hits_df['x'], hits_df['y'], hits_df['z'])
    hits_df['phi'] = np.arctan2(hits_df['y'], hits_df['x'])
    hits_df['theta'] = calculate_theta(hits_df['r'], hits_df['z'])
    hits_df['eta'] = calculate_eta(hits_df['theta'])
    
    contrib_dict = {
        'PDG': contributions[f"{detector_name}Contributions.PDG"][event_idx],
        'energy': contributions[f"{detector_name}Contributions.energy"][event_idx],
        'time': contributions[f"{detector_name}Contributions.time"][event_idx],
        'step_x': contributions[f"{detector_name}Contributions.stepPosition.x"][event_idx],
        'step_y': contributions[f"{detector_name}Contributions.stepPosition.y"][event_idx],
        'step_z': contributions[f"{detector_name}Contributions.stepPosition.z"][event_idx],
        'particle_id': particle_links[f"_{detector_name}Contributions_particle.index"][event_idx]
    }
    
    contrib_df = pd.DataFrame(contrib_dict)

    # Add hit positions to contributions
    contrib_df = _add_hit_positions_to_contributions(hits_df, contrib_df)

    if contrib_offset is not None:
        hits_df['contribution_begin'] += contrib_offset
        hits_df['contribution_end'] += contrib_offset
    
    return hits_df, contrib_df

def _add_hit_positions_to_contributions(hits_df, contrib_df):
    """Add hit positions (x, y, z) to the contributions dataframe"""
    # First, create the position columns in the contributions dataframe with float data type
    contrib_df['x'] = np.nan
    contrib_df['y'] = np.nan
    contrib_df['z'] = np.nan
    
    # Ensure these columns have float data type
    contrib_df['x'] = contrib_df['x'].astype(float)
    contrib_df['y'] = contrib_df['y'].astype(float)
    contrib_df['z'] = contrib_df['z'].astype(float)

    # set x, y, z for each hit in hits_df
    for _, hit in hits_df.iterrows():
        begin = int(hit['contribution_begin'])
        end = int(hit['contribution_end'])
        if begin == end:
            continue
        
        # make sure we don't go out of bounds
        end = min(end, len(contrib_df))

        contrib_df.iloc[begin:end, contrib_df.columns.get_indexer(['x', 'y', 'z'])] = hit[['x', 'y', 'z']].values
    
    return contrib_df

def build_particle_df(events, event_idx):
    """Build particle dataframes for a single event"""
    particles = events["MCParticles"].arrays()
    particle_dict = {
        'PDG': particles["MCParticles.PDG"][event_idx],
        'generatorStatus': particles["MCParticles.generatorStatus"][event_idx],
        'simulatorStatus': particles["MCParticles.simulatorStatus"][event_idx],
        'charge': particles["MCParticles.charge"][event_idx],
        'time': particles["MCParticles.time"][event_idx],
        'mass': particles["MCParticles.mass"][event_idx],
        'vx': particles["MCParticles.vertex.x"][event_idx],
        'vy': particles["MCParticles.vertex.y"][event_idx],
        'vz': particles["MCParticles.vertex.z"][event_idx],
        'px': particles["MCParticles.momentum.x"][event_idx],
        'py': particles["MCParticles.momentum.y"][event_idx],
        'pz': particles["MCParticles.momentum.z"][event_idx],
        'endpoint_x': particles["MCParticles.endpoint.x"][event_idx],
        'endpoint_y': particles["MCParticles.endpoint.y"][event_idx],
        'endpoint_z': particles["MCParticles.endpoint.z"][event_idx],
        'parents_begin': particles["MCParticles.parents_begin"][event_idx],
        'parents_end': particles["MCParticles.parents_end"][event_idx],
        'daughters_begin': particles["MCParticles.daughters_begin"][event_idx],
        'daughters_end': particles["MCParticles.daughters_end"][event_idx],
    }
    
    parents = events["_MCParticles_parents"].arrays()
    parent_dict = {
        'particle_id': parents["_MCParticles_parents.index"][event_idx],
        'collectionID': parents["_MCParticles_parents.collectionID"][event_idx]
    }
    
    daughters = events["_MCParticles_daughters"].arrays()
    daughter_dict = {
        'particle_id': daughters["_MCParticles_daughters.index"][event_idx],
        'collectionID': daughters["_MCParticles_daughters.collectionID"][event_idx]
    }
    
    particles_df = pd.DataFrame(particle_dict)
    parents_df = pd.DataFrame(parent_dict)
    daughters_df = pd.DataFrame(daughter_dict)
    
    particles_df['pt'] = np.sqrt(particles_df['px']**2 + particles_df['py']**2)
    particles_df['p'] = np.sqrt(particles_df['px']**2 + particles_df['py']**2 + particles_df['pz']**2)
    particles_df['eta'] = np.arcsinh(particles_df['pz']/particles_df['pt'])
    particles_df['phi'] = np.arctan2(particles_df['py'], particles_df['px'])
    
    return particles_df, parents_df, daughters_df

def create_truth_clusters(event, detector='ECalBarrelCollection'):
    """Create truth-based clustering from EDM4hep event data.
    
    Args:
        event (dict): Dictionary containing EDM4hep event data with keys:
            - tracker_df
            - calo_hits_df
            - particles_df
            - calo_contrib_df
            - parents_df
            - daughters_df
        detector (str): Name of detector collection to process
            
    Returns:
        pd.DataFrame: DataFrame containing cell information with clustering results:
            - All original cell information
            - highest_energy_particle_id: ID of particle contributing most energy
    """
    # Extract and filter relevant dataframes
    cells_df = event["calo_hits_df"]
    hits_df = event["calo_contrib_df"]
    
    # Filter for specific detector
    cells_df = cells_df[cells_df.detector == detector]
    hits_df = hits_df[hits_df.detector == detector]
    
    # Create hit to cell mapping
    hit_to_cell = pd.DataFrame({'hit_idx': np.arange(len(hits_df))})
    cells_df['cellID'] = cells_df['cellID'].astype('uint64')
    
    # Map hits to cells
    cell_starts = cells_df.contribution_begin.values
    cell_ends = cells_df.contribution_end.values
    hit_indices = hit_to_cell.hit_idx.values
    
    cell_idx = np.searchsorted(cell_starts, hit_indices, side='right') - 1
    valid_hits = (hit_indices >= cell_starts[cell_idx]) & (hit_indices < cell_ends[cell_idx])
    hit_to_cell['cell_idx'] = np.where(valid_hits, cell_idx, -1)
    
    # Get positions for valid hits
    valid_hits_df = hit_to_cell[hit_to_cell.cell_idx >= 0]
    positions = pd.merge(
        valid_hits_df,
        cells_df[['x', 'y', 'z', 'r', 'phi', 'eta', 'cellID']],
        left_on='cell_idx',
        right_index=True,
        how='left'
    )
    
    # Update hits with position information
    hits_df = hits_df.copy()
    hits_df.loc[valid_hits_df.hit_idx, ['x', 'y', 'z', 'r', 'phi', 'eta']] = \
        positions[['x', 'y', 'z', 'r', 'phi', 'eta']].values
    hits_df.loc[valid_hits_df.hit_idx, 'cellID'] = positions['cellID'].astype('uint64')
    
    # Aggregate hits by cell and particle
    particle_total_hits = hits_df.groupby(['cellID', 'particle_id']).agg({
        'energy': 'sum',
        'time': 'mean',
        'x': 'first',
        'y': 'first',
        'z': 'first',
        'detector': 'first'
    }).reset_index()
    
    # Find highest energy particle per cell
    highest_energy_particle = (particle_total_hits
        .sort_values(by='energy', ascending=False)
        .groupby('cellID')
        .first()[['particle_id']]
        .reset_index())
    
    # Create final dataframe with cluster labels
    calo_hits_df = cells_df.merge(
        highest_energy_particle, 
        on='cellID'
    ).rename(columns={'particle_id': 'highest_energy_particle_id'})
    
    return calo_hits_df