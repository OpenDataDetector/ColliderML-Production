import pandas as pd
import numpy as np
import uproot
from typing import Optional

# Define detector component lists (copied from edm4hep_utils.py)
pixel_readouts = [
    'PixelBarrelReadout',
    'PixelEndcapReadout',
]
strip_readouts = [
    'ShortStripBarrelReadout',
    'ShortStripEndcapReadout',
    'LongStripBarrelReadout',
    'LongStripEndcapReadout'
]
ecal = [
    'ECalBarrelCollection',
    'ECalEndcapCollection',
]
hcal = [
    'HCalBarrelCollection',
    'HCalEndcapCollection',
]
all_trackers = pixel_readouts + strip_readouts
all_calos = ecal + hcal

# --- Helper Functions (copied/adapted from edm4hep_utils.py) ---

def _calculate_R(x, y, z=None):
    """Calculate R (radial distance) from x,y,z coordinates"""
    if z is None:
        return np.sqrt(x**2 + y**2)
    return np.sqrt(x**2 + y**2 + z**2)

def _calculate_theta(r, z):
    """Calculate theta (polar angle) from r,z coordinates"""
    # Add small epsilon to avoid division by zero or issues at z=0
    return np.arctan2(r, z + 1e-15)

def _calculate_eta(theta):
    """Calculate pseudorapidity from theta"""
    # Ensure tan(theta/2) is positive
    tan_theta_half = np.tan(theta / 2)
    # Handle cases close to zero or pi which might result in negative tan_theta_half due to precision
    # or theta values slightly outside [0, pi]
    valid_tan = np.where(tan_theta_half > 1e-15, tan_theta_half, 1e-15)
    eta = -np.log(valid_tan)
    # Handle potential infinities if theta is exactly 0 or pi
    eta = np.where(np.isinf(eta), np.sign(eta) * 1e6, eta) # Replace inf with large number
    return eta


def get_simulator_status_bits(int_val):
    """
    Decodes the simulator status integer into a dictionary of boolean flags.
    Based on edm4hep::MCRecoParticleStatus definition (check specific EDM4hep version).
    Bits are read from right to left (LSB is bit 0).
      static const int BITCreatedInSimulation = 30;
      static const int BITBackscatter = 29 ;
      static const int BITVertexIsNotEndpointOfParent = 28 ;
      static const int BITDecayedInTracker = 27 ;
      static const int BITDecayedInCalorimeter = 26 ;
      static const int BITLeftDetector = 25 ;
      static const int BITStopped = 24 ;
      static const int BITOverlay = 23 ;
    """
    # Ensure input is integer
    status = int(int_val)
    return {
        'created_in_simulation': bool(status & (1 << 30)),
        'backscatter': bool(status & (1 << 29)),
        'vertex_not_endpoint': bool(status & (1 << 28)),
        'decayed_in_tracker': bool(status & (1 << 27)),
        'decayed_in_calorimeter': bool(status & (1 << 26)),
        'has_left_detector': bool(status & (1 << 25)),
        'stopped': bool(status & (1 << 24)),
        'overlay': bool(status & (1 << 23))
    }

# --- DataFrame Building Functions (adapted from edm4hep_utils.py) ---

def _build_particle_df(events_tree, event_idx):
    """Build particle dataframe and parent/daughter link dataframes for a single event"""
    particles = events_tree["MCParticles"].arrays(entry_start=event_idx, entry_stop=event_idx+1)
    
    # Check if event exists / has particles
    if len(particles["MCParticles.PDG"]) == 0 or len(particles["MCParticles.PDG"][0]) == 0:
        empty_cols_particles = ['PDG', 'generatorStatus', 'simulatorStatus', 'charge', 'time', 'mass', 'vx', 'vy', 'vz', 'px', 'py', 'pz', 'endpoint_x', 'endpoint_y', 'endpoint_z', 'parents_begin', 'parents_end', 'daughters_begin', 'daughters_end', 'pt', 'p', 'eta', 'phi']
        empty_cols_links = ['particle_id', 'collectionID']
        return pd.DataFrame(columns=empty_cols_particles), pd.DataFrame(columns=empty_cols_links), pd.DataFrame(columns=empty_cols_links)

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
        'parents_begin': particles["MCParticles.parents_begin"][0],
        'parents_end': particles["MCParticles.parents_end"][0],
        'daughters_begin': particles["MCParticles.daughters_begin"][0],
        'daughters_end': particles["MCParticles.daughters_end"][0],
    }
    particles_df = pd.DataFrame(particle_dict)

    # Handle potentially missing parent/daughter link branches gracefully
    try:
        parents = events_tree["_MCParticles_parents"].arrays(entry_start=event_idx, entry_stop=event_idx+1)
        parent_dict = {
            'particle_id': parents["_MCParticles_parents.index"][0],
            'collectionID': parents["_MCParticles_parents.collectionID"][0]
        }
        parents_df = pd.DataFrame(parent_dict)
    except KeyError:
        print("Warning: Branch '_MCParticles_parents' not found.")
        parents_df = pd.DataFrame(columns=['particle_id', 'collectionID'])

    try:
        daughters = events_tree["_MCParticles_daughters"].arrays(entry_start=event_idx, entry_stop=event_idx+1)
        daughter_dict = {
            'particle_id': daughters["_MCParticles_daughters.index"][0],
            'collectionID': daughters["_MCParticles_daughters.collectionID"][0]
        }
        daughters_df = pd.DataFrame(daughter_dict)
    except KeyError:
        print("Warning: Branch '_MCParticles_daughters' not found.")
        daughters_df = pd.DataFrame(columns=['particle_id', 'collectionID'])


    # Calculate derived quantities only if dataframe is not empty
    if not particles_df.empty:
        particles_df['pt'] = np.sqrt(particles_df['px']**2 + particles_df['py']**2)
        particles_df['p'] = np.sqrt(particles_df['pt']**2 + particles_df['pz']**2)
        # Handle pt=0 case for eta calculation
        particles_df['eta'] = np.arcsinh(particles_df['pz'] / np.where(particles_df['pt'] == 0, 1e-15, particles_df['pt']))
        particles_df['phi'] = np.arctan2(particles_df['py'], particles_df['px'])

        # Ensure index columns have the correct data types
        for col in ['parents_begin', 'parents_end', 'daughters_begin', 'daughters_end']:
             if col in particles_df: particles_df[col] = particles_df[col].astype(int)
        for col in ['particle_id', 'collectionID']:
            if col in parents_df: parents_df[col] = parents_df[col].astype(int)
            if col in daughters_df: daughters_df[col] = daughters_df[col].astype(int)

    return particles_df, parents_df, daughters_df

def _process_single_tracker(events_tree, event_idx, detector_name):
    """Process hits and links for a single tracker detector."""
    hits = events_tree[detector_name].arrays(entry_start=event_idx, entry_stop=event_idx+1)
    try:
        particle_links = events_tree[f"_{detector_name}_MCParticle"].arrays(entry_start=event_idx, entry_stop=event_idx+1)
    except:
        particle_links = events_tree[f"_{detector_name}_particle"].arrays(entry_start=event_idx, entry_stop=event_idx+1)

    # Check if event exists / has hits for this detector
    if len(hits[f"{detector_name}.cellID"]) == 0:
        return pd.DataFrame()

    hit_dict = {
        'cellID': hits[f"{detector_name}.cellID"][0],
        'time': hits[f"{detector_name}.time"][0],
        'pathLength': hits[f"{detector_name}.pathLength"][0],
        'quality': hits[f"{detector_name}.quality"][0],
        'x': hits[f"{detector_name}.position.x"][0],
        'y': hits[f"{detector_name}.position.y"][0],
        'z': hits[f"{detector_name}.position.z"][0],
        'px': hits[f"{detector_name}.momentum.x"][0],
        'py': hits[f"{detector_name}.momentum.y"][0],
        'pz': hits[f"{detector_name}.momentum.z"][0],
    }

    # Handle different EDep naming conventions
    if f"{detector_name}.EDep" in events_tree[detector_name]:
        hit_dict['EDep'] = hits[f"{detector_name}.EDep"][0]
    elif f"{detector_name}.eDep" in events_tree[detector_name]:
         hit_dict['EDep'] = hits[f"{detector_name}.eDep"][0]
    else:
        hit_dict['EDep'] = np.nan # Or some default value

    # Try both particle link formats
    try:
        hit_dict['particle_id'] = particle_links[f"_{detector_name}_MCParticle.index"][0]
    except:
        hit_dict['particle_id'] = particle_links[f"_{detector_name}_particle.index"][0]

    hits_df = pd.DataFrame(hit_dict)

    return hits_df


def _build_tracker_df(events_tree, event_idx):
    """Build combined dataframe for tracker hits for a single event"""
    all_hits_dfs = []

    for det in all_trackers:
        if det not in events_tree:
            continue

        hits_df = _process_single_tracker(events_tree, event_idx, det)
        hits_df['detector'] = det
        all_hits_dfs.append(hits_df)

    # Concatenate all hits and links
    combined_hits_df = pd.concat(all_hits_dfs, ignore_index=True) if all_hits_dfs else pd.DataFrame()

    # Calculate derived geometric quantities on the combined dataframe
    if not combined_hits_df.empty:
        combined_hits_df['r'] = _calculate_R(combined_hits_df['x'], combined_hits_df['y'])
        combined_hits_df['R'] = _calculate_R(combined_hits_df['x'], combined_hits_df['y'], combined_hits_df['z'])
        combined_hits_df['phi'] = np.arctan2(combined_hits_df['y'], combined_hits_df['x'])
        combined_hits_df['theta'] = _calculate_theta(combined_hits_df['r'], combined_hits_df['z'])
        combined_hits_df['eta'] = _calculate_eta(combined_hits_df['theta'])
        combined_hits_df['pt'] = np.sqrt(combined_hits_df['px']**2 + combined_hits_df['py']**2)

    return combined_hits_df


def _process_single_calo(events_tree, event_idx, detector_name, contrib_offset=None):
    """Process hits, contributions, and links for a single calorimeter detector."""
    hits = events_tree[detector_name].arrays(entry_start=event_idx, entry_stop=event_idx+1)
    contributions = events_tree[f"{detector_name}Contributions"].arrays(entry_start=event_idx, entry_stop=event_idx+1)
    particle_links = events_tree[f"_{detector_name}Contributions_particle"].arrays(entry_start=event_idx, entry_stop=event_idx+1)

     # Check if event exists / has hits for this detector
    if len(hits[f"{detector_name}.cellID"]) == 0:
        return pd.DataFrame(), pd.DataFrame()

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

    contrib_dict = {
        'PDG': contributions[f"{detector_name}Contributions.PDG"][0],
        'energy': contributions[f"{detector_name}Contributions.energy"][0],
        'time': contributions[f"{detector_name}Contributions.time"][0],
        'step_x': contributions[f"{detector_name}Contributions.stepPosition.x"][0],
        'step_y': contributions[f"{detector_name}Contributions.stepPosition.y"][0],
        'step_z': contributions[f"{detector_name}Contributions.stepPosition.z"][0],
        'particle_id': particle_links[f"_{detector_name}Contributions_particle.index"][0]
    }
    contrib_df = pd.DataFrame(contrib_dict)

    # Add hit positions to contributions
    contrib_df = _add_hit_positions_to_contributions(hits_df, contrib_df)

    if contrib_offset is not None:
        hits_df['contribution_begin'] += contrib_offset
        hits_df['contribution_end'] += contrib_offset
    
    return hits_df, contrib_df

def _build_calo_df(events_tree, event_idx):
    """Build combined dataframes for calo hits, contributions, and particle links for a single event"""
    all_hits_dfs = []
    all_contrib_dfs = []
    global_contrib_offset = 0

    for det in all_calos:
        if det not in events_tree:
            continue

        hits_df, contrib_df = _process_single_calo(events_tree, event_idx, det, global_contrib_offset)

        # Add detector info
        contrib_df['detector'] = det
        all_contrib_dfs.append(contrib_df)
        all_hits_dfs.append(hits_df)

        # Update offsets for the next detector
        global_contrib_offset += len(contrib_df) # Use length of contributions DF

    # Concatenate all dataframes
    combined_hits_df = pd.concat(all_hits_dfs, ignore_index=True) if all_hits_dfs else pd.DataFrame()
    combined_contrib_df = pd.concat(all_contrib_dfs, ignore_index=True) if all_contrib_dfs else pd.DataFrame()

    # Calculate derived geometric quantities on the combined hits dataframe
    combined_hits_df['r'] = _calculate_R(combined_hits_df['x'], combined_hits_df['y'])
    combined_hits_df['R'] = _calculate_R(combined_hits_df['x'], combined_hits_df['y'], combined_hits_df['z'])
    combined_hits_df['phi'] = np.arctan2(combined_hits_df['y'], combined_hits_df['x'])
    combined_hits_df['theta'] = _calculate_theta(combined_hits_df['r'], combined_hits_df['z'])
    combined_hits_df['eta'] = _calculate_eta(combined_hits_df['theta'])

    return combined_hits_df, combined_contrib_df

def _add_hit_positions_to_contributions(hits_df, contrib_df):
    """Add hit positions (x, y, z) to the contributions dataframe"""
    num_contribs = hits_df.contribution_end - hits_df.contribution_begin
    contrib_df["cellID"] = np.repeat(hits_df.index.values, num_contribs)

    # merge the contrib_df with hits_df to get the [x,y,z] of the hit
    contrib_df = contrib_df.merge(hits_df[["x", "y", "z"]], left_on="cellID", right_index=True, how="left")
    return contrib_df


# --- Main Loading Function ---

def load_event_data(file_path, event_idx):
    """
    Loads all relevant data for a single event from an EDM4hep ROOT file.

    Args:
        file_path (str): Path to the EDM4hep ROOT file.
        event_idx (int): The 0-based index of the event to load.

    Returns:
        dict: A dictionary containing the following Pandas DataFrames:
            - 'particles': MCParticles data.
            - 'parents': Parent links for MCParticles.
            - 'daughters': Daughter links for MCParticles.
            - 'tracker_hits': Combined tracker hits data.
            - 'tracker_links': Links between tracker hits and MCParticles.
            - 'calo_hits': Combined calorimeter hits data.
            - 'calo_contributions': Combined calorimeter contribution data.
            - 'calo_links': Links between calo contributions and MCParticles.
            - 'event_header': Basic event header info (as dict for now).
             Returns None if the event cannot be loaded or file not found.
    """
    try:
        root_file = uproot.open(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error opening file {file_path}: {e}")
        return None

    if "events" not in root_file:
        print(f"Error: 'events' tree not found in {file_path}")
        root_file.close()
        return None

    events_tree = root_file["events"]

    # Basic check for event index validity
    num_events = events_tree.num_entries
    if event_idx < 0 or event_idx >= num_events:
        print(f"Error: Event index {event_idx} is out of bounds (0-{num_events-1}).")
        root_file.close()
        return None

    print(f"Loading event {event_idx} from {file_path}...")

    particles_df, parents_df, daughters_df = _build_particle_df(events_tree, event_idx)
    print(f"  Loaded {len(particles_df)} particles.")

    tracker_hits_df = _build_tracker_df(events_tree, event_idx)
    print(f"  Loaded {len(tracker_hits_df)} tracker hits.")

    calo_hits_df, calo_contrib_df = _build_calo_df(events_tree, event_idx)
    print(f"  Loaded {len(calo_hits_df)} calo hits and {len(calo_contrib_df)} contributions.")

    # Load basic event header info
    header_data = {}
    header_branch = "EventHeader"
    if header_branch in events_tree:
         header = events_tree[header_branch].arrays(entry_start=event_idx, entry_stop=event_idx+1)
         if len(header[f"{header_branch}.eventNumber"]) > 0: # Check if event exists
             header_data = {
                 'eventNumber': header[f"{header_branch}.eventNumber"][0],
                 'runNumber': header[f"{header_branch}.runNumber"][0],
                 'timeStamp': header[f"{header_branch}.timeStamp"][0],
                 'weight': header[f"{header_branch}.weight"][0]
            }
         else:
             print("Warning: EventHeader branch exists but seems empty for this event.")
    else:
        print("Warning: EventHeader branch not found.")


    root_file.close()

    return {
        "particles": particles_df,
        "parents": parents_df,
        "daughters": daughters_df,
        "tracker_hits": tracker_hits_df,
        "calo_hits": calo_hits_df,
        "calo_contributions": calo_contrib_df,
        "event_header": header_data
    }