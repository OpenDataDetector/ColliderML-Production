#!/usr/bin/env python3
"""
Canonical Parquet schemas for ColliderML postprocessing outputs.

This module defines the intended on-disk Arrow types for each column in the
Parquet files written by the postprocessing scripts. These schemas are used
by the Parquet utilities to ensure we keep low-bit-width integer and float32
types on disk, even when Pandas / PyArrow would otherwise widen them.
"""

from __future__ import annotations

import pyarrow as pa


def list_of(item_type: pa.DataType) -> pa.ListType:
    """Return a list type with the given item type."""
    return pa.list_(item_type)


def nested_list_of(item_type: pa.DataType) -> pa.ListType:
    """Return a two-level nested list type (list<list<item_type>>)."""
    return pa.list_(pa.list_(item_type))


# Truth particles: one row per event, list-valued columns per event
PARTICLES_PARQUET_TYPES = {
    # Global event index
    "event_id": pa.uint32(),
    # Particle identifiers and properties
    "particle_id": list_of(pa.uint64()),
    "pdg_id": list_of(pa.int64()),
    "mass": list_of(pa.float32()),
    "energy": list_of(pa.float32()),
    "charge": list_of(pa.float32()),
    "vx": list_of(pa.float32()),
    "vy": list_of(pa.float32()),
    "vz": list_of(pa.float32()),
    "time": list_of(pa.float32()),
    "px": list_of(pa.float32()),
    "py": list_of(pa.float32()),
    "pz": list_of(pa.float32()),
    "perigee_d0": list_of(pa.float32()),
    "perigee_z0": list_of(pa.float32()),
    "num_tracker_hits": list_of(pa.uint16()),
    "num_calo_hits": list_of(pa.uint16()),
    "primary": list_of(pa.bool_()),
    # Vertex index (points to a small set of vertices)
    "vertex_primary": list_of(pa.uint16()),
    # Parent particle identifier (can be null)
    "parent_id": list_of(pa.int64()),
}


# Tracker hits (digitized hits): one row per event, list-valued columns
DIGIHITS_PARQUET_TYPES = {
    "event_id": pa.uint32(),
    "x": list_of(pa.float32()),
    "y": list_of(pa.float32()),
    "z": list_of(pa.float32()),
    "time": list_of(pa.float32()),
    "particle_id": list_of(pa.uint64()),
    "true_x": list_of(pa.float32()),
    "true_y": list_of(pa.float32()),
    "true_z": list_of(pa.float32()),
    "volume_id": list_of(pa.uint8()),
    "layer_id": list_of(pa.uint16()),
    "surface_id": list_of(pa.uint32()),
    "detector": list_of(pa.uint8()),
}


# Tracks: one row per event, list-valued track properties
TRACKS_PARQUET_TYPES = {
    "event_id": pa.uint32(),
    "majority_particle_id": list_of(pa.uint64()),
    "d0": list_of(pa.float32()),
    "z0": list_of(pa.float32()),
    "phi": list_of(pa.float32()),
    "theta": list_of(pa.float32()),
    "qop": list_of(pa.float32()),
    "hit_ids": nested_list_of(pa.uint32()),
    "track_id": list_of(pa.uint16()),
    # Genuine merged-cluster count per track (= ACTS nMeasurements), matches the
    # native Arrow track writer's num_measurements field.
    "num_measurements": list_of(pa.uint32()),
}


# Calorimeter hits: one row per event, list-valued cell properties
# Pandora particle-flow objects: one row per event, list-valued per PFO.
# cluster_ids index CALO_CLUSTERS rows; track_ids index the tracks-table track_id
# (= ActsTracks collection order).
PFOS_PARQUET_TYPES = {
    "event_id": pa.uint32(),
    "pfo_id": list_of(pa.uint32()),
    "pdg": list_of(pa.int32()),
    "charge": list_of(pa.int8()),
    "energy": list_of(pa.float32()),
    "px": list_of(pa.float32()),
    "py": list_of(pa.float32()),
    "pz": list_of(pa.float32()),
    "goodness_of_pid": list_of(pa.float32()),
    "cluster_ids": nested_list_of(pa.uint32()),
    "track_ids": nested_list_of(pa.uint32()),
}

# Pandora calorimeter clusters: one row per event, list-valued per cluster.
# cell_ids join CALO_CELLS.cell_id within the same event.
CALO_CLUSTERS_PARQUET_TYPES = {
    "event_id": pa.uint32(),
    "cluster_id": list_of(pa.uint32()),
    "energy": list_of(pa.float32()),
    "x": list_of(pa.float32()),
    "y": list_of(pa.float32()),
    "z": list_of(pa.float32()),
    "cell_ids": nested_list_of(pa.uint64()),
}


# Release-2 tracker RECO table: one row per event, one list entry per
# MEASUREMENT (the entry's position is the measurement id referenced by the
# tracks table's hit_ids). Local parameters are always filled; the subspace
# bitmask (bit0=loc0, bit1=loc1, bit2=time) says which were measured. Truth is
# LINKED, not embedded: particle_ids are particle-table row indices,
# simhit_ids are TRACKER_SIMHITS row indices of the same event.
TRACKER_HITS_PARQUET_TYPES = {
    "event_id": pa.uint32(),
    "loc0": list_of(pa.float32()),
    "loc1": list_of(pa.float32()),
    "var_loc0": list_of(pa.float32()),
    "var_loc1": list_of(pa.float32()),
    "time": list_of(pa.float32()),
    "var_time": list_of(pa.float32()),
    "subspace": list_of(pa.uint8()),
    "x": list_of(pa.float32()),
    "y": list_of(pa.float32()),
    "z": list_of(pa.float32()),
    "detector": list_of(pa.uint8()),
    "volume_id": list_of(pa.uint8()),
    "layer_id": list_of(pa.uint16()),
    "surface_id": list_of(pa.uint32()),
    "size_loc0": list_of(pa.uint16()),
    "size_loc1": list_of(pa.uint16()),
    "n_channels": list_of(pa.uint16()),
    "sum_activation": list_of(pa.float32()),
    "local_eta": list_of(pa.float32()),
    "local_phi": list_of(pa.float32()),
    "global_eta": list_of(pa.float32()),
    "global_phi": list_of(pa.float32()),
    "eta_angle": list_of(pa.float32()),
    "phi_angle": list_of(pa.float32()),
    "particle_ids": nested_list_of(pa.uint64()),
    "simhit_ids": nested_list_of(pa.uint32()),
}

# Release-2 tracker TRUTH table: one row per event, one list entry per sim-hit
# (ALL sim-hits, container order = the simhit_ids referenced above).
# Standalone-complete for re-digitization: position+time, 4-momentum at the
# hit, deposited energy, particle link, trajectory hit index, sensor ids.
TRACKER_SIMHITS_PARQUET_TYPES = {
    "event_id": pa.uint32(),
    "true_x": list_of(pa.float32()),
    "true_y": list_of(pa.float32()),
    "true_z": list_of(pa.float32()),
    "true_time": list_of(pa.float32()),
    "tpx": list_of(pa.float32()),
    "tpy": list_of(pa.float32()),
    "tpz": list_of(pa.float32()),
    "tE": list_of(pa.float32()),
    "dE": list_of(pa.float32()),
    "particle_id": list_of(pa.uint64()),
    "hit_index": list_of(pa.uint16()),
    "detector": list_of(pa.uint8()),
    "volume_id": list_of(pa.uint8()),
    "layer_id": list_of(pa.uint16()),
    "surface_id": list_of(pa.uint32()),
}


# Digitised calorimeter cells (DDCaloDigi output, the cells particle flow consumed):
# one row per event, list-valued columns per cell; truth via digi->sim links
# (contrib particle ids are MCParticle row indices, same convention as particles table).
CALO_CELLS_PARQUET_TYPES = {
    "event_id": pa.uint32(),
    "detector": list_of(pa.uint8()),
    "cell_id": list_of(pa.uint64()),
    "energy": list_of(pa.float32()),
    "x": list_of(pa.float32()),
    "y": list_of(pa.float32()),
    "z": list_of(pa.float32()),
    "time": list_of(pa.float32()),
    "contrib_particle_ids": nested_list_of(pa.uint64()),
    "contrib_energies": nested_list_of(pa.float32()),
}


CALOHITS_PARQUET_TYPES = {
    "event_id": pa.uint32(),
    "detector": list_of(pa.uint8()),
    "total_energy": list_of(pa.float32()),
    "x": list_of(pa.float32()),
    "y": list_of(pa.float32()),
    "z": list_of(pa.float32()),
    "contrib_particle_ids": nested_list_of(pa.uint64()),
    "contrib_energies": nested_list_of(pa.float32()),
    "contrib_times": nested_list_of(pa.float32()),
}



