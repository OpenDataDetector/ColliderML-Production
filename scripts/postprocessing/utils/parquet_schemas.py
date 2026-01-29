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
}


# Calorimeter hits: one row per event, list-valued cell properties
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



