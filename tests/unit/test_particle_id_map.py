"""Unit tests for the ACTS-native -> MCParticle-index particle id map."""

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "postprocessing"))
from utils.particle_id_map import NO_PARTICLE, RunParticleMap, match_event  # noqa: E402


def _cols(pdg, p, v):
    p = np.asarray(p, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    return {"pdg_id": np.asarray(pdg), "px": p[:, 0], "py": p[:, 1], "pz": p[:, 2],
            "vx": v[:, 0], "vy": v[:, 1], "vz": v[:, 2]}


def test_match_event_with_duplicates_and_skipped_mc():
    # MC has 5 particles; index 1 is generator-only (not in native), 3 and 4 are identical copies.
    mc = _cols([11, 21, 211, 22, 22], [[1, 0, 0], [9, 9, 9], [0, 2, 0], [0, 0, 3], [0, 0, 3]], [[0, 0, 0]] * 5)
    # native order differs from MC order
    nat = _cols([22, 211, 11, 22], [[0, 0, 3], [0, 2, 0], [1, 0, 0], [0, 0, 3]], [[0, 0, 0]] * 4)
    idx, amb = match_event(nat, mc)
    assert idx.tolist() == [3, 2, 0, 4]
    assert amb == 2


def test_match_event_unmatched_raises():
    mc = _cols([11], [[1, 0, 0]], [[0, 0, 0]])
    nat = _cols([13], [[1, 0, 0]], [[0, 0, 0]])
    with pytest.raises(RuntimeError, match="no unused MCParticle"):
        match_event(nat, mc)
    far = _cols([11], [[1.1, 0, 0]], [[0, 0, 0]])
    with pytest.raises(RuntimeError, match="within tolerance"):
        match_event(far, mc)


def test_match_event_last_bit_fallback():
    mc = _cols([211, 22], [[2, 3, 4], [1, 1, 1]], [[0, 0, 0], [0, 0, 0]])
    p = np.nextafter(np.float32(3), np.float32(4))
    nat = _cols([22, 211], [[1, 1, 1], [2, p, 4]], [[0, 0, 0], [0, 0, 0]])
    idx, _ = match_event(nat, mc)
    assert idx.tolist() == [1, 0]


def test_remap_nested_sentinel_and_parent():
    m = RunParticleMap.__new__(RunParticleMap)
    m.maps = {0: np.array([10, 11, 12], dtype=np.uint64), 1: np.array([20, 21], dtype=np.uint64)}
    m._count = np.array([3, 2])
    m._start = np.array([0, 3])
    m._flat = np.array([10, 11, 12, 20, 21], dtype=np.uint64)

    hits = pa.table({
        "event_id": pa.array([1, 0], type=pa.uint32()),
        "particle_ids": pa.array([[[1], [0, 1]], [[2], [], [0]]], type=pa.list_(pa.list_(pa.uint64()))),
    })
    out = m.remap_table(hits, "tracker_hits")
    assert out.column("particle_ids").to_pylist() == [[[21], [20, 21]], [[12], [], [10]]]

    tracks = pa.table({
        "event_id": pa.array([0], type=pa.uint32()),
        "majority_particle_id": pa.array([[2, int(NO_PARTICLE)]], type=pa.list_(pa.uint64())),
    })
    out = m.remap_table(tracks, "tracks")
    assert out.column("majority_particle_id").to_pylist() == [[12, int(NO_PARTICLE)]]

    parts = pa.table({
        "event_id": pa.array([0], type=pa.uint32()),
        "particle_id": pa.array([[0, 1, 2]], type=pa.list_(pa.uint64())),
        "parent_id": pa.array([[-1, 0, 1]], type=pa.list_(pa.int64())),
    })
    out = m.remap_table(parts, "particles")
    assert out.column("particle_id").to_pylist() == [[10, 11, 12]]
    assert out.column("parent_id").to_pylist() == [[-1, 10, 11]]
    assert out.schema.field("parent_id").type == pa.list_(pa.int64())


def test_remap_out_of_range_raises():
    m = RunParticleMap.__new__(RunParticleMap)
    m._count = np.array([1])
    m._start = np.array([0])
    m._flat = np.array([5], dtype=np.uint64)
    t = pa.table({"event_id": pa.array([0], type=pa.uint32()),
                  "majority_particle_id": pa.array([[3]], type=pa.list_(pa.uint64()))})
    with pytest.raises(RuntimeError, match="out of range"):
        m.remap_table(t, "tracks")
