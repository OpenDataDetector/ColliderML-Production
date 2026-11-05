"""calo_cells truth: at most one contribution per (cell, particle), energies summed."""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "postprocessing"))

from convert_calo_digi import merge_per_particle  # noqa: E402


def test_merges_repeated_particles_per_cell():
    pid = np.array([5, 3, 5, 7, 2, 2, 9], dtype=np.uint64)
    en = np.array([1.0, 2.0, 3.0, 4.0, 0.5, 0.5, 1.0], dtype=np.float32)
    counts = np.array([4, 0, 2, 1])
    p, e, c = merge_per_particle(pid, en, counts)
    assert p.tolist() == [3, 5, 7, 2, 9]
    assert e.tolist() == [2.0, 4.0, 4.0, 1.0, 1.0]
    assert c.tolist() == [3, 0, 1, 1]


def test_energy_is_conserved_per_cell():
    rng = np.random.default_rng(1)
    counts = rng.integers(0, 30, size=200)
    pid = rng.integers(0, 12, size=counts.sum()).astype(np.uint64)
    en = rng.random(counts.sum()).astype(np.float32)
    p, e, c = merge_per_particle(pid, en, counts)
    cell_in = np.repeat(np.arange(len(counts)), counts)
    cell_out = np.repeat(np.arange(len(counts)), c)
    np.testing.assert_allclose(np.bincount(cell_out, e, len(counts)), np.bincount(cell_in, en, len(counts)), rtol=1e-5)
    for i in range(len(counts)):
        ids = p[cell_out == i]
        assert len(ids) == len(set(ids.tolist())) == len(set(pid[cell_in == i].tolist()))


def test_empty_input():
    p, e, c = merge_per_particle(np.empty(0, np.uint64), np.empty(0, np.float32), np.array([0, 0]))
    assert len(p) == len(e) == 0 and c.tolist() == [0, 0]
