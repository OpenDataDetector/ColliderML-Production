"""utils/vertex_links: assignment against brute force, the documented tie rule, link CSV reader."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "postprocessing"))
from utils import vertex_links as vl  # noqa: E402


def brute(cost):
    n, m = cost.shape
    best = None
    if n <= m:
        for perm in itertools.permutations(range(m), n):
            c = sum(cost[i, perm[i]] for i in range(n))
            best = c if best is None or c < best else best
    else:
        for perm in itertools.permutations(range(n), m):
            c = sum(cost[perm[j], j] for j in range(m))
            best = c if best is None or c < best else best
    return best


@pytest.mark.parametrize("shape", [(3, 3), (2, 4), (4, 2), (5, 5), (1, 3)])
def test_hungarian_matches_bruteforce(shape):
    rng = np.random.default_rng(sum(shape))
    for _ in range(20):
        cost = rng.random(shape)
        pairs = vl.hungarian(cost)
        assert len(pairs) == min(shape)
        assert len({a for a, _ in pairs}) == len(pairs) == len({b for _, b in pairs})
        assert abs(sum(cost[a, b] for a, b in pairs) - brute(cost)) < 1e-12


def T(i, z, rec=True):
    return {"id": i, "x": 0.0, "y": 0.0, "z": z, "reconstructable": rec}


def R(z, s):
    return {"x": 0.0, "y": 0.0, "z": z, "sum_pt2": s}


def test_assign_rule():
    assert vl.assign([T(1, 0.0), T(2, 5.0)], [R(0.002, 10), R(5.001, 1)]) == {0: 1, 1: 2}
    assert vl.assign([T(1, 0.0)], [R(0.5, 10)]) == {}, "outside the fixed 0.1 mm gate"
    assert vl.assign([T(1, 0.0), T(2, 0.04, rec=False)], [R(0.0, 10), R(0.04, 1)]) == {0: 1}
    # two coincident interactions and vertices: highest sum_pt2 goes to the HS in either input order
    assert vl.assign([T(1, 0.0), T(2, 0.0)], [R(0.0, 5), R(0.0, 1)]) == {0: 1, 1: 2}
    assert vl.assign([T(1, 0.0), T(2, 0.0)], [R(0.0, 1), R(0.0, 5)]) == {1: 1, 0: 2}
    # maximum cardinality first: the chain 0-a-1 must use both edges
    assert vl.assign([T(1, 0.0), T(2, 0.09)], [R(0.045, 1), R(0.135, 1)]) == {0: 1, 1: 2}


def test_read_track_links(tmp_path):
    p = tmp_path / "vertex_track_links.csv"
    p.write_text("event,vertex,track_id,weight,vx,vy,vz\n0,0,3,0.9,0,0,1.5\n0,0,1,0.2,0,0,1.5\n0,1,-1,0,0,0,9\n")
    L = vl.read_track_links(p)
    assert L[(0, 0)] == ([3, 1], [0.9, 0.2], 1.5) and L[(0, 1)] == ([], [], 9.0)
    p.write_text("event,vertex,track_id,weight,vx,vy,vz\n0,0,3,0.9,0,0,1.5\n0,0,3,0.2,0,0,1.5\n")
    with pytest.raises(RuntimeError, match="twice"):
        vl.read_track_links(p)
    p.write_text("event,vertex,track,weight,vx,vy,vz\n")
    with pytest.raises(RuntimeError, match="header"):
        vl.read_track_links(p)
