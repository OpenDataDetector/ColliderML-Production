"""particles_full generator_particles option: interaction numbering, row selection, and the top -> W/b ->
final state walk across the generator table and the Release 1 rows."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "postprocessing"))
pytest.importorskip("scipy")
cpf = pytest.importorskip("convert_particles_full")

# Two pp interactions in one merged event, generator record first (as in the merged HepMC):
#  hard scatter: 0,1 beams (4) -> 2 top (22) -> 3 top (62) -> 4 W (22), 5 b (23)
#                4 -> 6 mu (1), 7 nu (1);  5 -> 8 B hadron (2) -> 9 pi (1)
#  pileup:       10,11 beams (4) -> 12 gluon (31) -> 13 pi (1)
#  Geant4:       14 electron (0) from 9
PDG = [2212, 2212, 6, 6, 24, 5, 13, -14, 511, 211, 2212, 2212, 21, 211, 11]
STATUS = [4, 4, 22, 62, 22, 23, 1, 1, 2, 1, 4, 4, 31, 1, 0]
PARENTS = [[], [], [0, 1], [2], [3], [3], [4], [4], [5], [8], [], [], [10, 11], [12], [9]]


def daughters():
    d = [[] for _ in PARENTS]
    for i, ps in enumerate(PARENTS):
        for p in ps:
            d[p].append(i)
    return d


def test_interactions_numbered_from_particle_zero():
    inter = cpf.interactions(PARENTS, len(PARENTS))
    assert inter.tolist() == [0] * 10 + [1] * 4 + [0]


def test_generator_rows_scope():
    inter = cpf.interactions(PARENTS, len(PARENTS))
    status = np.asarray(STATUS)
    assert cpf.generator_rows(status, inter, "all").tolist() == [0, 1, 2, 3, 4, 5, 10, 11, 12]
    assert cpf.generator_rows(status, inter, "hard_scatter").tolist() == [0, 1, 2, 3, 4, 5]


def test_walk_top_to_particles_rows():
    """From the last top copy in the generator table, daughters lead to W and b, and their final-state
    descendants are rows of the Release 1 particles table (status 0/1/2)."""
    inter = cpf.interactions(PARENTS, len(PARENTS))
    gen_ids = set(cpf.generator_rows(np.asarray(STATUS), inter, "hard_scatter").tolist())
    particles_ids = {i for i, s in enumerate(STATUS) if s in cpf.RELEASE1_STATUS}
    assert gen_ids.isdisjoint(particles_ids) and gen_ids | particles_ids | {10, 11, 12} == set(range(len(PDG)))
    dau = daughters()
    top = 3
    assert sorted(abs(PDG[d]) for d in dau[top]) == [5, 24]
    final, stack = set(), [top]
    while stack:
        i = stack.pop()
        if STATUS[i] == 1:
            final.add(i)
        else:
            stack.extend(dau[i])
    assert final == {6, 7, 9} and final <= particles_ids
