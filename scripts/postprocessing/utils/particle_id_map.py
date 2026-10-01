"""Map ACTS-native particle ids onto the edm4hep MCParticle index.

The ACTS Arrow writers number particles 0..N-1 within each event (the position in
ACTS' barcode-sorted SimParticleContainer). Every other ColliderML table - the
Release 1 particles/tracker_hits/tracks from convert_all.py and the calorimeter
tables from reco_tables - uses the edm4hep MCParticle index, so the native ids
must be translated before packaging or the tables do not join.

The ACTS build in the sw image has no particle -> MCParticle map (the fork's
``outputMCParticleMap`` was dropped in the tracker-hits-v2 rebase), so the map is
rebuilt here by matching each native particle to the MCParticle with the same
(PDG, momentum, vertex), compared as float32 - the precision edm4hep stores them
in. Measured on hard_scatter/ttbar/v1 run 0, events 0-15: 99.99% of native
particles match exactly one MCParticle and none are unmatched; the remainder
match several MCParticles with identical kinematics and vertex (indistinguishable
copies), which are paired in index order.

Native event_id is the ddsim file position, so native event e is read from entry e
of the ddsim edm4hep file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pyarrow as pa

logger = logging.getLogger(__name__)

# Columns holding particle ids, per native table. tracker_hits nests one level
# deeper (several particles per measurement).
ID_COLUMNS: dict[str, tuple[str, ...]] = {
    # parent_id: native parents map too; primaries keep -1 (Release 1 instead
    # records the generator-level parent, which ACTS does not carry).
    "particles": ("particle_id", "parent_id"),
    "tracker_hits": ("particle_ids",),
    "tracker_simhits": ("particle_id",),
    "tracks": ("majority_particle_id",),
    "truth_tracks": ("majority_particle_id",),
}
# ACTS writes this for "no majority particle"; it is passed through unchanged.
NO_PARTICLE = np.iinfo(np.uint64).max

_KEY_FIELDS = ("pdg_id", "px", "py", "pz", "vx", "vy", "vz")
_MC_BRANCHES = {
    "pdg_id": "MCParticles/MCParticles.PDG",
    "px": "MCParticles/MCParticles.momentum.x",
    "py": "MCParticles/MCParticles.momentum.y",
    "pz": "MCParticles/MCParticles.momentum.z",
    "vx": "MCParticles/MCParticles.vertex.x",
    "vy": "MCParticles/MCParticles.vertex.y",
    "vz": "MCParticles/MCParticles.vertex.z",
}


def _key_matrix(cols: dict[str, np.ndarray]) -> np.ndarray:
    """(n, 7) float32 matrix; PDG ids are exact in float32 up to 2^24, beyond
    that they are compared via their low bits through the view below."""
    out = np.empty((len(cols["pdg_id"]), len(_KEY_FIELDS)), dtype=np.float32)
    out[:, 0] = np.asarray(cols["pdg_id"], dtype=np.int32).view(np.float32)
    for j, f in enumerate(_KEY_FIELDS[1:], start=1):
        out[:, j] = np.asarray(cols[f], dtype=np.float32)
    return out


def match_event(native: dict[str, np.ndarray], mc: dict[str, np.ndarray]) -> tuple[np.ndarray, int]:
    """Return (mc_index for each native particle, number of ambiguous matches).

    Raises if any native particle has no MCParticle with identical key."""
    a = _key_matrix(native)
    b = _key_matrix(mc)
    both = np.ascontiguousarray(np.vstack([a, b]))
    rows = both.view(np.dtype((np.void, both.dtype.itemsize * both.shape[1]))).ravel()
    _, code = np.unique(rows, return_inverse=True)
    code_a, code_b = code[: len(a)], code[len(a):]

    # Pair the k-th native particle with key c to the k-th MCParticle with key c
    # (both in index order), so duplicates are assigned deterministically.
    def rank_within(codes):
        order = np.lexsort((np.arange(len(codes)), codes))
        sc = codes[order]
        start = np.r_[0, np.flatnonzero(np.diff(sc)) + 1]
        rank_sorted = np.arange(len(sc)) - np.repeat(start, np.diff(np.r_[start, len(sc)]))
        rank = np.empty_like(rank_sorted)
        rank[order] = rank_sorted
        return rank

    ra, rb = rank_within(code_a), rank_within(code_b)
    stride = max(len(a), len(b)) + 1
    kb = code_b.astype(np.int64) * stride + rb
    ka = code_a.astype(np.int64) * stride + ra
    order = np.argsort(kb)
    pos = np.clip(np.searchsorted(kb[order], ka), 0, len(kb) - 1)
    hit = kb[order][pos] == ka
    if not hit.all():
        raise RuntimeError(f"{int((~hit).sum())} of {len(a)} native particles have no matching MCParticle")
    n_b_per_code = np.bincount(code_b, minlength=code.max() + 1)
    ambiguous = int((n_b_per_code[code_a] > 1).sum())
    return order[pos].astype(np.uint64), ambiguous


class RunParticleMap:
    """Per-run native-id -> MCParticle-index lookup, built from the run's native
    ``particles`` table and its ddsim edm4hep file."""

    def __init__(self, particles: pa.Table, sim_file: Path):
        import uproot  # available in the stage container via setup_container_env.sh

        ev = particles.column("event_id").to_numpy().astype(np.int64)
        n_entries = int(ev.max()) + 1 if len(ev) else 0
        mc = uproot.open(sim_file)["events"].arrays(list(_MC_BRANCHES.values()), entry_stop=n_entries, library="np")
        self.maps: dict[int, np.ndarray] = {}
        ambiguous = total = 0
        cols = {f: particles.column(f) for f in ("particle_id",) + _KEY_FIELDS}
        for row, e in enumerate(ev):
            nat = {f: np.asarray(cols[f][row].values) for f in _KEY_FIELDS}
            pid = np.asarray(cols["particle_id"][row].values, dtype=np.int64)
            if len(pid) and not np.array_equal(pid, np.arange(len(pid))):
                # The map is indexed by native id; it must be the dense 0..N-1 index.
                raise RuntimeError(f"event {e}: native particle_id is not 0..N-1")
            mce = {f: mc[b][e] for f, b in _MC_BRANCHES.items()}
            self.maps[int(e)], amb = match_event(nat, mce)
            ambiguous += amb
            total += len(pid)
        # Flatten into one lookup: global index = start[event] + native id.
        n_ev = max(self.maps) + 1 if self.maps else 0
        self._count = np.zeros(n_ev, dtype=np.int64)
        for e, m in self.maps.items():
            self._count[e] = len(m)
        self._start = np.r_[0, np.cumsum(self._count)[:-1]].astype(np.int64) if n_ev else np.zeros(0, np.int64)
        self._flat = np.zeros(int(self._count.sum()), dtype=np.uint64)
        for e, m in self.maps.items():
            self._flat[self._start[e]:self._start[e] + len(m)] = m
        logger.info("%s: mapped %d native particles in %d events onto MCParticle index (%d ambiguous, paired in order)",
                    sim_file.parent.name, total, len(self.maps), ambiguous)

    def remap_table(self, table: pa.Table, obj: str) -> pa.Table:
        for col in ID_COLUMNS.get(obj, ()):
            if col not in table.column_names:
                continue
            arr = table.column(col).combine_chunks()
            ev = table.column("event_id").to_numpy().astype(np.int64)
            table = table.set_column(table.schema.get_field_index(col), col, self._remap_column(arr, ev))
        return table

    def _remap_column(self, arr: pa.ListArray, ev: np.ndarray) -> pa.Array:
        # Peel list levels down to the uint64 ids, remembering each level's
        # offsets, and the event of every leaf.
        levels = []
        leaf_event = ev
        cur = arr
        while pa.types.is_list(cur.type):
            off = cur.offsets.to_numpy()
            levels.append(pa.array(off - off[0], type=pa.int32()))
            leaf_event = np.repeat(leaf_event, np.diff(off))
            cur = cur.flatten()
        leaf_type = cur.type
        raw = cur.to_numpy(zero_copy_only=False)
        if pa.types.is_signed_integer(leaf_type):
            valid = raw >= 0
        else:
            valid = raw != NO_PARTICLE
        ids = np.where(valid, raw, 0).astype(np.uint64)
        out = raw.astype(np.uint64) if not pa.types.is_signed_integer(leaf_type) else raw.astype(np.int64)
        le = leaf_event[valid]
        if len(le):
            if le.max() >= len(self._count):
                raise RuntimeError(f"no particle map for event {int(le.max())}")
            bad = ids[valid] >= self._count[le].astype(np.uint64)
            if bad.any():
                raise RuntimeError(f"{int(bad.sum())} native particle ids out of range for their event "
                                   f"(first event {int(le[bad][0])})")
            out[valid] = self._flat[self._start[le] + ids[valid].astype(np.int64)].astype(out.dtype)
        rebuilt: pa.Array = pa.array(out, type=leaf_type)
        for off in reversed(levels):
            rebuilt = pa.ListArray.from_arrays(off, rebuilt)
        return rebuilt
