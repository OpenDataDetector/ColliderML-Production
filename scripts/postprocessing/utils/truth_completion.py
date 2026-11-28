"""D8 (2026-11-28): complete the Release 1 PU200 truth/particles table.

Release 1 built truth/particles by an inner join of the edm4hep MCParticles with the ACTS particles on the
float32-cast (vx, vy, vz, px, py, pz). A component that sits on a float32 rounding tie rounds differently on the
two sides, and the particle was dropped (about 0.2% of PU200 events), leaving calorimeter contributions and
tracker simhits that point to an id without a truth row.

The completed table keeps every Release 1 row unchanged (values, order, column types) and adds one row for each
MCParticle index that the run's native-particle map (utils/particle_id_map.RunParticleMap: native ACTS particles
of the v23 digitisation -> MCParticle index) produces and the Release 1 rows lack for that event. Added rows use
the Release 1 column formulas:

  particle_id, pdg_id, mass, energy, charge, vx, vy, vz, time, px, py, pz, parent_id, primary  <- edm4hep MCParticle
  vertex_primary                                                                              <- ACTS native particle
  perigee_d0, perigee_z0                       <- ACTS particles.root (same container order as the native table;
                                                  the v23 native writer leaves them NaN)

Equality is bit equality except that any NaN equals any NaN (Release 1 itself holds two NaN payloads).
(energy = sqrt(px^2 + py^2 + pz^2 + mass^2) in double, as pyedm4hep; parent_id = the first parent's index, -1 for
none; primary = not created in simulation, simulatorStatus bit 30). Each formula is checked against the existing
rows by ``column_agreement`` (testbed/v23_plan/r4/d8/).
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

# Release 1 full_pileup/ttbar/v1 truth/particles schema (column order and types)
V1_SCHEMA = pa.schema([
    ("event_id", pa.uint32()),
    ("particle_id", pa.list_(pa.uint64())),
    ("pdg_id", pa.list_(pa.int64())),
    ("mass", pa.list_(pa.float32())),
    ("energy", pa.list_(pa.float32())),
    ("charge", pa.list_(pa.float32())),
    ("vx", pa.list_(pa.float32())),
    ("vy", pa.list_(pa.float32())),
    ("vz", pa.list_(pa.float32())),
    ("time", pa.list_(pa.float32())),
    ("px", pa.list_(pa.float32())),
    ("py", pa.list_(pa.float32())),
    ("pz", pa.list_(pa.float32())),
    ("perigee_d0", pa.list_(pa.float32())),
    ("perigee_z0", pa.list_(pa.float32())),
    ("vertex_primary", pa.list_(pa.uint16())),
    ("parent_id", pa.list_(pa.int64())),
    ("primary", pa.list_(pa.bool_())),
])
LIST_COLS = [f.name for f in V1_SCHEMA][1:]
NP_TYPES = {f.name: f.type.value_type.to_pandas_dtype() for f in V1_SCHEMA if pa.types.is_list(f.type)}
ACTS_COLS = ("perigee_d0", "perigee_z0", "vertex_primary")
MC_COLS = tuple(c for c in LIST_COLS if c not in ACTS_COLS)

MC_BRANCHES = {
    "PDG": "MCParticles/MCParticles.PDG",
    "simulatorStatus": "MCParticles/MCParticles.simulatorStatus",
    "charge": "MCParticles/MCParticles.charge",
    "time": "MCParticles/MCParticles.time",
    "mass": "MCParticles/MCParticles.mass",
    "vx": "MCParticles/MCParticles.vertex.x",
    "vy": "MCParticles/MCParticles.vertex.y",
    "vz": "MCParticles/MCParticles.vertex.z",
    "px": "MCParticles/MCParticles.momentum.x",
    "py": "MCParticles/MCParticles.momentum.y",
    "pz": "MCParticles/MCParticles.momentum.z",
    "parents_begin": "MCParticles/MCParticles.parents_begin",
    "parents_end": "MCParticles/MCParticles.parents_end",
    "parents": "_MCParticles_parents/_MCParticles_parents.index",
}
KEY = ("vx", "vy", "vz", "px", "py", "pz")


def read_mc(sim_file, entry_start: int, entry_stop: int) -> dict:
    """{name: object array of per-entry numpy arrays} for entries [start, stop) of the edm4hep file."""
    import uproot
    t = uproot.open(sim_file)["events"]
    arr = t.arrays(list(MC_BRANCHES.values()), entry_start=entry_start, entry_stop=entry_stop, library="np")
    return {k: arr[b] for k, b in MC_BRANCHES.items()}


def mc_columns(mc: dict, i: int, ids: np.ndarray) -> dict:
    """Release 1 column values of the MCParticles ``ids`` of entry i (index into the read_mc arrays)."""
    ids = np.asarray(ids, dtype=np.int64)
    g = {k: v[i] for k, v in mc.items()}
    px, py, pz, m = (np.asarray(g[k], dtype=np.float64)[ids] for k in ("px", "py", "pz", "mass"))
    pb = np.asarray(g["parents_begin"], dtype=np.int64)[ids]
    pe = np.asarray(g["parents_end"], dtype=np.int64)[ids]
    parents = np.asarray(g["parents"], dtype=np.int64)
    has = pe > pb
    parent = np.full(len(ids), -1, dtype=np.int64)
    parent[has] = parents[pb[has]]
    out = {
        "particle_id": ids.astype(np.uint64),
        "pdg_id": np.asarray(g["PDG"], dtype=np.int64)[ids],
        "mass": m.astype(np.float32),
        "energy": np.sqrt(px ** 2 + py ** 2 + pz ** 2 + m ** 2).astype(np.float32),
        "charge": np.asarray(g["charge"], dtype=np.float32)[ids],
        "time": np.asarray(g["time"], dtype=np.float32)[ids],
        "parent_id": parent,
        "primary": ~((np.asarray(g["simulatorStatus"], dtype=np.int64)[ids] >> 30) & 1).astype(bool),
    }
    for k in KEY:
        out[k] = np.asarray(g[k], dtype=np.float64)[ids].astype(np.float32)
    return out


def list_columns(table: pa.Table, cols) -> tuple[np.ndarray, dict]:
    """(row offsets, {col: flat numpy}) of list columns of a table with one row per event."""
    out, off = {}, None
    for c in cols:
        a = table.column(c).combine_chunks()
        o = a.offsets.to_numpy()
        flat = a.flatten().to_numpy(zero_copy_only=False)
        if off is None:
            off = o - o[0]
        out[c] = flat
    return off, out


def read_acts_root(root_file, local_events) -> dict:
    """{run-local event: {vx..pz, perigee_d0, perigee_z0}} from the ACTS particles.root of the run."""
    import uproot
    t = uproot.open(root_file)["particles"]
    ev = t["event_id"].array(library="np")
    out = {}
    cols = list(KEY) + ["perigee_d0", "perigee_z0"]
    for le in local_events:
        j = np.flatnonzero(ev == le)
        if len(j) != 1:
            raise RuntimeError(f"{root_file}: {len(j)} entries for event {le}")
        j = int(j[0])
        a = t.arrays(cols, entry_start=j, entry_stop=j + 1, library="np")
        out[int(le)] = {c: np.asarray(a[c][0]) for c in cols}
    return out


def _acts_aligned(native_cols: dict, ns: int, ne: int, root_ev: dict, le: int) -> None:
    for c in KEY:
        if not np.array_equal(np.asarray(native_cols[c][ns:ne], np.float32), np.asarray(root_ev[c], np.float32)):
            raise RuntimeError(f"event {le}: particles.root and the native particles table differ in {c} "
                               "(not the same container order)")


def same_values(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise bit equality, any NaN equal to any NaN."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0:
        return np.zeros(0, bool)
    same = (a.view(np.uint8).reshape(len(a), -1) == b.view(np.uint8).reshape(len(b), -1)).all(1)
    if a.dtype.kind == "f":
        same |= np.isnan(a) & np.isnan(b)
    return same


def key_relation(mc_key: np.ndarray, acts_key: np.ndarray) -> str:
    """'exact' if all six float32 components agree, 'ulp<=2' if within 2 float32 ulp each, else 'other'."""
    def ordn(x):
        i = np.ascontiguousarray(x, dtype=np.float32).view(np.int32).astype(np.int64)
        return np.where(i < 0, -(i & 0x7FFFFFFF), i)
    if np.array_equal(np.asarray(mc_key, np.float32), np.asarray(acts_key, np.float32)):
        return "exact"
    return "ulp<=2" if (np.abs(ordn(mc_key) - ordn(acts_key)) <= 2).all() else "other"


def complete_window(base: pa.Table, native: pa.Table, mc: dict, mc_entry0: int, acts_root: dict, run_offset: int,
                    local_start: int, local_stop: int):
    """Completed truth rows for the run-local events [local_start, local_stop) of one run.

    base: Release 1 rows (global event_id) of at least those events; native: the run's native particles table
    with particle_id/parent_id already remapped to the MCParticle index (run-local event_id); mc: read_mc of the
    run's edm4hep file starting at entry mc_entry0; acts_root: read_acts_root of the run. Returns (table in V1_SCHEMA with global event ids,
    {global event: [{"particle_id", "cause"}...]})."""
    bev = base.column("event_id").to_numpy().astype(np.int64)
    nev = native.column("event_id").to_numpy().astype(np.int64)
    b_off, B = list_columns(base, LIST_COLS)
    n_off, N = list_columns(native, ("particle_id",) + ACTS_COLS + KEY)
    b_row = {int(e): i for i, e in enumerate(bev)}
    n_row = {int(e): i for i, e in enumerate(nev)}
    rows = {c: [] for c in LIST_COLS}
    events, added = [], {}
    for le in range(local_start, local_stop):
        ge = le + run_offset
        if ge not in b_row:
            raise RuntimeError(f"event {ge}: no Release 1 truth row")
        if le not in n_row:
            raise RuntimeError(f"run-local event {le}: no native particles row")
        bi, ni = b_row[ge], n_row[le]
        bs, be_ = b_off[bi], b_off[bi + 1]
        ns, ne = n_off[ni], n_off[ni + 1]
        bids = B["particle_id"][bs:be_].astype(np.int64)
        nids = N["particle_id"][ns:ne].astype(np.int64)
        new = np.setdiff1d(np.unique(nids), bids)
        ev_cols = {c: B[c][bs:be_] for c in LIST_COLS}
        if len(new):
            mcc = mc_columns(mc, le - mc_entry0, new)
            pos = {int(p): k for k, p in enumerate(nids)}
            sel = np.array([pos[int(p)] for p in new]) + ns
            _acts_aligned(N, ns, ne, acts_root[le], le)
            mcc["vertex_primary"] = N["vertex_primary"][sel].astype(NP_TYPES["vertex_primary"])
            for c in ("perigee_d0", "perigee_z0"):
                mcc[c] = np.asarray(acts_root[le][c])[sel - ns].astype(NP_TYPES[c])
            causes = []
            for k, p in enumerate(new):
                mk = np.array([mcc[c][k] for c in KEY], np.float32)
                ak = np.array([N[c][sel[k]] for c in KEY], np.float32)
                causes.append({"particle_id": int(p), "pdg_id": int(mcc["pdg_id"][k]),
                               "cause": "float32_tie" if key_relation(mk, ak) == "ulp<=2" else
                               ("acts_build_difference" if key_relation(mk, ak) == "exact" else "tolerance_match")})
            added[ge] = causes
            order = np.argsort(np.r_[bids, new], kind="stable")
            ev_cols = {c: np.r_[ev_cols[c].astype(NP_TYPES[c]), mcc[c].astype(NP_TYPES[c])][order]
                       for c in LIST_COLS}
        for c in LIST_COLS:
            rows[c].append(np.asarray(ev_cols[c], dtype=NP_TYPES[c]))
        events.append(ge)
    arrays = [pa.array(np.asarray(events, dtype=np.uint32), type=pa.uint32())]
    for f in list(V1_SCHEMA)[1:]:
        parts = rows[f.name]
        off = np.r_[0, np.cumsum([len(p) for p in parts])].astype(np.int32)
        flat = np.concatenate(parts) if parts else np.zeros(0, NP_TYPES[f.name])
        arrays.append(pa.ListArray.from_arrays(pa.array(off, pa.int32()), pa.array(flat, type=f.type.value_type)))
    return pa.Table.from_arrays(arrays, schema=V1_SCHEMA), added


def column_agreement(base: pa.Table, native: pa.Table, mc: dict, mc_entry0: int, acts_root: dict, run_offset: int,
                     local_start: int, local_stop: int) -> dict:
    """Apply the added-row formulas to every EXISTING Release 1 row of the window that the native map covers;
    {column: [rows compared, rows differing]} (bit equality; NaN equal to NaN), plus the counts of Release 1
    rows outside the map and the map's added rows."""
    bev = base.column("event_id").to_numpy().astype(np.int64)
    nev = native.column("event_id").to_numpy().astype(np.int64)
    b_off, B = list_columns(base, LIST_COLS)
    n_off, N = list_columns(native, ("particle_id", "vertex_primary") + KEY)
    b_row = {int(e): i for i, e in enumerate(bev)}
    n_row = {int(e): i for i, e in enumerate(nev)}
    res = {c: [0, 0] for c in LIST_COLS}
    res["_nan_payload_differences"] = {}
    res["_base_rows_not_in_map"] = 0
    res["_base_rows_compared_mc_only"] = 0
    for le in range(local_start, local_stop):
        bi, ni = b_row[le + run_offset], n_row[le]
        bs, be_ = b_off[bi], b_off[bi + 1]
        ns, ne = n_off[ni], n_off[ni + 1]
        bids = B["particle_id"][bs:be_].astype(np.int64)
        nids = N["particle_id"][ns:ne].astype(np.int64)
        mcc = mc_columns(mc, le - mc_entry0, bids)       # MCParticle columns: every Release 1 row
        for c in MC_COLS:
            a = np.asarray(B[c][bs:be_]).astype(NP_TYPES[c]); b = mcc[c].astype(NP_TYPES[c])
            _count(res, c, a, b)
        pos = {int(p): k for k, p in enumerate(nids)}
        inmap = np.array([int(p) in pos for p in bids], dtype=bool)
        res["_base_rows_not_in_map"] += int((~inmap).sum())
        sel = np.array([pos[int(p)] for p in bids[inmap]], dtype=np.int64)
        _acts_aligned(N, ns, ne, acts_root[le], le)
        _count(res, "vertex_primary", np.asarray(B["vertex_primary"][bs:be_])[inmap],
               N["vertex_primary"][sel + ns].astype(NP_TYPES["vertex_primary"]))
        for c in ("perigee_d0", "perigee_z0"):
            _count(res, c, np.asarray(B[c][bs:be_])[inmap].astype(np.float32),
                   np.asarray(acts_root[le][c])[sel].astype(np.float32))
    return res


def _count(res: dict, c: str, a: np.ndarray, b: np.ndarray) -> None:
    same = same_values(a, b)
    res[c][0] += len(a); res[c][1] += int((~same).sum())
    if a.dtype.kind == "f" and len(a):
        nanp = np.isnan(a) & np.isnan(b) & (a.view(np.uint8).reshape(len(a), -1) != b.view(np.uint8).reshape(len(b), -1)).any(1)
        if nanp.any():
            res["_nan_payload_differences"][c] = res["_nan_payload_differences"].get(c, 0) + int(nanp.sum())
