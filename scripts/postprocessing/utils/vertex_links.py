"""reco/vertices links (v23): vertex -> tracks from the image's VertexTrackLinkWriter CSV, and vertex -> truth
interaction by position. numpy only (the sim image has no scipy).

Truth link rule (documented in the dataset card and PILOT_PLAN_v5):
  interaction            a vertex_primary value >= 1 among primary particles; position = median production point
                         (vx, vy, vz) of its primary particles
  reconstructable        >= 2 charged primary particles of it with pT > 0.7 GeV, |eta| < 3.5 and >= 6 distinct
                         (volume_id, layer_id) pairs among their simulated tracker hits
  assignment             per event, between reco vertices and reconstructable interactions, edges only for
                         |z_reco - z_truth| <= 0.1 mm; lexicographic objective: (1) maximum number of assigned
                         pairs, (2) minimum total |dz| with |dz| quantised at 1e-7 mm, (3) minimum sum of squared
                         differences between the reco rank (-sum_pt2, z, x, y) and the interaction rank
                         (vertex_primary); remaining ties (identical keys) broken by input order
  truth population       the RELEASED truth/particles (primary flag and vertex_primary as published) and
                         simulated hits with particle ids remapped to the released ids
  columns                truth_interaction (vertex_primary of the assigned interaction, -1 if none) and
                         truth_interaction_dz (z_reco - z_truth in mm, NaN if none)
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

DZ_MAX_MM = 0.1
DZ_QUANTUM_MM = 1e-7   # z differences below 0.1 nm are a tie (resolved by rank)


def hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-cost assignment for an n x m matrix (n <= m handled by transposing); returns (row, col) pairs
    covering every row of the smaller side. O(n^2 m) shortest augmenting paths with potentials."""
    cost = np.asarray(cost, dtype=np.float64)
    transposed = cost.shape[0] > cost.shape[1]
    if transposed:
        cost = cost.T
    n, m = cost.shape
    if n == 0:
        return []
    INF = float("inf")
    u = np.zeros(n + 1); v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=np.int64); way = np.zeros(m + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF); used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = INF, -1
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta; v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = [(int(p[j]) - 1, j - 1) for j in range(1, m + 1) if p[j] != 0]
    return [(c, r) for r, c in pairs] if transposed else pairs


def interactions(part: dict, sim: dict) -> list[dict]:
    """part: per-event particle lists (particle_id, charge, px, py, pz, vx, vy, vz, vertex_primary, primary);
    sim: simhit lists (particle_id, volume_id, layer_id)."""
    vp = np.asarray(part["vertex_primary"], dtype=np.int64)
    prim = np.asarray(part["primary"], dtype=bool)
    px, py, pz = (np.asarray(part[k], dtype=np.float64) for k in ("px", "py", "pz"))
    pt = np.hypot(px, py)
    aeta = np.abs(np.arcsinh(pz / np.maximum(pt, 1e-12)))
    layers = {}
    for a, vol, lay in zip(sim["particle_id"], sim["volume_id"], sim["layer_id"]):
        layers.setdefault(int(a), set()).add((int(vol), int(lay)))
    nlay = np.array([len(layers.get(int(x), ())) for x in part["particle_id"]])
    good = prim & (np.abs(np.asarray(part["charge"], dtype=np.float64)) > 0) & (pt > 0.7) & (aeta < 3.5) & (nlay >= 6)
    out = []
    for k in np.unique(vp[prim & (vp >= 1)]):
        msk = prim & (vp == k)
        out.append({"id": int(k), "x": float(np.median(np.asarray(part["vx"])[msk])),
                    "y": float(np.median(np.asarray(part["vy"])[msk])), "z": float(np.median(np.asarray(part["vz"])[msk])),
                    "reconstructable": int((good & (vp == k)).sum()) >= 2})
    return out


def assign(truth: list[dict], reco: list[dict], dz_max: float = DZ_MAX_MM) -> dict[int, int]:
    """{reco index: truth vertex_primary} by the documented rule."""
    T = [t for t in truth if t["reconstructable"]]
    nr, nt = len(reco), len(T)
    if nr == 0 or nt == 0:
        return {}
    zr = np.array([v["z"] for v in reco]); zt = np.array([t["z"] for t in T])
    D = np.abs(zr[:, None] - zt[None, :])
    allowed = D <= dz_max
    if not allowed.any():
        return {}
    keys = [(-v["sum_pt2"], v["z"], v["x"], v["y"]) for v in reco]
    rank_r = np.empty(nr); rank_r[sorted(range(nr), key=lambda a: keys[a])] = np.arange(nr)
    rank_t = np.empty(nt); rank_t[sorted(range(nt), key=lambda j: T[j]["id"])] = np.arange(nt)
    # connected components of the allowed edges are independent sub-problems (small in z)
    parent = list(range(nr + nt))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in zip(*np.nonzero(allowed)):
        parent[find(a)] = find(nr + b)
    comps = {}
    for a in range(nr):
        comps.setdefault(find(a), ([], []))[0].append(a)
    for b in range(nt):
        comps.setdefault(find(nr + b), ([], []))[1].append(b)
    out = {}
    for rows, cols in comps.values():
        if not rows or not cols:
            continue
        sub = D[np.ix_(rows, cols)]
        ok = allowed[np.ix_(rows, cols)]
        # lexicographic objective as one integer cost: (1) number of assigned pairs, (2) total |dz| quantised at
        # DZ_QUANTUM_MM (distances closer than that are a tie), (3) sum of squared rank differences
        n = max(len(rows), len(cols))
        tie = (rank_r[rows][:, None] - rank_t[cols][None, :]) ** 2
        tie_scale = float(n) * float(nr + nt) ** 2 + 1.0       # > any total of tie terms
        dq = np.rint(sub / DZ_QUANTUM_MM)                        # integer distance units
        big = (float(n) + 1.0) * (float(dq[ok].max(initial=0.0)) * tie_scale + tie_scale)   # > any allowed total
        if big * (n + 1) > 2.0 ** 52:
            raise RuntimeError("assignment cost exceeds exact float64 integers")
        cost = np.where(ok, dq * tie_scale + tie, big)
        for i, j in hungarian(cost):
            if ok[i, j]:
                out[rows[i]] = T[cols[j]]["id"]
    return out


def read_track_links(path: Path) -> dict[tuple[int, int], tuple[list[int], list[float], tuple[float, float, float]]]:
    """{(event, vertex): (track_ids, weights, (vx, vy, vz))} from vertex_track_links.csv. A vertex without tracks
    has exactly one line with track_id -1; any other negative id, a sentinel mixed with tracks, a repeated track,
    a non-finite value or two positions for one vertex is an error."""
    out, empty = {}, set()
    with open(path) as f:
        r = csv.DictReader(f)
        if r.fieldnames != ["event", "vertex", "track_id", "weight", "vx", "vy", "vz"]:
            raise RuntimeError(f"{path}: unexpected header {r.fieldnames}")
        for row in r:
            key = (int(row["event"]), int(row["vertex"]))
            pos = (float(row["vx"]), float(row["vy"]), float(row["vz"]))
            w = float(row["weight"])
            if not all(np.isfinite(pos)) or not np.isfinite(w):
                raise RuntimeError(f"{path}: non-finite value for vertex {key}")
            ids, ws, p0 = out.setdefault(key, ([], [], pos))
            if p0 != pos:
                raise RuntimeError(f"{path}: vertex {key} has two positions")
            t = int(row["track_id"])
            if t == -1:
                if ids or key in empty:
                    raise RuntimeError(f"{path}: empty-vertex marker for {key} mixed with tracks or repeated")
                empty.add(key)
                continue
            if t < 0 or key in empty:
                raise RuntimeError(f"{path}: invalid track id {t} or tracks after the empty marker for {key}")
            if t in ids:
                raise RuntimeError(f"{path}: vertex {key} lists track {t} twice")
            ids.append(t); ws.append(w)
    return out
