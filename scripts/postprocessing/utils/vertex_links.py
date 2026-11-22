"""reco/vertices links (v23): vertex -> tracks from the image's VertexTrackLinkWriter CSV, and vertex -> truth
interaction by position. numpy only (the sim image has no scipy).

Truth link rule (documented in the dataset card and PILOT_PLAN_v5):
  interaction            a vertex_primary value >= 1 among primary particles; position = median production point
                         (vx, vy, vz) of its primary particles
  reconstructable        >= 2 charged primary particles of it with pT > 0.7 GeV, |eta| < 3.5 and >= 6 distinct
                         (volume_id, layer_id) pairs among their simulated tracker hits
  assignment             per event, between reco vertices and reconstructable interactions, edges only for
                         |z_reco - z_truth| <= 0.1 mm; lexicographic objective: (1) maximum number of assigned
                         pairs, (2) minimum total |dz| with |dz| quantised at 1e-7 mm, (3) among those optima, reco
                         vertices in rank order (-sum_pt2, z, x, y) each take the lowest-vertex_primary interaction
                         (or none, ranked last) that still allows an optimum; reco vertices with identical rank keys
                         are taken in input order
  truth population       the RELEASED truth/particles (primary flag and vertex_primary as published) and
                         simulated hits with particle ids remapped to the released ids
  track weights          AMVF weight per link; NaN where AMVF returned a non-finite weight (kept, never counted)
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
    vx, vy, vz = (np.asarray(part[k], dtype=np.float64) for k in ("vx", "vy", "vz"))
    pid = np.asarray(part["particle_id"], dtype=np.uint64)
    pt = np.hypot(px, py)
    aeta = np.abs(np.arcsinh(pz / np.maximum(pt, 1e-12)))
    # distinct (volume, layer) per particle id, vectorised
    sp = np.asarray(sim["particle_id"], dtype=np.uint64)
    key = np.asarray(sim["volume_id"], dtype=np.int64) * 1_000_000 + np.asarray(sim["layer_id"], dtype=np.int64)
    if len(sp):
        pairs = np.unique(np.stack([sp, key.astype(np.uint64)], axis=1), axis=0)
        upid, cnt = np.unique(pairs[:, 0], return_counts=True)
        pos = np.searchsorted(upid, pid)
        pos_c = np.minimum(pos, len(upid) - 1)
        nlay = np.where(upid[pos_c] == pid, cnt[pos_c], 0)
    else:
        nlay = np.zeros(len(pid), dtype=np.int64)
    good = prim & (np.abs(np.asarray(part["charge"], dtype=np.float64)) > 0) & (pt > 0.7) & (aeta < 3.5) & (nlay >= 6)
    out = []
    sel = prim & (vp >= 1)
    order = np.argsort(vp[sel], kind="stable")
    v_s, x_s, y_s, z_s, g_s = vp[sel][order], vx[sel][order], vy[sel][order], vz[sel][order], good[sel][order]
    ks, starts = np.unique(v_s, return_index=True)
    ends = np.append(starts[1:], len(v_s))
    for k, a, b in zip(ks, starts, ends):
        out.append({"id": int(k), "x": float(np.median(x_s[a:b])), "y": float(np.median(y_s[a:b])),
                    "z": float(np.median(z_s[a:b])), "reconstructable": int(g_s[a:b].sum()) >= 2})
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
        ok = allowed[np.ix_(rows, cols)]
        dq = np.rint(D[np.ix_(rows, cols)] / DZ_QUANTUM_MM)       # integer distance units
        big = float(dq[ok].sum()) + 1.0                           # one missing pair costs more than any distance

        def solve(mask):
            """(number of pairs, total quantised distance, pairs) of the optimum with only `mask` edges."""
            if big * (max(mask.shape) + 1) > 2.0 ** 52:
                raise RuntimeError("assignment cost exceeds exact float64 integers")
            pr = [(i, j) for i, j in hungarian(np.where(mask, dq, big)) if mask[i, j]]
            return len(pr), float(sum(dq[i, j] for i, j in pr)), pr
        n_opt, d_opt, pairs = solve(ok)
        # final key: in reco rank order, each reco vertex takes the best-ranked interaction (or none, ranked last)
        # that still allows an optimal (pairs, distance) assignment
        mask = ok.copy()
        order_r = sorted(range(len(rows)), key=lambda a: rank_r[rows[a]])
        for a in order_r:
            cands = sorted(np.flatnonzero(mask[a]).tolist(), key=lambda b: rank_t[cols[b]]) + [None]
            for b in cands:
                trial = mask.copy()
                trial[a, :] = False
                if b is not None:
                    trial[:, b] = False
                    trial[a, b] = True
                n_, d_, pr = solve(trial)
                if n_ == n_opt and d_ == d_opt and (b is None or (a, b) in pr):
                    mask, pairs = trial, pr
                    break
        for i, j in pairs:
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
            if not all(np.isfinite(pos)):
                raise RuntimeError(f"{path}: non-finite position for vertex {key}")
            # a non-finite AMVF weight is kept as NaN (observed: 1 of about 88k links in PU200 run 0); it never counts
            # towards n_tracks (weight > 0.1 is false) and the ROOT ntuple does not list it either
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
