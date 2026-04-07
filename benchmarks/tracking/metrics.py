"""Tracking metrics.

Pure-Python implementations suitable for both client-side evaluation (in
the pip package) and server-side re-scoring (in the backend). No external
dependencies beyond numpy/pyarrow.

The primary metric is **TrackML weighted efficiency**, defined as the sum of
hit weights across all correctly reconstructed tracks divided by the total
hit weight. A reconstructed track is "correct" if the majority of its hits
come from a single particle AND that particle contributes at least 50% of
its own hits to the track.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pyarrow as pa


def _group_hits_by_event(table: pa.Table) -> dict[int, dict]:
    """Return {event_id: {"hit_id": ..., "track_id": ..., "weight": ...}}."""
    events: dict[int, dict] = defaultdict(lambda: {"hit_id": [], "track_id": [], "weight": []})
    cols = table.to_pydict()
    evt = cols["event_id"]
    hit = cols["hit_id"]
    trk = cols["track_id"]
    weights = cols.get("weight") or [1.0] * len(evt)
    for e, h, t, w in zip(evt, hit, trk, weights):
        events[e]["hit_id"].append(h)
        events[e]["track_id"].append(t)
        events[e]["weight"].append(w)
    return events


def trackml_weighted_efficiency(
    preds: pa.Table,
    truth_hits: pa.Table,
) -> float:
    """Weighted efficiency à la TrackML.

    preds: columns event_id, hit_id, track_id[, weight]
    truth_hits: columns event_id, hit_id, particle_id, weight (optional)

    Returns a value in [0, 1].
    """
    truth_cols = truth_hits.to_pydict()
    # Build (event_id, hit_id) -> (particle_id, weight)
    truth_map: dict[tuple, tuple] = {}
    for e, h, pid, w in zip(
        truth_cols["event_id"],
        truth_cols["hit_id"],
        truth_cols.get("particle_id") or truth_cols.get("majority_particle_id"),
        truth_cols.get("weight") or [1.0] * len(truth_cols["event_id"]),
    ):
        truth_map[(e, h)] = (pid, float(w))

    pred_cols = preds.to_pydict()
    # Build predicted track -> list of (particle_id, weight)
    tracks: dict[tuple, list] = defaultdict(list)
    total_weight = 0.0
    for e, h, t in zip(pred_cols["event_id"], pred_cols["hit_id"], pred_cols["track_id"]):
        key = (e, t)
        tr = truth_map.get((e, h))
        if tr is None:
            continue
        pid, w = tr
        tracks[key].append((pid, w))
        total_weight += w

    # Particle -> total weight (for the 50% self-contribution rule)
    particle_totals: dict[tuple, float] = defaultdict(float)
    for (e, h), (pid, w) in truth_map.items():
        particle_totals[(e, pid)] += w

    correct_weight = 0.0
    for (e, _track), members in tracks.items():
        if not members:
            continue
        # Majority particle by weight
        by_particle: dict = defaultdict(float)
        for pid, w in members:
            by_particle[pid] += w
        majority_pid, majority_weight = max(by_particle.items(), key=lambda kv: kv[1])
        track_total = sum(w for _, w in members)
        particle_total = particle_totals.get((e, majority_pid), 0)
        if particle_total == 0:
            continue
        # Both criteria: >=50% of track is the majority particle
        # AND >=50% of that particle's hits are in this track
        if (majority_weight / track_total >= 0.5
                and majority_weight / particle_total >= 0.5):
            correct_weight += majority_weight

    if total_weight == 0:
        return 0.0
    return round(correct_weight / total_weight, 6)


def fake_rate(preds: pa.Table, truth_hits: pa.Table) -> float:
    """Fraction of reconstructed tracks with no majority truth particle."""
    events = _group_hits_by_event(preds)
    truth_map = {}
    truth_cols = truth_hits.to_pydict()
    pid_col = truth_cols.get("particle_id") or truth_cols.get("majority_particle_id")
    for e, h, pid in zip(truth_cols["event_id"], truth_cols["hit_id"], pid_col):
        truth_map[(e, h)] = pid

    fakes = 0
    total = 0
    for e, d in events.items():
        by_track: dict = defaultdict(list)
        for hit_id, trk_id in zip(d["hit_id"], d["track_id"]):
            pid = truth_map.get((e, hit_id))
            if pid is not None:
                by_track[trk_id].append(pid)
        for trk_id, pids in by_track.items():
            total += 1
            from collections import Counter
            top = Counter(pids).most_common(1)
            if not top or top[0][1] / len(pids) < 0.5:
                fakes += 1
    if total == 0:
        return 0.0
    return round(fakes / total, 6)


def duplicate_rate(preds: pa.Table, truth_hits: pa.Table) -> float:
    """Fraction of truth particles matched to more than one reco track."""
    # particle -> set of matched track_ids
    truth_cols = truth_hits.to_pydict()
    pid_col = truth_cols.get("particle_id") or truth_cols.get("majority_particle_id")
    truth_map = {
        (e, h): p
        for e, h, p in zip(
            truth_cols["event_id"], truth_cols["hit_id"], pid_col
        )
    }
    pred_cols = preds.to_pydict()
    particle_tracks: dict = defaultdict(set)
    for e, h, t in zip(pred_cols["event_id"], pred_cols["hit_id"], pred_cols["track_id"]):
        pid = truth_map.get((e, h))
        if pid is not None:
            particle_tracks[(e, pid)].add(t)
    if not particle_tracks:
        return 0.0
    dupes = sum(1 for s in particle_tracks.values() if len(s) > 1)
    return round(dupes / len(particle_tracks), 6)


def physics_eff_pt1(preds: pa.Table, particles: pa.Table) -> float:
    """Fraction of truth particles with pT > 1 GeV that have a reco track."""
    truth_cols = particles.to_pydict()
    px = np.array(truth_cols.get("px", []), dtype=float)
    py = np.array(truth_cols.get("py", []), dtype=float)
    pt = np.sqrt(px * px + py * py) if len(px) else np.array([])
    pids = truth_cols.get("particle_id", [])
    primary = truth_cols.get("primary", [True] * len(pids))
    high_pt = set()
    for i, (p, pr) in enumerate(zip(pids, primary)):
        if not pr:
            continue
        if i < len(pt) and pt[i] > 1.0:
            high_pt.add(p)
    if not high_pt:
        return 0.0

    # Compare against reco tracks: use majority_particle_id if present
    matched = set()
    pcols = preds.to_pydict()
    if "majority_particle_id" in pcols:
        matched = set(int(p) for p in pcols["majority_particle_id"] if p in high_pt)
    return round(len(matched) / len(high_pt), 6)
