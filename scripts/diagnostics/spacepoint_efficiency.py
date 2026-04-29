#!/usr/bin/env python3
"""Strip-spacepoint efficiency analyser, v5 ACTS Examples schema."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

ALLOWED = {(28, 2), (28, 4), (28, 6), (28, 8), (28, 10), (28, 12),
           (30, 2), (30, 4), (30, 6), (30, 8), (30, 10), (30, 12)}


def pack_barcode(vp, vs, p, g, sub):
    return ((vp.astype(np.uint64) << np.uint64(52))
            | (vs.astype(np.uint64) << np.uint64(40))
            | (p.astype(np.uint64) << np.uint64(24))
            | (g.astype(np.uint64) << np.uint64(16))
            | sub.astype(np.uint64))


def load_inputs(run_dir, max_events):
    sp = uproot.open(f"{run_dir}/spacepoints.root")["spacepoints"].arrays(
        library="np")
    if max_events is not None:
        keep = sp["event_id"] < max_events
        sp = {k: v[keep] for k, v in sp.items()}

    m = uproot.open(f"{run_dir}/measurements.root")["measurements"].arrays(
        ["event_nr", "volume_id", "layer_id", "surface_id",
         "particles_vertex_primary", "particles_vertex_secondary",
         "particles_particle", "particles_generation", "particles_sub_particle"],
        library="ak")
    m_ev = ak.to_numpy(m["event_nr"])
    if max_events is not None:
        m = m[m_ev < max_events]
        m_ev = ak.to_numpy(m["event_nr"])

    p = uproot.open(f"{run_dir}/particles.root")["particles"].arrays(
        ["event_id", "vertex_primary", "vertex_secondary", "particle",
         "generation", "sub_particle", "pt"], library="ak")
    p_ev_per_event = ak.to_numpy(p["event_id"]).astype(int)
    if max_events is not None:
        keep = p_ev_per_event < max_events
        p = p[keep]
        p_ev_per_event = p_ev_per_event[keep]

    return sp, m, m_ev, p, p_ev_per_event


def build_measurement_psets(m, n_m):
    def flat(field):
        return ak.to_numpy(ak.flatten(m[field])).astype(np.uint64)

    bc = pack_barcode(flat("particles_vertex_primary"),
                      flat("particles_vertex_secondary"),
                      flat("particles_particle"),
                      flat("particles_generation"),
                      flat("particles_sub_particle"))
    counts = ak.num(m["particles_vertex_primary"]).to_numpy()
    offs = np.concatenate([[0], np.cumsum(counts)])
    return [frozenset(bc[offs[i]:offs[i + 1]].tolist()) for i in range(n_m)]


def build_particle_info(p, p_ev_per_event):
    def flat(field):
        return ak.to_numpy(ak.flatten(p[field])).astype(np.uint64)

    counts = ak.num(p["vertex_primary"]).to_numpy()
    p_ev = np.repeat(p_ev_per_event, counts)
    p_vs = flat("vertex_secondary")
    p_gg = flat("generation")
    p_bc = pack_barcode(flat("vertex_primary"), p_vs, flat("particle"),
                        p_gg, flat("sub_particle"))
    p_pt = ak.to_numpy(ak.flatten(p["pt"])).astype(np.float32)

    info = {}
    for i in range(len(p_bc)):
        is_primary = (p_vs[i] == 0) and (p_gg[i] == 0)
        info[(int(p_ev[i]), int(p_bc[i]))] = (bool(is_primary), float(p_pt[i]))
    return info


def compute_within_event_indices(m_ev, n_m):
    # v5 measurements are written out-of-event-order; remap.
    mid_per_row = np.zeros(n_m, dtype=int)
    seen = defaultdict(int)
    for i, e in enumerate(m_ev):
        mid_per_row[i] = seen[int(e)]
        seen[int(e)] += 1
    return {(int(m_ev[i]), int(mid_per_row[i])): i for i in range(n_m)}


def report(label, sp, mv, ml, ms, m_ev, m_psets, ev_mid_to_row, n_sp,
           n_m, passes):
    # SP couples per (event, vol, lay): {(lo, hi)}, plus the set of
    # passing-filter particles shared between the two faces of each SP.
    sp_couple_pset = defaultdict(set)  # (event, vol, lay, lo, hi) -> shared&passing particles
    built_per_layer = defaultdict(int)
    correct_built_per_layer = defaultdict(int)
    n_correct_built = 0
    for i in range(n_sp):
        e = int(sp["event_id"][i])
        r1 = ev_mid_to_row.get((e, int(sp["measurement_id"][i])))
        r2 = ev_mid_to_row.get((e, int(sp["measurement_id_2"][i])))
        if r1 is None or r2 is None:
            continue
        v, l = int(mv[r1]), int(ml[r1])
        if (v, l) not in ALLOWED:
            continue
        built_per_layer[(v, l)] += 1
        s1, s2 = int(ms[r1]), int(ms[r2])
        lo, hi = min(s1, s2), max(s1, s2)
        shared_passing = {bc for bc in (m_psets[r1] & m_psets[r2])
                          if passes(e, bc)}
        if shared_passing:
            sp_couple_pset[(e, v, l, lo, hi)] |= shared_passing
            n_correct_built += 1
            correct_built_per_layer[(v, l)] += 1

    # Truth pairs: per (event, v, l, particle), find the (s, s+1) couple
    # the particle hit. One per particle per layer.
    truth_hits = defaultdict(set)
    for i in range(n_m):
        if (int(mv[i]), int(ml[i])) not in ALLOWED:
            continue
        e = int(m_ev[i])
        for bc in m_psets[i]:
            if passes(e, bc):
                truth_hits[(e, int(mv[i]), int(ml[i]), int(bc))].add(
                    int(ms[i]))

    true_per_layer = defaultdict(int)
    matched_per_layer = defaultdict(int)
    true_total = 0
    matched_total = 0
    for (e, v, l, bc), surfs in truth_hits.items():
        sset = set(surfs)
        couples = [(s, s + 1) for s in surfs
                   if s % 2 == 1 and (s + 1) in sset]
        if not couples:
            continue
        true_total += 1
        true_per_layer[(v, l)] += 1
        if any(bc in sp_couple_pset.get((e, v, l, lo, hi), set())
               for lo, hi in couples):
            matched_total += 1
            matched_per_layer[(v, l)] += 1

    n_built = sum(built_per_layer.values())
    eff = 100 * matched_total / true_total if true_total else 0.0
    fake = 100 * (n_built - n_correct_built) / n_built if n_built else 0.0
    print(f"\n=== Variant: {label} ===")
    print(f"  N built:              {n_built}")
    print(f"  N correct (any built shares a passing particle): {n_correct_built}")
    print(f"  N true cluster pairs: {true_total}")
    print(f"  N truth pairs matched by a built SP at the right couple: {matched_total}")
    print(f"  Headline efficiency:  {eff:.2f}%  (matched/true)")
    print(f"  Headline fake rate:   {fake:.2f}%  ((built-correct)/built)")
    print(f"\n  Per-(vol, layer):")
    print(f"  {'(v, l)':<10} {'built':>8} {'correct':>8} {'true':>8} {'eff%':>8} {'fake%':>8}")
    for vl in sorted(ALLOWED):
        b = built_per_layer[vl]
        c = matched_per_layer[vl]
        t = true_per_layer[vl]
        e_ = 100 * c / t if t else 0.0
        f_ = 100 * (b - correct_built_per_layer[vl]) / b if b else 0.0
        print(f"  {str(vl):<10} {b:>8} {c:>8} {t:>8} {e_:>7.1f}% {f_:>7.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Directory containing {spacepoints,measurements,particles}.root")
    ap.add_argument("--max-events", type=int, default=None,
                    help="Restrict to event_id < N. Use --max-events 1 for a smoke test.")
    ap.add_argument("--pt-min", type=float, default=1.0,
                    help="pT cut for the primary+pT variant (GeV)")
    args = ap.parse_args()

    print(f"Loading {args.run_dir}", file=sys.stderr)
    sp, m, m_ev, p, p_ev_per_event = load_inputs(args.run_dir, args.max_events)
    n_sp = len(sp["event_id"])
    n_m = len(m_ev)
    print(f"  spacepoints: {n_sp}, measurements: {n_m}", file=sys.stderr)

    m_psets = build_measurement_psets(m, n_m)
    p_info = build_particle_info(p, p_ev_per_event)
    print(f"  particles: {len(p_info)} (in {len(p_ev_per_event)} events)",
          file=sys.stderr)

    mv = ak.to_numpy(m["volume_id"]).astype(int)
    ml = ak.to_numpy(m["layer_id"]).astype(int)
    ms = ak.to_numpy(m["surface_id"]).astype(int)
    ev_mid_to_row = compute_within_event_indices(m_ev, n_m)

    def all_particles(_e, _bc):
        return True

    def primary_high_pt(e, bc):
        info = p_info.get((e, int(bc)))
        if info is None:
            return False
        is_primary, pt = info
        return is_primary and pt > args.pt_min

    report("all particles",
           sp, mv, ml, ms, m_ev, m_psets, ev_mid_to_row, n_sp, n_m,
           all_particles)
    report(f"primary + pT > {args.pt_min} GeV",
           sp, mv, ml, ms, m_ev, m_psets, ev_mid_to_row, n_sp, n_m,
           primary_high_pt)


if __name__ == "__main__":
    main()
