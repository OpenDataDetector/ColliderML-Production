#!/usr/bin/env python3
"""Spacepoint efficiency analyser.

Implements docs/spacepoint_efficiency_spec.md. Given a digi run directory
(containing spacepoints.root, measurements.root, particles.root), compute
microaveraged strip spacepoint efficiency and fake rate, in two variants:

    1. all-particles
    2. primary + pT > 1 GeV  (primary = vertex_secondary==0 & generation==0
                              & sub_particle==0)

Reports headline + per-(volume, layer) breakdown + per-event distribution.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot


def _now() -> float:
    return time.time()


def load_inputs(run_dir: Path, max_events: int | None = None):
    """Load the three ROOT files. Returns (sp_arrs, m_table, p_data).

    - sp_arrs: dict of np arrays from spacepoints.root.
    - m_table: dict with numpy arrays for event_nr, volume_id, layer_id,
        extra_id, mid (within-event index), and `particles` as awkward
        jagged uint32 (one flat list per measurement, channels merged).
    - p_data: dict event_id -> per-event particle DataFrame.

    If `max_events` is set, restrict to event_id < max_events in all three
    inputs (for fast iteration during development).
    """
    sp_path = run_dir / "spacepoints.root"
    m_path = run_dir / "measurements.root"
    p_path = run_dir / "particles.root"
    for p in (sp_path, m_path, p_path):
        if not p.exists():
            raise FileNotFoundError(p)

    t0 = _now()
    print(f"  loading spacepoints.root ...", file=sys.stderr)
    sp_arrs = uproot.open(sp_path)["spacepoints"].arrays(library="np")
    if max_events is not None:
        sp_keep = sp_arrs["event_id"] < max_events
        sp_arrs = {k: v[sp_keep] for k, v in sp_arrs.items()}
    print(
        f"    -> {len(sp_arrs['event_id'])} spacepoint rows (took {_now()-t0:.1f}s)",
        file=sys.stderr,
    )

    t0 = _now()
    print(f"  loading measurements.root ...", file=sys.stderr)
    m_tree = uproot.open(m_path)["measurements"]
    m_branches = ["event_nr", "volume_id", "layer_id", "particles"]
    if "extra_id" in m_tree.keys():
        m_branches.append("extra_id")
    m_arrs = m_tree.arrays(m_branches, library="ak")

    event_nr_full = ak.to_numpy(m_arrs["event_nr"]).astype(np.uint32)
    if max_events is not None:
        m_keep_mask = event_nr_full < max_events
        m_arrs = m_arrs[m_keep_mask]
        event_nr_full = event_nr_full[m_keep_mask]
        print(
            f"    restricted measurements to events < {max_events}: "
            f"{len(event_nr_full)} rows",
            file=sys.stderr,
        )
    print(
        f"    -> {len(event_nr_full)} measurement rows (took {_now()-t0:.1f}s)",
        file=sys.stderr,
    )

    event_nr = event_nr_full
    volume_id = ak.to_numpy(m_arrs["volume_id"]).astype(np.uint16)
    layer_id = ak.to_numpy(m_arrs["layer_id"]).astype(np.uint16)
    extra_id = (
        ak.to_numpy(m_arrs["extra_id"]).astype(np.uint16)
        if "extra_id" in m_arrs.fields
        else None
    )

    # Within-event index for each measurement (matches sourceLink.index()).
    mid = pd.Series(event_nr).groupby(event_nr).cumcount().to_numpy().astype(np.uint32)

    # Flatten the per-channel inner ragged dimension. m_arrs["particles"]
    # has shape n * var * var * uint32; we want n * var * uint32 (one flat
    # list of contributing particle ids per measurement).
    parts = ak.flatten(m_arrs["particles"], axis=2)

    # Build first_idx[event] = first row index for that event, so we can
    # convert (event, mid) to a global row index in O(1).
    n_events_max = int(event_nr.max()) + 1
    first_idx = np.full(n_events_max, -1, dtype=np.int64)
    diffs = np.diff(event_nr.astype(np.int64), prepend=-1)
    boundaries = np.where(diffs != 0)[0]
    for b in boundaries:
        first_idx[int(event_nr[b])] = int(b)

    m_table = {
        "event_nr": event_nr,
        "volume_id": volume_id,
        "layer_id": layer_id,
        "extra_id": extra_id,
        "mid": mid,
        "particles": parts,
        "first_idx": first_idx,
    }

    t0 = _now()
    print(f"  loading particles.root ...", file=sys.stderr)
    p_tree = uproot.open(p_path)["particles"]
    p_arrs = p_tree.arrays(library="ak")
    p_data: dict[int, pd.DataFrame] = {}
    n_total = p_tree.num_entries
    if max_events is not None:
        n_total = min(n_total, max_events)
    for i in range(n_total):
        ev = int(p_arrs["event_id"][i])
        if max_events is not None and ev >= max_events:
            continue
        p_data[ev] = pd.DataFrame(
            {
                "particle_hash": ak.to_numpy(p_arrs["particle_hash"][i]),
                "vertex_primary": ak.to_numpy(p_arrs["vertex_primary"][i]),
                "vertex_secondary": ak.to_numpy(p_arrs["vertex_secondary"][i]),
                "particle": ak.to_numpy(p_arrs["particle"][i]),
                "generation": ak.to_numpy(p_arrs["generation"][i]),
                "sub_particle": ak.to_numpy(p_arrs["sub_particle"][i]),
                "pt": ak.to_numpy(p_arrs["pt"][i]),
                "particle_type": ak.to_numpy(p_arrs["particle_type"][i]),
            }
        )
    print(f"    -> {len(p_data)} events loaded (took {_now()-t0:.1f}s)", file=sys.stderr)
    return sp_arrs, m_table, p_data


def discover_join_key(m_table: dict, p_data: dict[int, pd.DataFrame]) -> str:
    """Empirically figure out which particles.root column corresponds to
    the uint32 in measurements.root's `particles` field. We test:
      - particle_hash (lower 32 bits)
      - particle_hash (upper 32 bits)
      - particle (uint32 directly)
    Pick the column with highest overlap on a sample event.
    """
    ev = int(m_table["event_nr"][0])
    if ev not in p_data:
        raise RuntimeError(f"particles.root has no event {ev}")

    # Sample first ~5000 measurements of this event
    mask = m_table["event_nr"] == ev
    sample_idx = np.where(mask)[0][:5000]
    m_pids: set[int] = set()
    for i in sample_idx:
        for x in ak.to_numpy(m_table["particles"][i]):
            m_pids.add(int(x))
    if not m_pids:
        raise RuntimeError(f"event {ev} measurements have no particles at all")

    p_event = p_data[ev]
    candidates = {
        "particle_hash_low32": set(
            int(h) & 0xFFFFFFFF for h in p_event["particle_hash"].tolist()
        ),
        "particle_hash_high32": set(
            int(h) >> 32 for h in p_event["particle_hash"].tolist()
        ),
        "particle": set(int(x) for x in p_event["particle"].tolist()),
    }
    overlaps = {k: len(m_pids & v) for k, v in candidates.items()}
    print(
        f"  [join] event {ev}: |m_pids|={len(m_pids)}, overlaps={overlaps}",
        file=sys.stderr,
    )
    best = max(overlaps, key=lambda k: overlaps[k])
    if overlaps[best] == 0:
        raise RuntimeError(f"No overlap on event {ev}")
    return best


def make_particle_filter(
    p_data: dict[int, pd.DataFrame],
    join_key: str,
    *,
    require_primary: bool,
    pt_min: float,
) -> dict[int, set[int]] | None:
    """Per-event set of particle ids (truncated per join_key) that pass
    the requested filter. Returns None if no filter is requested (passthrough).
    """
    if not require_primary and pt_min <= 0:
        return None
    out: dict[int, set[int]] = {}
    for ev, df in p_data.items():
        mask = pd.Series(True, index=df.index)
        if require_primary:
            mask &= (
                (df["vertex_secondary"] == 0)
                & (df["generation"] == 0)
                & (df["sub_particle"] == 0)
            )
        if pt_min > 0:
            mask &= df["pt"] > pt_min
        if join_key == "particle_hash_low32":
            ids_64 = df.loc[mask, "particle_hash"].to_numpy().astype(np.uint64)
            ids = ids_64 & np.uint64(0xFFFFFFFF)
        elif join_key == "particle_hash_high32":
            ids_64 = df.loc[mask, "particle_hash"].to_numpy().astype(np.uint64)
            ids = ids_64 >> np.uint64(32)
        elif join_key == "particle":
            ids = df.loc[mask, "particle"].to_numpy().astype(np.uint64)
        else:
            raise ValueError(f"unknown join key {join_key}")
        out[ev] = set(int(x) for x in ids)
    return out


def compute_denominator(
    m_table: dict,
    layers: list[tuple[int, int]],
    particle_allow: dict[int, set[int]] | None,
) -> dict[tuple[int, int, int], int]:
    """Per (event, vol, layer): truth particles that hit BOTH extra=1 and 2."""
    if m_table["extra_id"] is None:
        raise RuntimeError("measurements.root has no extra_id branch")

    layer_set = set(layers)
    event_nr = m_table["event_nr"]
    volume_id = m_table["volume_id"]
    layer_id = m_table["layer_id"]
    extra_id = m_table["extra_id"]
    parts = m_table["particles"]

    # Mask: only listed (vol, layer) and extra in {1, 2}
    layer_keys = (volume_id.astype(np.uint64) << np.uint64(16)) | layer_id.astype(np.uint64)
    layer_set_keys = np.array(
        sorted(np.uint64(v) << np.uint64(16) | np.uint64(l) for (v, l) in layer_set),
        dtype=np.uint64,
    )
    layer_mask = np.isin(layer_keys, layer_set_keys)
    extra_mask = (extra_id == 1) | (extra_id == 2)
    keep = layer_mask & extra_mask
    print(
        f"  [denominator] kept {int(keep.sum())} / {len(event_nr)} measurements "
        f"(layers + extra in 1,2)",
        file=sys.stderr,
    )

    ev_k = event_nr[keep]
    vol_k = volume_id[keep]
    ly_k = layer_id[keep]
    ex_k = extra_id[keep]
    parts_k = parts[keep]

    # Explode (measurement -> particles). Repeat each row by num particles,
    # then flatten parts.
    nparts = ak.to_numpy(ak.num(parts_k)).astype(np.int64)
    ev_rep = np.repeat(ev_k, nparts)
    vol_rep = np.repeat(vol_k, nparts)
    ly_rep = np.repeat(ly_k, nparts)
    ex_rep = np.repeat(ex_k, nparts)
    parts_flat = ak.to_numpy(ak.flatten(parts_k)).astype(np.uint64)

    flat = pd.DataFrame(
        {
            "event": ev_rep.astype(np.uint32),
            "vol": vol_rep.astype(np.uint16),
            "layer": ly_rep.astype(np.uint16),
            "extra": ex_rep.astype(np.uint8),
            "particle": parts_flat,
        }
    )
    print(f"  [denominator] exploded to {len(flat)} (meas, particle) rows", file=sys.stderr)

    if particle_allow is not None:
        # For each event, restrict to particles in allow[event].
        allow_pairs = set()
        for ev, ids in particle_allow.items():
            for pid in ids:
                allow_pairs.add((int(ev), int(pid)))
        ep_tuples = list(zip(flat["event"].tolist(), flat["particle"].tolist()))
        mask = pd.Series(ep_tuples).isin(allow_pairs).to_numpy()
        flat = flat[mask].reset_index(drop=True)
        print(f"  [denominator] after particle filter: {len(flat)} rows", file=sys.stderr)

    if flat.empty:
        return {}

    # Group by (event, vol, layer, particle): keep groups with both extras 1 and 2.
    g = flat.groupby(["event", "vol", "layer", "particle"])["extra"]
    has_one = g.transform(lambda s: (s == 1).any())
    has_two = g.transform(lambda s: (s == 2).any())
    flat["both"] = has_one & has_two
    distinct = flat[flat["both"]].drop_duplicates(["event", "vol", "layer", "particle"])
    counts = distinct.groupby(["event", "vol", "layer"]).size()

    out: dict[tuple[int, int, int], int] = {}
    for (ev, v, ly), c in counts.items():
        out[(int(ev), int(v), int(ly))] = int(c)
    return out


def compute_numerator(
    sp_arrs: dict,
    m_table: dict,
    layers: list[tuple[int, int]],
    particle_allow: dict[int, set[int]] | None,
):
    """Per-(event, vol, layer): correct (shared-particle) and total spacepoints."""
    layer_set = set(layers)
    g1 = sp_arrs["geometry_id"].astype(np.uint64)
    sp_vol = ((g1 >> np.uint64(56)) & np.uint64(0xFF)).astype(np.uint16)
    sp_layer = ((g1 >> np.uint64(36)) & np.uint64(0xFFF)).astype(np.uint16)
    sp_ev = sp_arrs["event_id"].astype(np.uint32)
    sp_m1 = sp_arrs["measurement_id"].astype(np.uint32)
    sp_m2 = sp_arrs["measurement_id_2"].astype(np.uint32)

    n_sp = len(sp_ev)

    layer_keep = np.array(
        [(int(v), int(l)) in layer_set for v, l in zip(sp_vol, sp_layer)],
        dtype=bool,
    )
    print(
        f"  [numerator] kept {int(layer_keep.sum())} / {n_sp} spacepoints in listed layers",
        file=sys.stderr,
    )

    first_idx = m_table["first_idx"]
    parts = m_table["particles"]

    correct: dict[tuple[int, int, int], int] = {}
    total: dict[tuple[int, int, int], int] = {}

    log_every = max(1, n_sp // 10)
    t0 = _now()
    for i in range(n_sp):
        if i % log_every == 0 and i > 0:
            print(
                f"    [numerator] {i}/{n_sp} ({100*i/n_sp:.0f}%) elapsed {_now()-t0:.1f}s",
                file=sys.stderr,
            )
        if not layer_keep[i]:
            continue
        ev = int(sp_ev[i])
        v, ly = int(sp_vol[i]), int(sp_layer[i])
        bucket = (ev, v, ly)
        total[bucket] = total.get(bucket, 0) + 1

        if first_idx[ev] < 0:
            continue
        idx1 = int(first_idx[ev]) + int(sp_m1[i])
        idx2 = int(first_idx[ev]) + int(sp_m2[i])
        s1 = set(int(x) for x in ak.to_numpy(parts[idx1]))
        s2 = set(int(x) for x in ak.to_numpy(parts[idx2]))
        shared = s1 & s2
        if particle_allow is not None:
            shared &= particle_allow.get(ev, set())
        if shared:
            correct[bucket] = correct.get(bucket, 0) + 1
    print(
        f"    [numerator] done {n_sp}/{n_sp} elapsed {_now()-t0:.1f}s",
        file=sys.stderr,
    )
    return correct, total


def microaverage(
    correct: dict, total: dict, denom: dict
) -> tuple[float, float, int, int, int]:
    n_correct = sum(correct.values())
    n_total = sum(total.values())
    n_denom = sum(denom.values())
    eff = n_correct / n_denom if n_denom else 0.0
    fake = (n_total - n_correct) / n_total if n_total else 0.0
    return eff, fake, n_correct, n_total, n_denom


def per_layer_report(correct: dict, total: dict, denom: dict, layers: list):
    rows = []
    for v, ly in layers:
        c = sum(cnt for (e, vv, ll), cnt in correct.items() if (vv, ll) == (v, ly))
        t = sum(cnt for (e, vv, ll), cnt in total.items() if (vv, ll) == (v, ly))
        d = sum(cnt for (e, vv, ll), cnt in denom.items() if (vv, ll) == (v, ly))
        rows.append(
            {
                "vol": v,
                "layer": ly,
                "n_correct": c,
                "n_total": t,
                "n_denom": d,
                "efficiency": c / d if d else 0.0,
                "fake_rate": (t - c) / t if t else 0.0,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--layers-json", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--pt-min", type=float, default=1.0)
    ap.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Restrict to event_id < N. Use --max-events 1 for a smoke test.",
    )
    args = ap.parse_args()

    print(
        f"Loading {args.run_dir}"
        + (f" (max-events={args.max_events})" if args.max_events else ""),
        file=sys.stderr,
    )
    sp_arrs, m_table, p_data = load_inputs(args.run_dir, max_events=args.max_events)
    print(
        f"  spacepoints={len(sp_arrs['event_id'])} entries\n"
        f"  measurements={len(m_table['event_nr'])} rows over "
        f"{int(np.unique(m_table['event_nr']).size)} events\n"
        f"  particles.root events={len(p_data)}",
        file=sys.stderr,
    )

    join_key = discover_join_key(m_table, p_data)
    print(f"  [join] using key '{join_key}'", file=sys.stderr)

    # Discover layers
    if args.layers_json is None:
        g1 = sp_arrs["geometry_id"].astype(np.uint64)
        v = ((g1 >> np.uint64(56)) & np.uint64(0xFF))
        ly = ((g1 >> np.uint64(36)) & np.uint64(0xFFF))
        layers_all = sorted({(int(a), int(b)) for a, b in zip(v, ly)})
        layers = [x for x in layers_all if x[0] != 29]
        print(
            f"  [layers] discovered from spacepoints.root: {layers_all}; "
            f"vol 29 dropped -> evaluating {layers}",
            file=sys.stderr,
        )
    else:
        with open(args.layers_json) as f:
            data = json.load(f)
        layers = sorted({(int(d["volume"]), int(d["layer"])) for d in data})
        layers = [(v, l) for (v, l) in layers if v != 29]
        print(f"  [layers] from JSON, dropping vol 29: {layers}", file=sys.stderr)

    results = {}
    for variant_name, require_primary, pt_min in [
        ("all_particles", False, 0.0),
        ("primary_pt_gt_1gev", True, args.pt_min),
    ]:
        print(f"\n=== Variant: {variant_name} ===", file=sys.stderr)
        allow = make_particle_filter(
            p_data, join_key, require_primary=require_primary, pt_min=pt_min
        )
        if allow is not None:
            print(
                f"  filter passes {sum(len(s) for s in allow.values())} particles",
                file=sys.stderr,
            )

        denom = compute_denominator(m_table, layers, allow)
        correct, total = compute_numerator(sp_arrs, m_table, layers, allow)

        eff, fake, n_correct, n_total, n_denom = microaverage(correct, total, denom)
        rows = per_layer_report(correct, total, denom, layers)

        event_ids = sorted({ev for (ev, _, _) in denom} | {ev for (ev, _, _) in total})
        per_event = []
        for ev in event_ids:
            c_ev = sum(cnt for (e, _, _), cnt in correct.items() if e == ev)
            d_ev = sum(cnt for (e, _, _), cnt in denom.items() if e == ev)
            t_ev = sum(cnt for (e, _, _), cnt in total.items() if e == ev)
            per_event.append(
                {
                    "event": ev,
                    "n_correct": c_ev,
                    "n_total": t_ev,
                    "n_denom": d_ev,
                    "efficiency": (c_ev / d_ev) if d_ev else 0.0,
                    "fake_rate": ((t_ev - c_ev) / t_ev) if t_ev else 0.0,
                }
            )

        results[variant_name] = {
            "n_correct": int(n_correct),
            "n_total": int(n_total),
            "n_denom": int(n_denom),
            "efficiency": eff,
            "fake_rate": fake,
            "per_layer": rows,
            "per_event": per_event,
        }

    print("\n" + "=" * 78)
    print("SPACEPOINT EFFICIENCY REPORT")
    print("=" * 78)
    print(f"Run dir: {args.run_dir}")
    print(f"Layers : {layers}")
    for v_name, r in results.items():
        print(f"\n--- {v_name} ---")
        print(f"  efficiency : {r['efficiency']:.4f}  ({r['n_correct']} / {r['n_denom']})")
        print(f"  fake rate  : {r['fake_rate']:.4f}  (built {r['n_total']})")
        print("  per-layer:")
        print(
            f"    {'(vol,layer)':14s} {'eff':>8s} {'fake':>8s} {'correct':>8s} "
            f"{'total':>8s} {'denom':>8s}"
        )
        for row in r["per_layer"]:
            print(
                f"    ({row['vol']:3d},{row['layer']:4d})   "
                f"{row['efficiency']:8.4f} {row['fake_rate']:8.4f} "
                f"{row['n_correct']:8d} {row['n_total']:8d} {row['n_denom']:8d}"
            )

    if args.output:
        args.output.write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
