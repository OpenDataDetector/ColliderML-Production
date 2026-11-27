#!/usr/bin/env python3
"""
Compare the kinematics and link counts of the Pandora output of two EDM4hep files:
  - GaudiPandoraPFOs (PDG/type, energy, momentum, referencePoint, charge, mass,
                      linked cluster count + energies/positions, linked track count)
  - GaudiPandoraClusters (type, energy, energyError, position, iTheta, phi,
                          directionError, subdetectorEnergies, linked calo hit count + energy sum)
  - GaudiPandoraStartVertices (algorithmType/PDG, chi2, ndf, position)

This is the first check of compare_reco.py, which adds the remaining fields, the relation
targets and the calorimeter hits. A collection missing from a file is read as empty here;
compare_reco.py treats that as an error.

Exits 0 if max |delta| == 0.0 across the fields above, in order or after sorting.
Exits 2 if not, but the summary metrics agree within --metrics-tol.
Exits 1 otherwise.
"""
import argparse
import math
import sys
from podio import root_io


def _get_coll(frame, names):
    for n in names:
        try:
            c = frame.get(n)
            if c is not None:
                return c
        except Exception:
            pass
    return None


def _pid(obj):
    for acc in ("getPDG", "getType", "getAlgorithmType"):
        if hasattr(obj, acc):
            return int(getattr(obj, acc)())
    return 0


def _vec3(v):
    return (float(v.x), float(v.y), float(v.z))


def _extract_event(frame):
    pfos = _get_coll(frame, ["GaudiPandoraPFOs", "PandoraPFOs"])
    clus = _get_coll(frame, ["GaudiPandoraClusters", "PandoraClusters"])
    vtxs = _get_coll(frame, ["GaudiPandoraStartVertices", "PandoraStartVertices"])

    pfo_list = []
    if pfos is not None:
        for p in pfos:
            c_links = []
            if hasattr(p, "getClusters"):
                for c in p.getClusters():
                    c_links.append((float(c.getEnergy()),) + _vec3(c.getPosition()))
            n_trks = len(p.getTracks()) if hasattr(p, "getTracks") else 0
            ref_pt = _vec3(p.getReferencePoint()) if hasattr(p, "getReferencePoint") else (0.0, 0.0, 0.0)
            pfo_list.append({
                "pdg": _pid(p),
                "charge": float(p.getCharge()),
                "energy": float(p.getEnergy()),
                "mass": float(p.getMass()) if hasattr(p, "getMass") else 0.0,
                "mom": _vec3(p.getMomentum()),
                "ref": ref_pt,
                "n_tracks": int(n_trks),
                "clusters": tuple(c_links),
            })

    clu_list = []
    if clus is not None:
        for c in clus:
            n_hits = 0
            hit_e_sum = 0.0
            if hasattr(c, "getHits"):
                hits = c.getHits()
                n_hits = len(hits)
                for h in hits:
                    hit_e_sum += float(h.getEnergy())
            sub_e = tuple(float(x) for x in c.getSubdetectorEnergies()) if hasattr(c, "getSubdetectorEnergies") else ()
            dir_err = _vec3(c.getDirectionError()) if hasattr(c, "getDirectionError") else (0.0, 0.0, 0.0)
            clu_list.append({
                "type": int(c.getType()) if hasattr(c, "getType") else 0,
                "energy": float(c.getEnergy()),
                "energy_err": float(c.getEnergyError()) if hasattr(c, "getEnergyError") else 0.0,
                "pos": _vec3(c.getPosition()),
                "theta": float(c.getITheta()) if hasattr(c, "getITheta") else 0.0,
                "phi": float(c.getPhi()) if hasattr(c, "getPhi") else 0.0,
                "dir_err": dir_err,
                "sub_e": sub_e,
                "n_hits": int(n_hits),
                "hit_e_sum": float(hit_e_sum),
            })

    vtx_list = []
    if vtxs is not None:
        for v in vtxs:
            vtx_list.append({
                "type": _pid(v),
                "chi2": float(v.getChi2()) if hasattr(v, "getChi2") else 0.0,
                "ndf": int(v.getNdf()) if hasattr(v, "getNdf") else 0,
                "pos": _vec3(v.getPosition()),
            })

    return {"pfos": pfo_list, "clusters": clu_list, "vertices": vtx_list}


def load_file(fn):
    reader = root_io.Reader(str(fn))
    return [_extract_event(frame) for frame in reader.get("events")]


def _pfo_sort_key(p):
    return (p["pdg"], round(p["energy"], 5), round(p["mom"][0], 5), round(p["mom"][1], 5), round(p["mom"][2], 5), p["n_tracks"], len(p["clusters"]))


def _clu_sort_key(c):
    return (c["type"], round(c["energy"], 5), round(c["pos"][0], 4), round(c["pos"][1], 4), round(c["pos"][2], 4), c["n_hits"])


def _vtx_sort_key(v):
    return (v["type"], round(v["pos"][0], 5), round(v["pos"][1], 5), round(v["pos"][2], 5))


def _max_float_diff(a, b):
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) or math.isnan(b):
            return 0.0 if (math.isnan(a) and math.isnan(b)) else float("inf")
        return abs(a - b)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        if len(a) != len(b):
            return float("inf")
        return max((_max_float_diff(x, y) for x, y in zip(a, b)), default=0.0)
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return float("inf")
        return max((_max_float_diff(a[k], b[k]) for k in a), default=0.0)
    return 0.0 if a == b else float("inf")


def compare_events(evs_a, evs_b, order_invariant=False):
    if len(evs_a) != len(evs_b):
        return False, float("inf"), f"event count mismatch: {len(evs_a)} vs {len(evs_b)}"

    max_diff = 0.0
    mismatch_msgs = []

    for ie, (ea, eb) in enumerate(zip(evs_a, evs_b)):
        for coll, key_fn in [("pfos", _pfo_sort_key), ("clusters", _clu_sort_key), ("vertices", _vtx_sort_key)]:
            la, lb = ea[coll], eb[coll]
            if len(la) != len(lb):
                mismatch_msgs.append(f"ev {ie} {coll} len {len(la)} != {len(lb)}")
                max_diff = float("inf")
                continue
            if order_invariant:
                la = sorted(la, key=key_fn)
                lb = sorted(lb, key=key_fn)
            for idx, (xa, xb) in enumerate(zip(la, lb)):
                d = _max_float_diff(xa, xb)
                if d > max_diff:
                    max_diff = d
                if d > 0.0 and len(mismatch_msgs) < 10:
                    mismatch_msgs.append(f"ev {ie} {coll}[{idx}] diff={d:.3e}")

    ok = (max_diff == 0.0 and not mismatch_msgs)
    return ok, max_diff, "; ".join(mismatch_msgs[:5])


def compute_metrics(evs):
    m = {"npfo": 0, "epfo": 0.0, "ppfo": 0.0, "nclu": 0, "eclu": 0.0, "nhit_clu": 0, "types": {}}
    for e in evs:
        for p in e["pfos"]:
            m["npfo"] += 1
            m["epfo"] += p["energy"]
            mx, my, mz = p["mom"]
            m["ppfo"] += math.sqrt(mx * mx + my * my + mz * mz)
            t = p["pdg"]
            m["types"][t] = m["types"].get(t, 0) + 1
        for c in e["clusters"]:
            m["nclu"] += 1
            m["eclu"] += c["energy"]
            m["nhit_clu"] += c["n_hits"]
    return m


def reldiff_pct(a, b):
    if a == 0 and b == 0:
        return 0.0
    return 100.0 * (b - a) / (a if a != 0 else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("cand")
    ap.add_argument("--float-tol", type=float, default=0.0, help="Allowed float tolerance for byte-identical check")
    ap.add_argument("--metrics-tol", type=float, default=0.0, help="Allowed % tolerance for metrics-identical fallback")
    args = ap.parse_args()

    evs_a = load_file(args.ref)
    evs_b = load_file(args.cand)

    exact_order_ok, exact_order_diff, exact_msg = compare_events(evs_a, evs_b, order_invariant=False)
    sorted_ok, sorted_diff, sorted_msg = compare_events(evs_a, evs_b, order_invariant=True)

    ma = compute_metrics(evs_a)
    mb = compute_metrics(evs_b)

    worst_pct = 0.0
    for k in ("npfo", "epfo", "ppfo", "nclu", "eclu", "nhit_clu"):
        worst_pct = max(worst_pct, abs(reldiff_pct(ma[k], mb[k])))
    all_types = sorted(set(ma["types"]) | set(mb["types"]))
    for t in all_types:
        ca, cb = ma["types"].get(t, 0), mb["types"].get(t, 0)
        if ca + cb >= 20:
            worst_pct = max(worst_pct, abs(reldiff_pct(ca, cb)))

    print(f"Events={len(evs_a)} | PFOs={ma['npfo']}/{mb['npfo']} | Clusters={ma['nclu']}/{mb['nclu']} | "
          f"exact_order max|d|={exact_order_diff:.3e} | sorted max|d|={sorted_diff:.3e} | worst_metric_rel%={worst_pct:.4f}%")

    if exact_order_diff <= args.float_tol:
        print("RESULT: BYTE_IDENTICAL_EXACT_ORDER")
        sys.exit(0)
    if sorted_diff <= args.float_tol:
        print(f"RESULT: BYTE_IDENTICAL_SORTED (order differs: {exact_msg})")
        sys.exit(0)
    if worst_pct <= args.metrics_tol:
        print(f"RESULT: METRICS_IDENTICAL (sorted max|d|={sorted_diff:.3e}: {sorted_msg})")
        sys.exit(2)

    print(f"RESULT: DIFFERS ({sorted_msg})")
    for k in ("npfo", "epfo", "ppfo", "nclu", "eclu", "nhit_clu"):
        print(f"  {k:<10}: ref={ma[k]:.4f} cand={mb[k]:.4f} ({reldiff_pct(ma[k], mb[k]):+.4f}%)")
    for t in all_types:
        ca, cb = ma["types"].get(t, 0), mb["types"].get(t, 0)
        print(f"  pdg {t:<6}: ref={ca} cand={cb} ({reldiff_pct(ca, cb):+.4f}%)")
    sys.exit(1)


if __name__ == "__main__":
    main()
