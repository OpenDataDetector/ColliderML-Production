#!/usr/bin/env python3
"""Quick content check of one pilot run directory: table event counts, per-event
multiplicities, cluster-shape columns, merged-cluster fraction, and the Release-2
reco tables if present. Read-only. Usage: check_run.py <run_dir>"""
import glob
import sys

import pyarrow.compute as pc
import pyarrow.parquet as pq

R = sys.argv[1] if len(sys.argv) > 1 else \
    "/global/cfs/cdirs/m4958/data/ColliderML/staging/release2_pilot/ttbar_pu0/v1/runs/0"
KEY = {"particles": "particle_id", "tracker_simhits": "particle_id", "tracker_hits": "loc0",
       "tracks": "d0", "truth_tracks": "d0", "calo_cells": "energy", "calo_clusters": "energy",
       "pfos": "energy"}
for t, col in KEY.items():
    fs = sorted(glob.glob(f"{R}/{t}/*.parquet"))
    if not fs:
        print(f"{t:16s} MISSING")
        continue
    tb = pq.read_table(fs[0], columns=["event_id", col])
    per = pc.list_value_length(tb[col])
    print(f"{t:16s} events={tb.num_rows:5d} per-event mean={pc.mean(per).as_py():9.1f}")

fs = glob.glob(f"{R}/tracker_hits/*.parquet")
if fs:
    h = pq.read_table(fs[0], columns=["size_loc0", "n_channels", "particle_ids"])
    s = pc.list_flatten(h["size_loc0"])
    print("tracker_hits: size_loc0 nonzero frac %.3f, n_channels mean %.2f" % (
        pc.sum(pc.greater(s, 0)).as_py() / len(s),
        pc.mean(pc.list_flatten(h["n_channels"])).as_py()))
    p = pc.list_flatten(h["particle_ids"])
    print("tracker_hits: measurements %d, merged (>1 particle) frac %.4f" % (
        len(p), pc.sum(pc.greater(pc.list_value_length(p), 1)).as_py() / len(p)))
fs = glob.glob(f"{R}/pfos/*.parquet")
if fs:
    f = pq.read_table(fs[0], columns=["pdg", "charge", "energy", "track_ids"])
    pdg = pc.list_flatten(f["pdg"]); q = pc.list_flatten(f["charge"])
    print("pfos: total %d, charged frac %.3f, pdg counts %s" % (
        len(pdg), pc.sum(pc.not_equal(q, 0)).as_py() / len(pdg),
        dict(zip(*[x.to_pylist() for x in pc.value_counts(pdg).flatten()]))))
