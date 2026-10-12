#!/usr/bin/env python3
"""Validate a particles_full table against Release 1 truth/particles and edm4hep.

  1. every v1 particle row exists in particles_full and every v1 column is identical (bit-exact; energy may differ
     by float rounding, reported separately);
  2. particles_full holds every MCParticle (count = edm4hep MCParticles size per event);
  3. the rows absent from v1 are exactly the generator history (generator_status not in 0, 1, 2);
  4. primary == not simulatorStatus bit 30; parent_id == first of parent_ids; links are symmetric.

Usage: check_particles_full_vs_v1.py <particles_full.parquet> <v1 particles parquet> <edm4hep.root> <global event offset>
       check_particles_full_vs_v1.py --published <v21 file> <v1 file> <runs dir with <run>/edm4hep.root> <run_size>
       (published mode: global event_ids; MCParticle counts read from the run that holds each event)
"""
import sys
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
import uproot

if sys.argv[1] == "--published":
    full_f, v1_f, runs_dir, run_size = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    full = pq.read_table(full_f).to_pydict()
    evs = full["event_id"]
    offset = 0
    v1 = pq.read_table(v1_f, filters=[("event_id", ">=", min(evs)), ("event_id", "<=", max(evs))]).to_pydict()
    v1row = {e: i for i, e in enumerate(v1["event_id"])}
    nmc = {}
    for run in sorted({e // run_size for e in evs}):
        counts = uproot.open(f"{runs_dir}/{run}/edm4hep.root")["events"]["MCParticles/MCParticles.PDG"].array(library="np")
        for pos, x in enumerate(counts):
            nmc[run * run_size + pos] = len(x)
else:
    full_f, v1_f, sim_f, offset = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    full = pq.read_table(full_f).to_pydict()
    nev = len(full["event_id"])
    v1 = pq.read_table(v1_f, filters=[("event_id", ">=", offset), ("event_id", "<", offset + nev)]).to_pydict()
    v1row = {e - offset: i for i, e in enumerate(v1["event_id"])}
    nmc = [len(x) for x in uproot.open(sim_f)["events"]["MCParticles/MCParticles.PDG"].array(entry_stop=nev, library="np")]
nev = len(full["event_id"])
V1COLS = [c for c in v1 if c not in ("event_id", "particle_id")]
bad = Counter(); extra_status = Counter(); n_v1 = n_full = 0
for r, e in enumerate(full["event_id"]):
    ids = np.asarray(full["particle_id"][r], np.int64)
    n_full += len(ids)
    if len(ids) != nmc[e] or not np.array_equal(ids, np.arange(nmc[e])):
        bad["count != MCParticles"] += 1
    j = v1row[e]
    vids = np.asarray(v1["particle_id"][j], np.int64)
    n_v1 += len(vids)
    gs = np.asarray(full["generator_status"][r]); ss = np.asarray(full["simulator_status"][r], np.int64)
    for c in V1COLS:
        a = np.asarray(full[c][r])[vids]; b = np.asarray(v1[c][j])
        if c in ("perigee_d0", "perigee_z0"):
            if not np.array_equal(a, b.astype(np.float32), equal_nan=True):
                bad[f"{c} DIFFERS"] += 1
        elif not np.array_equal(a, b):
            close = np.allclose(a.astype(float), b.astype(float), rtol=1e-6, atol=0)
            bad[f"{c} {'differs within 1e-6' if close else 'DIFFERS'}"] += 1
    extra = np.setdiff1d(np.arange(nmc[e]), vids)
    extra_status.update(gs[extra].tolist())
    if not np.array_equal(np.asarray(full["primary"][r]), ((ss >> 30) & 1) == 0):
        bad["primary != !bit30"] += 1
    pids = full["parent_ids"][r]; dids = full["daughter_ids"][r]
    if any((p[0] if p else -1) != q for p, q in zip(pids, full["parent_id"][r])):
        bad["parent_id != first parent"] += 1
    pairs_p = {(int(p), i) for i, ps in enumerate(pids) for p in ps}
    pairs_d = {(i, int(d)) for i, ds in enumerate(dids) for d in ds}
    if pairs_p != pairs_d:
        bad["parent/daughter links not symmetric"] += 1
print(f"events {nev}; particles_full rows {n_full}; v1 rows {n_v1}; extra rows {n_full - n_v1}")
print("generator_status of rows NOT in v1:", dict(sorted(extra_status.items())))
print("problems (events affected):", dict(bad) if bad else "none")
