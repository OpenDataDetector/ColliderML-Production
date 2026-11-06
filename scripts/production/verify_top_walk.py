"""Walk top -> W/b -> final state through truth/generator_particles + the published truth/particles
table of a v22 version, and check every final-state particle reached is a row of truth/particles with
the same pdg. Reads the per-run generator table and the published (packaged) particles table.

Usage: verify_top_walk.py <version_dir> <run> <run_size> <n_events>
Prints, per event, the last-copy tops with their daughters' |pdg| and the number of final-state
particles reached, then a summary; exit status 1 if any walk fails.
"""
import glob
import sys

import numpy as np
import pyarrow.parquet as pq

V, run, run_size, n_ev = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
gen = pq.read_table(glob.glob(f"{V}/runs/{run}/generator_particles/*.parquet")[0]).to_pylist()
lo = run * run_size
pfiles = sorted(glob.glob(f"{V}/parquet/truth/particles/*.parquet"))
cols = ["event_id", "particle_id", "pdg_id", "parent_ids", "daughter_ids"]
parts = {}
for f in pfiles:
    a, b = f.rsplit(".events", 1)[1].split(".parquet")[0].split("-")
    if int(b) < lo or int(a) >= lo + n_ev:
        continue
    names = pq.read_schema(f).names
    for r in pq.read_table(f, columns=[c for c in cols if c in names],
                           filters=[("event_id", ">=", lo), ("event_id", "<", lo + n_ev)]).to_pylist():
        parts[r["event_id"] - lo] = r

ok = bad = 0
for e in range(n_ev):
    g, p = gen[e], parts.get(e)
    if p is None:
        print(f"event {e}: no particles row"); bad += 1; continue
    pdg = dict(zip(g["particle_id"], g["pdg_id"])); dau = dict(zip(g["particle_id"], g["daughter_ids"]))
    ppdg = dict(zip(p["particle_id"], p["pdg_id"]))
    if "daughter_ids" in p:                       # v21 full table: links for every particle
        dau.update({i: d for i, d in zip(p["particle_id"], p["daughter_ids"]) if i not in dau})
    tops = [i for i, x in pdg.items() if abs(x) == 6 and not any(abs(pdg.get(d, 0)) == 6 for d in dau[i])]
    res = []
    for t in tops:
        stack, final, seen = list(dau[t]), set(), set()
        while stack:
            i = stack.pop()
            if i in seen:
                continue
            seen.add(i)
            if i in ppdg and i not in pdg and not dau.get(i):
                final.add(i)                       # a particles-table row the generator record ends on
            stack += [d for d in dau.get(i, []) if d in pdg or d in ppdg]
            if i in ppdg and i not in pdg and dau.get(i) is None:
                final.add(i)
        reached = [i for i in seen if i in ppdg]
        good = sorted({abs(pdg[d]) for d in dau[t]}) == [5, 24] and len(reached) > 0 \
            and all(ppdg[i] == pdg.get(i, ppdg[i]) for i in reached)
        res.append((sorted({abs(pdg[d]) for d in dau[t]}), len(reached), good))
        ok += good; bad += not good
    if e < 3:
        print(f"event {e}: last-copy tops {len(tops)}; (daughter |pdg|, particles-table rows reached, ok): {res}")
print(f"walks OK {ok}, failed {bad} over {n_ev} events")
sys.exit(1 if bad else 0)
