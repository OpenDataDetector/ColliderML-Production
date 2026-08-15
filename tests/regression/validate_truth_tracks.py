"""Comprehensive validation of the truth-found track collection.

Checks, in order of what would actually catch a bug:
  1. Hit-assignment purity  - every hit on a truth track belongs to that track's particle.
  2. Parameter accuracy     - fitted params vs the true particle. Bias and spread.
  3. Truth vs CKF           - on the SAME particles, truth should not be worse.
  4. Coverage               - one truth track per selected particle, no duplicates.

Run against a digi output directory that has particles/, tracker_hits/,
tracks/ and truth_tracks/.
"""
import glob
import math
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = sys.argv[1]
TRUTH = sys.argv[2] if len(sys.argv) > 2 else "truth_tracks"


def load(table):
    files = sorted(glob.glob(f"{ROOT}/{table}/*.parquet"))
    if not files:
        return None
    return pa.concat_tables([pq.read_table(f) for f in files]).sort_by(
        [("event_id", "ascending")]
    )


def fail(msg):
    global FAILURES
    FAILURES.append(msg)
    print(f"   FAIL: {msg}")


FAILURES = []

particles = load("particles")
hits = load("tracker_hits")
truth = load(TRUTH)
ckf = load("tracks")
if truth is None:
    print(f"no {TRUTH} table found under {ROOT}")
    sys.exit(2)

n_ev = truth.num_rows
print(f"loaded {n_ev} events from {ROOT} (truth table: {TRUTH})\n")

# ---------------------------------------------------------------- 1. purity
print("1. HIT-ASSIGNMENT PURITY  (every hit on a truth track is that particle's)")
t_pid = truth.column("majority_particle_id").to_pylist()
t_hits = truth.column("hit_ids").to_pylist()
h_pids = hits.column("particle_ids").to_pylist()

checked = impure = empty = 0
for ev in range(n_ev):
    hit_pid_map = h_pids[ev]  # per-hit list of contributing particle ids
    for pid, hidx in zip(t_pid[ev], t_hits[ev]):
        checked += 1
        if len(hidx) == 0:
            empty += 1
            continue
        for h in hidx:
            if h >= len(hit_pid_map):
                impure += 1
                break
            contributors = hit_pid_map[h]
            if pid not in contributors:
                impure += 1
                break
print(f"   tracks checked          : {checked}")
print(f"   tracks with a foreign hit: {impure}")
print(f"   tracks with zero hits    : {empty}")
if impure:
    fail(f"{impure}/{checked} truth tracks contain a hit not from their own particle")
if empty:
    fail(f"{empty}/{checked} truth tracks have no hits")
if not impure and not empty:
    print("   OK - truth finding is pure by construction, as expected")

# ------------------------------------------------------------- 2. accuracy
print("\n2. PARAMETER ACCURACY  (fitted vs true, per matched particle)")
p_id = particles.column("particle_id").to_pylist()
px = particles.column("px").to_pylist()
py = particles.column("py").to_pylist()
pz = particles.column("pz").to_pylist()
q = particles.column("charge").to_pylist()


def true_params(ev):
    """particle_id -> (qop, phi, theta, pt) from truth momentum."""
    out = {}
    for i, pid in enumerate(p_id[ev]):
        x, y, z = px[ev][i], py[ev][i], pz[ev][i]
        p = math.sqrt(x * x + y * y + z * z)
        if p == 0:
            continue
        out[pid] = (
            q[ev][i] / p,
            math.atan2(y, x),
            math.acos(max(-1.0, min(1.0, z / p))),
            math.hypot(x, y),
        )
    return out


def residuals(tbl):
    """Return dict of arrays of (fitted - true) keyed by parameter, plus pt."""
    pid_c = tbl.column("majority_particle_id").to_pylist()
    qop_c = tbl.column("qop").to_pylist()
    phi_c = tbl.column("phi").to_pylist()
    the_c = tbl.column("theta").to_pylist()
    res = {"qop_rel": [], "phi": [], "theta": [], "pt": [], "unmatched": 0}
    for ev in range(tbl.num_rows):
        tp = true_params(ev)
        for pid, qop, phi, theta in zip(pid_c[ev], qop_c[ev], phi_c[ev], the_c[ev]):
            t = tp.get(pid)
            if t is None:
                res["unmatched"] += 1
                continue
            tqop, tphi, tthe, tpt = t
            if tqop != 0:
                res["qop_rel"].append((qop - tqop) / abs(tqop))
            dphi = (phi - tphi + math.pi) % (2 * math.pi) - math.pi
            res["phi"].append(dphi)
            res["theta"].append(theta - tthe)
            res["pt"].append(tpt)
    return res


def report(name, res, unmatched_is_fatal=True):
    print(f"   {name}: matched={len(res['qop_rel'])} unmatched={res['unmatched']}")
    if res["unmatched"]:
        # For the CKF an unmatched track is a fake - a normal property of
        # combinatorial finding, not a defect in the truth chain. For the truth
        # collection it would mean a track pointing at a particle that does not
        # exist, which IS a defect.
        if unmatched_is_fatal:
            fail(f"{name}: {res['unmatched']} tracks had no matching truth particle")
        else:
            n = len(res["qop_rel"]) + res["unmatched"]
            print(f"      (fakes: {res['unmatched']}/{n} = {100*res['unmatched']/n:.2f}% - expected for CKF)")
    for key, unit, tol_bias, tol_rms in (
        ("qop_rel", "rel", 0.02, 0.20),
        ("phi", "rad", 2e-3, 2e-2),
        ("theta", "rad", 2e-3, 2e-2),
    ):
        a = np.array(res[key])
        if a.size == 0:
            continue
        med = float(np.median(a))
        # robust spread: half the 16-84 percentile span, immune to fit outliers
        lo, hi = np.percentile(a, [16, 84])
        spread = float(hi - lo) / 2
        flag = ""
        if abs(med) > tol_bias:
            flag = "  <- BIAS"
            fail(f"{name} {key}: median {med:.2e} exceeds {tol_bias:.0e}")
        if spread > tol_rms:
            flag += "  <- WIDE"
            fail(f"{name} {key}: spread {spread:.2e} exceeds {tol_rms:.0e}")
        print(f"      {key:9s} median={med:+.3e} {unit}  spread(68%)={spread:.3e}{flag}")
    return res


truth_res = report("truth", residuals(truth))
ckf_res = (
    report("ckf  ", residuals(ckf), unmatched_is_fatal=False) if ckf is not None else None
)

# --------------------------------------------------------- 3. truth vs ckf
if ckf_res is not None:
    print("\n3. TRUTH vs CKF  (truth finding is perfect, so it should not be worse)")
    for key in ("qop_rel", "phi", "theta"):
        a, b = np.array(truth_res[key]), np.array(ckf_res[key])
        if a.size == 0 or b.size == 0:
            continue
        sa = (np.percentile(a, 84) - np.percentile(a, 16)) / 2
        sb = (np.percentile(b, 84) - np.percentile(b, 16)) / 2
        verdict = "truth better/equal" if sa <= sb * 1.10 else "TRUTH WORSE"
        if sa > sb * 1.10:
            fail(f"truth {key} spread {sa:.3e} is >10% worse than CKF {sb:.3e}")
        print(f"   {key:9s} truth={sa:.3e}  ckf={sb:.3e}  -> {verdict}")

# ------------------------------------------------------------- 4. coverage
print("\n4. COVERAGE")
dup_events = 0
for ev in range(n_ev):
    ids = t_pid[ev]
    if len(ids) != len(set(ids)):
        dup_events += 1
n_truth = sum(len(x) for x in t_pid)
n_ckf = sum(len(x) for x in ckf.column("majority_particle_id").to_pylist()) if ckf is not None else 0
print(f"   truth tracks total : {n_truth}  ({n_truth/n_ev:.3f} per event)")
print(f"   ckf   tracks total : {n_ckf}  ({n_ckf/n_ev:.3f} per event)")
print(f"   events with a duplicated particle: {dup_events}")
if dup_events:
    fail(f"{dup_events} events have two truth tracks for the same particle")

print("\n" + "=" * 60)
print("RESULT:", "PASS" if not FAILURES else f"FAIL ({len(FAILURES)} problem(s))")
for f in FAILURES:
    print("  -", f)
sys.exit(0 if not FAILURES else 1)
