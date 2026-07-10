"""Compare ACTS-native parquet output against the legacy convert_all.py path.

The two pipelines start from the same EDM4hep file and should produce
parquets that are *semantically equivalent*. Row semantics (Release-2 layout):

  - v1 tracker_hits: one row per measurement (ACTS Digitization cluster
    centroid), flat layout
  - ACTS-native tracker_hits: also one row per MEASUREMENT (the Release-2
    schema; the earlier one-row-per-simhit layout moved to the separate
    tracker_simhits table), event-nested layout

See ``tests/regression/README.md`` for how to populate the two parquet
trees this suite reads.
"""

from __future__ import annotations

import polars as pl
import pytest

# Tolerances are deliberately generous; ACTS's float32 → mm conversion and
# polars' fast-math comparisons can show ~0.5 ULP at scale.
FLOAT_ABS_TOL = 5.0e-3   # mm — looser than float32 ULP at 1m scale, well below
                         # any physically meaningful threshold
FLOAT_REL_TOL = 1.0e-5


# ---------------------------------------------------------------------------
# Schema / row-count invariants
# ---------------------------------------------------------------------------


def _per_event_row_count(df: pl.DataFrame) -> pl.DataFrame:
    """For a per-event nested-layout table, get per-event list lengths.

    Uses the first list column as the reference (every list column for a
    given event has the same length by construction).
    """
    # Pick any list column and use its per-row length.
    list_cols = [c for c, dt in df.schema.items() if isinstance(dt, pl.List)]
    assert list_cols, "expected at least one list column in nested-layout table"
    ref = list_cols[0]
    return df.select("event_id", pl.col(ref).list.len().alias("n"))


def test_particles_event_count_matches(acts_particles, v1_particles):
    assert acts_particles.height == v1_particles.height, (
        f"particles: ACTS-native {acts_particles.height} events vs "
        f"v1 {v1_particles.height} events"
    )


def test_tracker_hits_event_count_matches(acts_tracker_hits, v1_tracker_hits):
    assert acts_tracker_hits.height == v1_tracker_hits.height


def test_tracks_event_sets_consistent(acts_tracks, v1_tracks):
    """convert_all drops events with zero reconstructed tracks; the native writer
    keeps them as empty event-rows. So v1's event set must be a SUBSET of native's,
    and every native-only event must carry zero tracks (content is unchanged; only
    empty-event handling differs — downstream consumers must not assume event_id
    density is identical between the two)."""
    a_ev = set(acts_tracks["event_id"].to_list())
    v_ev = set(v1_tracks["event_id"].to_list())
    assert v_ev <= a_ev, f"v1 has track-events absent from native: {sorted(v_ev - a_ev)[:10]}"
    only = sorted(a_ev - v_ev)
    if only:
        n = _per_event_row_count(acts_tracks.filter(pl.col("event_id").is_in(only)))
        nonempty = n.filter(pl.col("n") > 0)
        assert nonempty.height == 0, (
            f"native-only events must be empty (0 tracks) but some have tracks:\n{nonempty}")


# ---------------------------------------------------------------------------
# Particles: per-event count must match exactly. Kinematics are identical
# at f32 tolerance because both paths start from the same MCParticle set.
# ---------------------------------------------------------------------------


def test_particles_per_event_count(acts_particles, v1_particles):
    acts_n = _per_event_row_count(acts_particles).sort("event_id")
    v1_n = _per_event_row_count(v1_particles).sort("event_id")
    diffs = acts_n.join(v1_n, on="event_id", suffix="_v1").with_columns(
        (pl.col("n") - pl.col("n_v1")).alias("delta")
    )
    bad = diffs.filter(pl.col("delta") != 0)
    if bad.height > 0:
        pytest.fail(
            f"particles per-event row count differs:\n{bad}\n"
            "ACTS-native and v1 should converge to the same MCParticle set."
        )


def test_particles_pdg_multiset(acts_particles, v1_particles):
    """The (event_id, pdg_id) multiset should be identical."""
    import collections

    def explode_pdg(df, col_candidates=("pdg_id", "PDG")):
        col = next((c for c in col_candidates if c in df.columns), None)
        if col is None:
            pytest.skip(f"no pdg column among {col_candidates}")
        return df.select("event_id", pl.col(col).alias("pdg_id")).explode("pdg_id")

    a = explode_pdg(acts_particles)
    v = explode_pdg(v1_particles)
    for ev in sorted(set(a["event_id"].to_list())):
        a_pdg = collections.Counter(
            a.filter(pl.col("event_id") == ev)["pdg_id"].to_list()
        )
        v_pdg = collections.Counter(
            v.filter(pl.col("event_id") == ev)["pdg_id"].to_list()
        )
        if a_pdg != v_pdg:
            diff = {k: (a_pdg.get(k, 0), v_pdg.get(k, 0))
                    for k in (a_pdg.keys() | v_pdg.keys())
                    if a_pdg.get(k, 0) != v_pdg.get(k, 0)}
            pytest.fail(
                f"event {ev}: pdg multiset differs (ACTS, v1):\n"
                f"{dict(sorted(diff.items())[:10])}"
            )


# ---------------------------------------------------------------------------
# Tracker hits: ACTS-native is per-simhit, v1 is per-measurement.
# Verify the dedup invariant + single-contributor exact match.
# ---------------------------------------------------------------------------


def test_tracker_hits_fewer_measurements_than_v1_simhit_rows(acts_tracker_hits, v1_tracker_hits):
    """Native rows are now MEASUREMENTS (Release-2 schema), v1 rows are
    simhit-rows: per event, native <= v1 (merging can only reduce)."""
    a = _per_event_row_count(acts_tracker_hits).sort("event_id")
    v = _per_event_row_count(v1_tracker_hits).sort("event_id")
    j = a.join(v, on="event_id", suffix="_v1").with_columns(
        (pl.col("n_v1") - pl.col("n")).alias("excess")
    )
    if j.filter(pl.col("excess") < 0).height > 0:
        pytest.fail(
            f"some events have MORE native measurement rows than v1 simhit rows:\n"
            f"{j.filter(pl.col('excess') < 0)}"
        )


@pytest.mark.skip(reason="superseded: native rows ARE measurements now; "
                  "the dedup-vs-measurement check is tautological. "
                  "Cross-pipeline count covered by "
                  "test_tracker_hits_fewer_measurements_than_v1_simhit_rows.")
def test_tracker_hits_dedup_matches_measurements(acts_parquet_root, v1_tracker_hits):
    """After unique(['x','y','z']) on ACTS-native, per-event row count should
    match v1's measurement count. On a same-seed run the digitization is
    identical, so the delta should be ~0; we keep a small tolerance for the
    handful of below-digi-threshold simhits (gx=NaN).

    Memory-frugal: the native tracker_hits frame is read column-pruned and
    processed per-event with numpy, never materializing the full exploded
    140k+ row table (that exploded the 7.6 GB host)."""
    import pyarrow.parquet as pq
    import numpy as np

    # Same-seed: expect exact. Tolerance covers below-threshold simhits only.
    TOLERATED_DELTA_PER_EVENT = 50

    v = _per_event_row_count(v1_tracker_hits)
    v1_by_event = {int(r["event_id"]): int(r["n"]) for r in v.iter_rows(named=True)}

    shards = sorted((acts_parquet_root / "tracker_hits").glob("*.parquet"))
    if not shards:
        pytest.skip("ACTS-native has no tracker_hits parquet shards")

    bad = []
    for shard in shards:
        cols = pq.read_table(str(shard), columns=["event_id", "x", "y", "z"]).to_pydict()
        for i, ev in enumerate(cols["event_id"]):
            ev = int(ev)
            x = np.asarray(cols["x"][i], dtype=np.float64)
            y = np.asarray(cols["y"][i], dtype=np.float64)
            z = np.asarray(cols["z"][i], dtype=np.float64)
            finite = ~np.isnan(x)
            key = np.stack(
                [np.round(x[finite], 4), np.round(y[finite], 4), np.round(z[finite], 4)],
                axis=1,
            )
            n_uniq = len(np.unique(key, axis=0)) if len(key) else 0
            v1n = v1_by_event.get(ev, 0)
            if abs(v1n - n_uniq) > TOLERATED_DELTA_PER_EVENT:
                bad.append((ev, v1n, n_uniq, v1n - n_uniq))
    if bad:
        rows = "\n".join(f"  ev{e}: v1={a} native_uniq={u} delta={d:+d}" for e, a, u, d in bad)
        pytest.fail(
            f"native unique(x,y,z) differs from v1 measurement count by more than "
            f"{TOLERATED_DELTA_PER_EVENT} in some events:\n{rows}"
        )


def test_tracker_hits_particle_id_no_silent_sentinel(acts_tracker_hits):
    """Orphan hits must not silently leak the unmatched sentinel
    (std::numeric_limits<uint64_t>::max(), per ArrowSimHitOutputConverter::execute)
    into the particle_ids column.

    NB: particle_id == 0 is a VALID barcode — the first primary. In a single-muon
    event the muon *is* particle_id 0 and legitimately owns ~90% of the hits, so
    checking pid==0 (as this test used to) false-positives. We check the actual
    sentinel value instead (uint64 max, or -1 if the column is stored signed)."""
    import numpy as np
    pids = acts_tracker_hits.select(
        pl.col("particle_ids").explode().explode().alias("pid")
    )["pid"].drop_nulls().to_numpy()
    # dtype-safe: whether the column is stored uint64 or int64 (-1), casting to
    # uint64 maps the sentinel to 2**64-1 either way. (Comparing an unsigned
    # array against python -1 is numpy-version-dependent under NEP 50.)
    sentinel = pids.astype(np.uint64) == np.uint64(2**64 - 1)
    n_sent = int(sentinel.sum())
    frac = n_sent / max(len(pids), 1)
    assert frac < 0.01, (
        f"{n_sent}/{len(pids)} ({100*frac:.2f}%) tracker hits carry the unmatched "
        f"uint64-max sentinel — orphan hits leaking into particle_ids")


@pytest.mark.skip(reason="needs rework for the Release-2 nested truth links; "
                  "in-table particle consistency is covered by test_tracker_tables.py")
def test_tracker_hits_single_contributor_same_particle(
    acts_parquet_root, v1_tracker_hits, pid_bijection
):
    """For tracker-hit positions unique on the ACTS side (single-contributor,
    ~98% of measurements), every native row must reference the SAME PHYSICAL
    PARTICLE as its v1 partner at the same (event_id, x, y, z).

    The raw particle_id integers differ between pipelines (different particle
    enumeration orders), so we compare through `pid_bijection`
    (native_pid -> v1_pid, matched on kinematics). native_pid mapped through
    the bijection must equal the v1 hit's particle_id.

    Memory-frugal: small v1 lookup once, stream native per-event with numpy."""
    import pyarrow.parquet as pq
    import numpy as np

    v_flat = v1_tracker_hits.explode(["x", "y", "z", "particle_id"])
    v1_lookup = {}
    for r in v_flat.iter_rows(named=True):
        v1_lookup[(int(r["event_id"]), round(float(r["x"]), 4),
                   round(float(r["y"]), 4), round(float(r["z"]), 4))] = int(r["particle_id"])

    shards = sorted((acts_parquet_root / "tracker_hits").glob("*.parquet"))
    if not shards:
        pytest.skip("ACTS-native has no tracker_hits parquet shards")

    n_matched = 0      # positions present + singleton on both sides, pid in bijection
    n_same = 0         # native particle maps to the v1 particle
    unmapped = 0       # native pid not in the kinematic bijection (ambiguous kin)
    mismatches = []
    for shard in shards:
        cols = pq.read_table(
            str(shard), columns=["event_id", "x", "y", "z", "particle_id"]
        ).to_pydict()
        for i, ev in enumerate(cols["event_id"]):
            ev = int(ev)
            bij = pid_bijection.get(ev, {})
            x = np.asarray(cols["x"][i], dtype=np.float64)
            y = np.asarray(cols["y"][i], dtype=np.float64)
            z = np.asarray(cols["z"][i], dtype=np.float64)
            pid = np.asarray(cols["particle_id"][i], dtype=np.int64)
            finite = ~np.isnan(x)
            xs, ys, zs, ps = x[finite], y[finite], z[finite], pid[finite]
            key = np.stack([np.round(xs, 4), np.round(ys, 4), np.round(zs, 4)], axis=1)
            uniq, counts = np.unique(key, axis=0, return_counts=True)
            singleton_set = {tuple(k) for k, c in zip(uniq.tolist(), counts.tolist()) if c == 1}
            for j in range(len(xs)):
                k = (round(float(xs[j]), 4), round(float(ys[j]), 4), round(float(zs[j]), 4))
                if k not in singleton_set:
                    continue
                v1pid = v1_lookup.get((ev,) + k)
                if v1pid is None:
                    continue
                mapped = bij.get(int(ps[j]))
                if mapped is None:
                    unmapped += 1
                    continue
                n_matched += 1
                if mapped == v1pid:
                    n_same += 1
                elif len(mismatches) < 5:
                    mismatches.append((ev, k, int(ps[j]), mapped, v1pid))

    if n_matched == 0:
        pytest.skip("no single-contributor measurements matched — check your inputs")
    # Demand essentially all matched hits reference the same physical particle.
    frac_same = n_same / n_matched
    if frac_same < 0.999:
        rows = "\n".join(
            f"  ev{e} xyz={k}: native_pid={a}→v1 {m}, but v1 hit says {b}"
            for e, k, a, m, b in mismatches
        )
        pytest.fail(
            f"only {n_same}/{n_matched} ({100*frac_same:.2f}%) single-contributor "
            f"hits reference the same physical particle ({unmapped} unmapped). "
            f"Sample mismatches:\n{rows}"
        )


# ---------------------------------------------------------------------------
# Tracks: per-event count + majority particle distribution
# ---------------------------------------------------------------------------


def test_tracks_per_event_count(acts_tracks, v1_tracks):
    a = _per_event_row_count(acts_tracks).sort("event_id")
    v = _per_event_row_count(v1_tracks).sort("event_id")
    j = a.join(v, on="event_id", suffix="_v1").with_columns(
        (pl.col("n") - pl.col("n_v1")).abs().alias("delta")
    )
    # Allow up to 2 tracks of slack per event (track finder/ambi may pick
    # tracks in a slightly different order; the totals should still agree).
    bad = j.filter(pl.col("delta") > 2)
    if bad.height > 0:
        pytest.fail(f"tracks per-event count diverges:\n{bad}")


def test_tracks_majority_particle_consistency(acts_tracks, acts_particles):
    """ACTS-native tracks must reference particles in the same enumeration as the
    ACTS-native particles table: every majority_particle_id must appear in that
    event's particle_id column. (This test used to compute the set and assert
    nothing — vacuous; it now checks the subset relation on the first few
    non-empty events.)"""
    checked = 0
    for i, ev in enumerate(acts_tracks["event_id"].to_list()):
        mpids = set(acts_tracks.filter(pl.col("event_id") == ev)
                    ["majority_particle_id"].explode().to_list())
        mpids.discard(None)
        if not mpids:
            continue
        pids = set(acts_particles.filter(pl.col("event_id") == ev)
                   ["particle_id"].explode().to_list())
        orphans = mpids - pids
        assert not orphans, (
            f"event {ev}: track majority_particle_id(s) {sorted(orphans)[:5]} "
            f"not present in the native particles table")
        checked += 1
        if checked >= 10:   # spot-check is plenty
            break
    if checked == 0:
        pytest.skip("no events with tracks")


def test_tracks_num_measurements_matches(acts_tracks, v1_tracks):
    """`num_measurements` (genuine merged-cluster count) must agree exactly
    between native and v1, per track. Native emits `track.nMeasurements()`;
    v1 carries the ROOT `nMeasurements` branch. Compared as the per-event
    multiset (track ordering may differ)."""
    import collections
    if "num_measurements" not in acts_tracks.columns:
        pytest.skip("native tracks lack num_measurements (rebuild the image)")
    if "num_measurements" not in v1_tracks.columns:
        pytest.skip("v1 tracks lack num_measurements")
    for ev in sorted(set(acts_tracks["event_id"].to_list())):
        a = collections.Counter(
            int(x) for x in acts_tracks.filter(pl.col("event_id") == ev)
            ["num_measurements"].explode().to_list())
        v = collections.Counter(
            int(x) for x in v1_tracks.filter(pl.col("event_id") == ev)
            ["num_measurements"].explode().to_list())
        assert a == v, (
            f"event {ev}: num_measurements multiset differs native vs v1; "
            f"symmetric diff sample: "
            f"{list((a - v).items())[:5]} | {list((v - a).items())[:5]}"
        )


def test_track_parameters_match_v1(acts_tracks, v1_tracks):
    """Fitted track parameters must agree VALUE-FOR-VALUE between native and v1,
    compared as per-event sorted multisets of (d0, z0, phi, theta, qop) tuples
    (particle-enumeration-independent; see comment below on why we can't match
    by majority_particle_id).

    This is the strongest back-compat guard: it checks the actual physics
    quantities downstream ML consumes, not just row counts or hit linkage.
    Same-seed digitization -> the Kalman fit is identical, so these are
    bit-identical in practice; the tolerance only guards f32/f64 storage drift.

    STRICT on event coverage: this suite's contract is same-seed (one
    digitization feeding both writers), so per-event track counts must be equal
    in every common event — a count mismatch here FAILS rather than skipping,
    closing the hole where test_tracks_per_event_count's ±2 slack plus a skip
    here would leave extra/missing tracks' parameters never compared."""
    pars = ["d0", "z0", "phi", "theta", "qop"]
    for c in pars:
        if c not in acts_tracks.columns or c not in v1_tracks.columns:
            pytest.skip(f"missing column {c} in one pipeline")
    # The two pipelines enumerate particles differently, so we CANNOT match tracks
    # by majority_particle_id. Instead compare the per-event MULTISET of parameter
    # tuples (order- and id-independent): same tracks -> identical sorted tuples.
    common = sorted(set(acts_tracks["event_id"].to_list())
                    & set(v1_tracks["event_id"].to_list()))
    import math
    worst = 0.0
    n_pairs = 0
    count_mismatch = []
    for ev in common:
        a = acts_tracks.filter(pl.col("event_id") == ev)
        v = v1_tracks.filter(pl.col("event_id") == ev)
        A = sorted(zip(*[[float(x) for x in a[p][0]] for p in pars]))
        V = sorted(zip(*[[float(x) for x in v[p][0]] for p in pars]))
        if len(A) != len(V):
            count_mismatch.append((ev, len(A), len(V)))
            continue
        for ta, tv in zip(A, V):
            n_pairs += 1
            for x, y in zip(ta, tv):
                d = abs(x - y)
                # NaN anywhere is a divergence, not something max() may swallow
                worst = math.inf if math.isnan(d) else max(worst, d)
    assert not count_mismatch, (
        f"{len(count_mismatch)} common events have differing track counts "
        f"(native vs v1) under the same-seed contract, so their parameters were "
        f"never compared: {count_mismatch[:5]}")
    assert n_pairs > 0, "no common events to compare track parameters"
    assert worst <= 1e-3, (
        f"track parameters (d0/z0/phi/theta/qop) diverge native vs v1: "
        f"max |diff| = {worst:.3e} over {n_pairs} matched tracks")


def test_tracks_hit_outlier_excluded_matches_v1(
    acts_tracks, v1_tracks, acts_tracker_hits, v1_tracker_hits
):
    """Filtering native `hit_ids` to `hit_outlier==False` and dereferencing to
    hit POSITIONS must, per track, equal v1's hit positions exactly.

    Position-based (not count-based) because native lists hits at sim-hit level
    — a merged cluster contributes multiple hit_ids at the *same* reco position
    — so only after dedup-by-position do the two pipelines line up. The outlier
    flag is what accounts for the rest of the difference: dropping outliers must
    reproduce v1's measurement-cluster set exactly.

    Parquets are event-nested (list-per-event); explode track-level columns to
    one row per track and match native↔v1 by (event_id, track_id)."""
    if "hit_outlier" not in acts_tracks.columns:
        pytest.skip("native tracks lack hit_outlier (rebuild the image)")

    # per-event hit-position lookup tables (row index -> (x,y,z))
    def pos_table(df):
        out = {}
        for r in df.iter_rows(named=True):
            out[int(r["event_id"])] = list(
                zip((round(float(v), 3) for v in r["x"]),
                    (round(float(v), 3) for v in r["y"]),
                    (round(float(v), 3) for v in r["z"])))
        return out
    nat_pos = pos_table(acts_tracker_hits)
    v1_pos = pos_table(v1_tracker_hits)

    at = acts_tracks.explode(["hit_ids", "hit_outlier", "track_id"])
    vt = v1_tracks.explode(["hit_ids", "track_id"])
    # v1 lookup: (event, track_id) -> set of hit positions
    v1_lut = {}
    for r in vt.iter_rows(named=True):
        ev = int(r["event_id"])
        v1_lut[(ev, int(r["track_id"]))] = {
            v1_pos[ev][int(i)] for i in r["hit_ids"]}

    mismatch = []
    checked = 0
    for r in at.iter_rows(named=True):
        ev = int(r["event_id"]); tid = int(r["track_id"])
        key = (ev, tid)
        if key not in v1_lut:
            continue
        nat_set = {nat_pos[ev][int(i)]
                   for i, o in zip(r["hit_ids"], r["hit_outlier"]) if not o}
        checked += 1
        if nat_set != v1_lut[key]:
            if len(mismatch) < 5:
                mismatch.append((ev, tid, len(nat_set), len(v1_lut[key]),
                                 len(nat_set & v1_lut[key])))
        # The v1 hit_ids come from the tracksummary measurementIDs branch - a
        # colliderml-fork ACTS patch that rebased builds (tracker-hits-v2) do
        # not carry. When v1 extracted nothing, the comparison is unavailable.
        if all(len(v) == 0 for v in v1_lut.values()):
            pytest.skip(
                "v1 tracks have no hit_ids (tracksummary measurementIDs branch "
                "absent in this ACTS build) - legacy comparison unavailable; "
                "native track->hit linkage is covered by test_tracker_tables.py"
            )
    assert checked > 0, "no native↔v1 tracks matched by (event, track_id)"
    if mismatch:
        rows = "\n".join(
            f"  ev{e} tk{t}: native_nonoutlier={a} v1={b} shared={s}"
            for e, t, a, b, s in mismatch)
        pytest.fail(
            f"{len(mismatch)}/{checked} tracks: non-outlier native hit positions "
            f"!= v1 hit positions:\n{rows}")


# ---------------------------------------------------------------------------
# Calo hits: contributor multiset
# ---------------------------------------------------------------------------


def test_calo_hits_event_count_matches(acts_calo_hits, v1_calo_hits):
    assert acts_calo_hits.height == v1_calo_hits.height


def test_calo_contrib_same_particles(acts_calo_hits, v1_calo_hits, pid_bijection):
    """Per event: the set of physical particles depositing in the calorimeter
    should agree between pipelines. Native contrib particle_ids are mapped
    through `pid_bijection` (native_pid -> v1_pid) before comparison, since the
    two pipelines use different particle enumerations. We check that the native
    contributor set (mapped to v1 ids) covers v1's top contributors."""
    import collections

    def union(df, ev):
        nested = df.filter(pl.col("event_id") == ev)["contrib_particle_ids"]
        flat = []
        for cell_lists in nested.to_list():
            for cell in cell_lists:
                flat.extend(int(p) for p in cell)
        return collections.Counter(flat)

    for ev in sorted(set(acts_calo_hits["event_id"].to_list())):
        bij = pid_bijection.get(ev, {})
        a_raw = union(acts_calo_hits, ev)
        # Map native contributor ids into v1 id space; drop unmapped (ambiguous
        # kinematic keys — a tiny fraction).
        a_mapped = {bij[p] for p in a_raw if p in bij}
        v = union(v1_calo_hits, ev)
        # v1's top-20 contributors (by deposit count) must all appear among the
        # native contributors once relabeled. Top contributors are high-energy
        # and well above any threshold-ordering ambiguity.
        missing = [pid for pid, _ in v.most_common(20) if pid not in a_mapped]
        assert not missing, (
            f"event {ev}: v1 top calo contributors {missing[:5]} not found among "
            f"native contributors (relabeled). native_uniq={len(a_mapped)} "
            f"v1_uniq={len(v)}"
        )
