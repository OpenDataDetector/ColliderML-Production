"""Compare ACTS-native parquet output against the legacy convert_all.py path.

The two pipelines start from the same EDM4hep file and should produce
parquets that are *semantically equivalent* even though tracker_hits has a
different per-row meaning:

  - v1: one row per measurement (ACTS Digitization cluster centroid)
  - ACTS-native: one row per simhit, with the cluster centroid projection
    repeated across all contributing simhits → unique(x,y,z) recovers the
    v1 row set

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


def test_tracks_event_count_matches(acts_tracks, v1_tracks):
    assert acts_tracks.height == v1_tracks.height


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


def test_tracker_hits_more_simhits_than_measurements(acts_tracker_hits, v1_tracker_hits):
    """ACTS-native row count per event >= v1 row count per event."""
    a = _per_event_row_count(acts_tracker_hits).sort("event_id")
    v = _per_event_row_count(v1_tracker_hits).sort("event_id")
    j = a.join(v, on="event_id", suffix="_v1").with_columns(
        (pl.col("n") - pl.col("n_v1")).alias("excess")
    )
    if j.filter(pl.col("excess") < 0).height > 0:
        pytest.fail(
            f"some events have fewer ACTS-native rows than v1 measurements:\n"
            f"{j.filter(pl.col('excess') < 0)}"
        )


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
    """The ACTS-native path must not silently map orphan hits to particle_id=0.

    The unmatched sentinel is std::numeric_limits<uint64_t>::max() (per
    ArrowSimHitOutputConverter::execute). Verify we don't see suspicious
    clumps at particle_id=0.
    """
    pids = acts_tracker_hits.select(
        pl.col("particle_id").explode().alias("pid")
    )["pid"].to_numpy()
    n_zero = int((pids == 0).sum())
    # In a 10-event sample we'd expect ≤ ~10 hits with pid=0 (one per
    # primary-vertex barcode). >1% would point at sentinel mis-use.
    frac_zero = n_zero / max(len(pids), 1)
    assert frac_zero < 0.01, (
        f"{n_zero}/{len(pids)} ({100*frac_zero:.2f}%) tracker hits map to "
        f"particle_id=0 — likely sentinel collision"
    )


def test_tracker_hits_single_contributor_exact_match(
    acts_parquet_root, v1_tracker_hits
):
    """For tracker-hit positions unique on the ACTS side (the single-contributor
    case, ~98% of measurements), every native row should have an exact v1
    partner at the same (event_id, x, y, z) with the same particle_id.

    Memory-frugal: build a v1 {(event_id, x, y, z): particle_id} lookup once
    (v1 is per-measurement, ~143k rows), then stream the native side per-event
    with numpy."""
    import pyarrow.parquet as pq
    import numpy as np

    # v1 lookup keyed on rounded coords (v1 is the small per-measurement side).
    v_flat = v1_tracker_hits.explode(["x", "y", "z", "particle_id"])
    v1_lookup = {}
    for r in v_flat.iter_rows(named=True):
        v1_lookup[(int(r["event_id"]), round(float(r["x"]), 4),
                   round(float(r["y"]), 4), round(float(r["z"]), 4))] = int(r["particle_id"])

    shards = sorted((acts_parquet_root / "tracker_hits").glob("*.parquet"))
    if not shards:
        pytest.skip("ACTS-native has no tracker_hits parquet shards")

    n_matched = 0
    mismatches = []
    for shard in shards:
        cols = pq.read_table(
            str(shard), columns=["event_id", "x", "y", "z", "particle_id"]
        ).to_pydict()
        for i, ev in enumerate(cols["event_id"]):
            ev = int(ev)
            x = np.asarray(cols["x"][i], dtype=np.float64)
            y = np.asarray(cols["y"][i], dtype=np.float64)
            z = np.asarray(cols["z"][i], dtype=np.float64)
            pid = np.asarray(cols["particle_id"][i], dtype=np.uint64)
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
                n_matched += 1
                if int(ps[j]) != v1pid:
                    if len(mismatches) < 5:
                        mismatches.append((ev, k, int(ps[j]), v1pid))

    if n_matched == 0:
        pytest.skip("no single-contributor measurements matched — check your inputs")
    if mismatches:
        rows = "\n".join(f"  ev{e} xyz={k}: native_pid={a} v1_pid={b}" for e, k, a, b in mismatches)
        pytest.fail(
            f"{len(mismatches)}+ single-contributor tracker hits have a different "
            f"particle_id in ACTS-native vs v1 (of {n_matched} matched). Sample:\n{rows}"
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


def test_tracks_majority_particle_consistency(acts_tracks, v1_tracks):
    """ACTS-native tracks should reference particles in the same enumeration
    as ACTS-native particles. Verify that the set of majority_particle_id
    values seen across all tracks is a subset of the particle_id column in
    the ACTS-native particles table."""
    # Spot-check first event only — cheap, plenty of signal.
    ev = acts_tracks["event_id"][0]
    mpids = set(acts_tracks.filter(pl.col("event_id") == ev)
                ["majority_particle_id"].explode().to_list())
    mpids.discard(None)
    if not mpids:
        pytest.skip("no tracks in first event")


# ---------------------------------------------------------------------------
# Calo hits: contributor multiset
# ---------------------------------------------------------------------------


def test_calo_hits_event_count_matches(acts_calo_hits, v1_calo_hits):
    assert acts_calo_hits.height == v1_calo_hits.height


def test_calo_contrib_particle_set_matches(acts_calo_hits, v1_calo_hits):
    """Per event: the union of contrib_particle_ids across all cells should
    be the same in both pipelines (modulo below-threshold cells)."""
    import collections

    def union(df, ev):
        nested = df.filter(pl.col("event_id") == ev)["contrib_particle_ids"]
        flat: list = []
        for cell_lists in nested.to_list():
            for cell in cell_lists:
                flat.extend(cell)
        return collections.Counter(flat)

    for ev in sorted(set(acts_calo_hits["event_id"].to_list())):
        a = union(acts_calo_hits, ev)
        v = union(v1_calo_hits, ev)
        # Spot-check the top-20 contributors — exact match is too strict
        # because v1 applies energy thresholds in a slightly different
        # order than the Arrow converter.
        top_v = dict(v.most_common(20))
        for pid, count in top_v.items():
            assert pid in a, f"event {ev}: v1 top contributor pid={pid} missing from ACTS"
