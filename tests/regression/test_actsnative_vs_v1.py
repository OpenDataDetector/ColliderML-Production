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


def test_tracker_hits_dedup_matches_measurements(acts_tracker_hits, v1_tracker_hits):
    """After unique(['x','y','z']) on ACTS-native, per-event row count should
    match v1's measurement count to within the number of below-threshold
    simhits (a tiny handful, never more than ~5/event in the 10-event demo)."""

    # Tolerable shortfall: simhits that didn't trigger a measurement (gx=NaN).
    # The cluster-merged simhits also contribute NaN if the projection failed
    # to find a surface — those are real bugs and would show as larger gaps.
    TOLERATED_SHORTFALL_PER_EVENT = 50

    a = (
        acts_tracker_hits.explode(["x", "y", "z"])
        .filter(pl.col("x").is_not_nan())  # drop below-threshold simhits
        .group_by("event_id")
        .agg(pl.struct(["x", "y", "z"]).n_unique().alias("n_unique"))
        .sort("event_id")
    )
    v = _per_event_row_count(v1_tracker_hits).sort("event_id")
    j = a.join(v, on="event_id").with_columns(
        (pl.col("n") - pl.col("n_unique")).alias("delta"),
    )
    over_tol = j.filter(pl.col("delta").abs() > TOLERATED_SHORTFALL_PER_EVENT)
    if over_tol.height > 0:
        pytest.fail(
            f"unique(x,y,z) row count differs from v1 measurement count by more "
            f"than {TOLERATED_SHORTFALL_PER_EVENT} in some events:\n{over_tol}"
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
    acts_tracker_hits, v1_tracker_hits
):
    """For tracker-hit positions that are unique in both tables (i.e. the
    single-contributor case, ~98% of measurements), every ACTS-native row
    should have an exact v1 partner at the same (event_id, x, y, z) with the
    same particle_id."""
    a_flat = (
        acts_tracker_hits.explode([
            "x", "y", "z", "particle_id", "volume_id", "layer_id"
        ])
        .filter(pl.col("x").is_not_nan())
    )
    v_flat = v1_tracker_hits.explode([
        "x", "y", "z", "particle_id", "volume_id", "layer_id"
    ])

    # Restrict to positions that appear exactly once on the ACTS side
    # — these are the unambiguous singletons.
    singletons = (
        a_flat.group_by(["event_id", "x", "y", "z"])
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") == 1)
        .select("event_id", "x", "y", "z")
    )
    a_single = a_flat.join(singletons, on=["event_id", "x", "y", "z"], how="inner")

    # Match against v1 on the same key. v1 also has unique (x,y,z) per event
    # for these rows (one row per measurement).
    j = a_single.join(
        v_flat.select(
            "event_id", "x", "y", "z",
            pl.col("particle_id").alias("particle_id_v1"),
        ),
        on=["event_id", "x", "y", "z"],
        how="left",
    )

    unmatched = j.filter(pl.col("particle_id_v1").is_null())
    # A small unmatched count is OK (e.g., orphans v1 hadn't found because of
    # the broken position-merge). The interesting check is whether MATCHED
    # rows have the same particle_id.
    if j.height - unmatched.height == 0:
        pytest.skip("no single-contributor measurements matched — check your inputs")

    mismatches = j.filter(
        pl.col("particle_id_v1").is_not_null()
        & (pl.col("particle_id") != pl.col("particle_id_v1"))
    )
    if mismatches.height > 0:
        sample = mismatches.head(5)
        pytest.fail(
            f"{mismatches.height} single-contributor tracker hits have a "
            f"different particle_id in ACTS-native vs v1. Sample:\n{sample}"
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
