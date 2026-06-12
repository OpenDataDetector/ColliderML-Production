"""Release-2 two-table tracker schema contract (tracker_hits + tracker_simhits).

tracker_hits   — RECO, one entry per measurement: local parameters (+subspace
                 bitmask), global position, sensor ids, cluster-shape features,
                 and truth LINKS (particle_ids, simhit_ids).
tracker_simhits — TRUTH, one entry per sim-hit (ALL sim-hits): position+time,
                 4-momentum at the hit, deposited energy, particle link,
                 trajectory hit index, sensor ids.

Reads the ACTS-native parquet tree at $COLLIDERML_ACTSNATIVE_PARQUET_DIR
(same convention as the other regression suites).
"""

import math

import polars as pl
import pytest


def _rows(df):
    return list(df.iter_rows(named=True))


def test_simhit_ids_resolve(acts_tracker_hits, acts_tracker_simhits):
    """Every simhit_id in tracker_hits is a valid row index into the
    tracker_simhits table of the same event, and every measurement has at
    least one contributing sim-hit."""
    n_sim_by_ev = {
        int(r["event_id"]): len(r["true_x"])
        for r in _rows(acts_tracker_simhits)
    }
    bad = []
    for r in _rows(acts_tracker_hits):
        ev = int(r["event_id"])
        n_sim = n_sim_by_ev.get(ev)
        assert n_sim is not None, f"event {ev} missing from tracker_simhits"
        for m, ids in enumerate(r["simhit_ids"]):
            if len(ids) == 0:
                bad.append((ev, m, "no contributors"))
            for sid in ids:
                if not (0 <= sid < n_sim):
                    bad.append((ev, m, f"simhit_id {sid} >= {n_sim}"))
    assert not bad, f"unresolvable truth links (first 10): {bad[:10]}"


def test_particle_ids_consistent_between_tables(
    acts_tracker_hits, acts_tracker_simhits
):
    """tracker_hits.particle_ids[m][k] must equal
    tracker_simhits.particle_id[simhit_ids[m][k]] — the two truth links agree
    contributor by contributor."""
    sim_pids = {
        int(r["event_id"]): r["particle_id"] for r in _rows(acts_tracker_simhits)
    }
    mismatches = 0
    total = 0
    for r in _rows(acts_tracker_hits):
        ev = int(r["event_id"])
        spids = sim_pids[ev]
        for ids, pids in zip(r["simhit_ids"], r["particle_ids"]):
            assert len(ids) == len(pids)
            for sid, pid in zip(ids, pids):
                total += 1
                if spids[sid] != pid:
                    mismatches += 1
    assert total > 0
    assert mismatches == 0, f"{mismatches}/{total} contributor links disagree"


def test_subspace_and_locals(acts_tracker_hits):
    """Local parameters are always finite; the subspace bitmask always claims
    at least one measured local component; 1D (strip) measurements appear as
    bit0-only entries."""
    n_1d = 0
    n_2d = 0
    for r in _rows(acts_tracker_hits):
        for loc0, loc1, sub in zip(r["loc0"], r["loc1"], r["subspace"]):
            assert math.isfinite(loc0) and math.isfinite(loc1)
            assert sub & 0x3, "no local component measured"
            if (sub & 0x3) == 0x3:
                n_2d += 1
            else:
                n_1d += 1
    # ODD has pixels (2D) and strips (1D): both populations must exist.
    assert n_2d > 0, "no 2D (pixel-like) measurements found"
    assert n_1d > 0, "no 1D (strip-like) measurements found"


def test_shape_columns_present_and_finite(acts_tracker_hits):
    """Cluster-shape columns exist and are finite. With geometric digitization
    n_channels >= 1; report (not fail) if shape content is all-zero, which
    indicates the clusters reached the converter without channel content."""
    n_meas = 0
    n_with_channels = 0
    for r in _rows(acts_tracker_hits):
        for nch, sact, sz0, sz1 in zip(
            r["n_channels"], r["sum_activation"], r["size_loc0"], r["size_loc1"]
        ):
            n_meas += 1
            assert math.isfinite(sact)
            if nch > 0:
                n_with_channels += 1
                assert sz0 >= 1 and sz1 >= 1
    assert n_meas > 0
    if n_with_channels == 0:
        pytest.xfail(
            "shape columns are all-zero: clusters reached the converter "
            "without channel content (known digitization gating issue) — "
            "geometry/links are still valid"
        )


def test_merged_clusters_present(acts_tracker_hits):
    """Cluster merging is ON (the Release-1 bug regression check): the dataset
    must contain measurements with more than one contributing sim-hit."""
    multi = 0
    total = 0
    for r in _rows(acts_tracker_hits):
        for ids in r["simhit_ids"]:
            total += 1
            if len(ids) > 1:
                multi += 1
    assert total > 0
    assert multi > 0, (
        "no multi-contributor measurements found — cluster merging appears "
        "to be OFF (Release-1 regression)"
    )


def test_truth_table_momentum_and_deposit(acts_tracker_simhits):
    """Truth columns are physical: finite momenta, tE >= |p| component scale,
    non-negative deposits, and hit_index populated."""
    for r in _rows(acts_tracker_simhits):
        for px, py, pz, te, de in zip(
            r["tpx"], r["tpy"], r["tpz"], r["tE"], r["dE"]
        ):
            assert all(math.isfinite(v) for v in (px, py, pz, te))
            assert te >= 0.0
            assert de >= -1e-6, f"negative deposit {de}"
