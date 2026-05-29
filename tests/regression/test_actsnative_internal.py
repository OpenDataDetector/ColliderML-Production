"""Internal-consistency checks on the ACTS-native parquet output.

These exercise invariants the four parquets are supposed to share
*regardless* of how the legacy v1 path behaves — particles ↔ tracker_hits ↔
calo_hits ↔ tracks should all key on the same per-event particle_id space.
Run before the v1 ↔ ACTS-native diff to catch problems where the bug is on
the ACTS-native side alone.
"""

from __future__ import annotations

import polars as pl
import pytest


def _per_event_particle_ids(df: pl.DataFrame, col: str = "particle_id") -> dict[int, set[int]]:
    """Return {event_id: set(particle_id)} for a per-event nested-layout table."""
    out: dict[int, set[int]] = {}
    for row in df.iter_rows(named=True):
        ev = int(row["event_id"])
        ids = row[col]
        out[ev] = set(int(x) for x in ids if x is not None)
    return out


def test_tracker_hits_particle_id_subset_of_particles(acts_tracker_hits, acts_particles):
    """Every tracker hit's particle_id must exist in the particles table of
    the same event. If this fails, the (particles, tracker_hits) tables key
    on different enumerations and downstream joins are broken."""
    pids_by_ev = _per_event_particle_ids(acts_particles)
    hit_pids_by_ev = _per_event_particle_ids(acts_tracker_hits)

    sentinel_max_u64 = (1 << 64) - 1
    bad: list[str] = []
    for ev, hpids in hit_pids_by_ev.items():
        # Drop the documented "no match" sentinel.
        hpids = hpids - {sentinel_max_u64}
        missing = hpids - pids_by_ev.get(ev, set())
        if missing:
            bad.append(
                f"  event {ev}: {len(missing)} hit particle_ids not in particles table "
                f"(e.g. {sorted(missing)[:5]})"
            )
    if bad:
        pytest.fail(
            "tracker_hits.particle_id values must all appear in particles.particle_id "
            f"for the same event:\n" + "\n".join(bad[:5])
        )


def test_tracks_majority_particle_id_subset_of_particles(acts_tracks, acts_particles):
    """Same invariant for tracks.majority_particle_id."""
    pids_by_ev = _per_event_particle_ids(acts_particles)
    mpids_by_ev = _per_event_particle_ids(acts_tracks, col="majority_particle_id")

    sentinel_max_u64 = (1 << 64) - 1
    bad: list[str] = []
    for ev, mpids in mpids_by_ev.items():
        mpids = mpids - {sentinel_max_u64}
        missing = mpids - pids_by_ev.get(ev, set())
        if missing:
            bad.append(
                f"  event {ev}: {len(missing)} track majority_particle_ids not in particles table "
                f"(e.g. {sorted(missing)[:5]})"
            )
    if bad:
        pytest.fail(
            "tracks.majority_particle_id values must all appear in particles.particle_id "
            f"for the same event:\n" + "\n".join(bad[:5])
        )


def test_calo_contrib_particle_ids_subset_of_particles(acts_calo_hits, acts_particles):
    """Same invariant for calo_hits.contrib_particle_ids (nested list)."""
    pids_by_ev = _per_event_particle_ids(acts_particles)

    sentinel_max_u64 = (1 << 64) - 1
    bad: list[str] = []
    for row in acts_calo_hits.iter_rows(named=True):
        ev = int(row["event_id"])
        nested = row["contrib_particle_ids"]
        cell_pids = {int(p) for cell in nested for p in cell} - {sentinel_max_u64}
        missing = cell_pids - pids_by_ev.get(ev, set())
        if missing:
            bad.append(
                f"  event {ev}: {len(missing)} calo contrib pids not in particles table "
                f"(e.g. {sorted(missing)[:5]})"
            )
            if len(bad) >= 5:
                break
    if bad:
        pytest.fail(
            "calo_hits.contrib_particle_ids values must all appear in particles.particle_id "
            f"for the same event:\n" + "\n".join(bad)
        )


def test_tracker_hits_unique_xyz_count_consistent_with_simhit_count(acts_parquet_root):
    """Per event, the number of distinct (x, y, z) tuples should be <= the
    number of rows (simhits). Equality means no clustering; strict inequality
    means some simhits were cluster-merged. More unique positions than rows
    would mean the writer is emitting per-measurement, not per-simhit, rows.

    Memory-frugal: stream per-event via pyarrow + numpy (no full explode)."""
    import pyarrow.parquet as pq
    import numpy as np

    shards = sorted((acts_parquet_root / "tracker_hits").glob("*.parquet"))
    if not shards:
        pytest.skip("ACTS-native has no tracker_hits parquet shards")

    over = []
    total_simhits = total_clustered = 0
    for shard in shards:
        cols = pq.read_table(str(shard), columns=["event_id", "x", "y", "z"]).to_pydict()
        for i, ev in enumerate(cols["event_id"]):
            x = np.asarray(cols["x"][i], dtype=np.float64)
            y = np.asarray(cols["y"][i], dtype=np.float64)
            z = np.asarray(cols["z"][i], dtype=np.float64)
            finite = ~np.isnan(x)
            key = np.stack([np.round(x[finite], 4), np.round(y[finite], 4),
                            np.round(z[finite], 4)], axis=1)
            n_simhits = len(key)
            n_unique = len(np.unique(key, axis=0)) if n_simhits else 0
            if n_unique > n_simhits:
                over.append((int(ev), n_simhits, n_unique))
            total_simhits += n_simhits
            total_clustered += n_simhits - n_unique
    if over:
        pytest.fail(
            f"some events have more unique (x,y,z) than rows — the writer "
            f"may be emitting per-measurement, not per-simhit:\n{over}"
        )
    frac = total_clustered / max(total_simhits, 1)
    print(
        f"\ncluster-merging fraction in ACTS-native output: "
        f"{total_clustered}/{total_simhits} = {100*frac:.2f}%"
    )
