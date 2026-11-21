"""v23 packaging: n_shared_hits from hit_ids, outer rows sorted by event_id, and the chunk
writer applying both (sorted output, shared hits after the covariance-join column)."""

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "postprocessing"))
pytest.importorskip("scipy")
pkg = pytest.importorskip("package_native_parquet")


def _tracks(events, hit_ids, outliers=None, with_zero_col=True):
    """events: list of event ids; hit_ids/outliers: per event, per track lists."""
    if outliers is None:
        outliers = [[[False] * len(t) for t in ev] for ev in hit_ids]
    cols = {
        "event_id": pa.array(events, type=pa.uint32()),
        "hit_ids": pa.array(hit_ids, type=pa.list_(pa.list_(pa.uint32()))),
        "hit_outlier": pa.array(outliers, type=pa.list_(pa.list_(pa.bool_()))),
        "track_id": pa.array([list(range(len(ev))) for ev in hit_ids], type=pa.list_(pa.uint16())),
    }
    if with_zero_col:
        cols["n_shared_hits"] = pa.array([[0] * len(ev) for ev in hit_ids], type=pa.list_(pa.int32()))
    return pa.table(cols)


def _shared(table):
    return table.column("n_shared_hits").to_pylist()


def test_shared_hits_basic():
    t = _tracks([0], [[[1, 2, 3, 4], [3, 4, 5], [9]]])
    assert _shared(pkg.compute_shared_hits(t)) == [[2, 2, 0]]


def test_shared_hits_per_event_not_across_events():
    # same hit id in two events is not shared
    t = _tracks([0, 1], [[[1, 2]], [[1, 2]]])
    assert _shared(pkg.compute_shared_hits(t)) == [[0], [0]]


def test_shared_hits_outliers_do_not_count():
    # hit 3 is an outlier on track 1: not shared for either track
    t = _tracks([0], [[[1, 3], [3, 4]]], outliers=[[[False, False], [True, False]]])
    assert _shared(pkg.compute_shared_hits(t)) == [[0, 0]]


def test_shared_hits_repeat_within_track_counts_once():
    t = _tracks([0], [[[7, 7, 8], [8]]])
    assert _shared(pkg.compute_shared_hits(t)) == [[1, 1]]


def test_shared_hits_three_tracks_and_empty_event():
    t = _tracks([0, 1], [[[1], [1], [1, 2]], []])
    out = pkg.compute_shared_hits(t)
    assert _shared(out) == [[1, 1, 1], []]
    assert out.schema.field("n_shared_hits").type == pa.list_(pa.int32())
    assert out.column_names == t.column_names, "replaced in place, schema order kept"


def test_shared_hits_appended_when_absent():
    t = _tracks([0], [[[1, 2], [2]]], with_zero_col=False)
    out = pkg.compute_shared_hits(t)
    assert out.column_names[-1] == "n_shared_hits"
    assert _shared(out) == [[1, 1]]


def test_shared_hits_matches_bruteforce_random():
    rng = np.random.default_rng(1)
    hit_ids, outl = [], []
    for _ in range(20):
        ev_h, ev_o = [], []
        for _ in range(rng.integers(0, 12)):
            n = int(rng.integers(1, 10))
            ev_h.append(rng.integers(0, 40, n).tolist())
            ev_o.append((rng.random(n) < 0.1).tolist())
        hit_ids.append(ev_h)
        outl.append(ev_o)
    out = _shared(pkg.compute_shared_hits(_tracks(list(range(20)), hit_ids, outl)))
    for e in range(20):
        meas = [{h for h, o in zip(hs, os_) if not o} for hs, os_ in zip(hit_ids[e], outl[e])]
        expect = [sum(1 for h in m if any(h in meas[j] for j in range(len(meas)) if j != i))
                  for i, m in enumerate(meas)]
        assert out[e] == expect


def test_sort_by_event_id():
    t = _tracks([5, 2, 9], [[[1]], [[2]], [[3]]])
    s = pkg.sort_by_event_id(t, "tracks")
    assert s.column("event_id").to_pylist() == [2, 5, 9]
    assert s.column("hit_ids").to_pylist() == [[[2]], [[1]], [[3]]], "rows move as a whole"


def test_sort_rejects_duplicate_event_ids():
    t = _tracks([1, 1], [[[1]], [[2]]])
    with pytest.raises(RuntimeError, match="duplicated"):
        pkg.sort_by_event_id(t, "tracks")


def test_package_chunk_sorted_and_shared(tmp_path):
    # two runs of 3 events, native rows out of order; chunk spans both runs
    runs = []
    for r, order in enumerate([[2, 0, 1], [1, 2, 0]]):
        rd = tmp_path / "runs" / str(r)
        (rd / "tracks").mkdir(parents=True)
        t = _tracks(order, [[[10 + e, 20], [20]] for e in order])
        pq.write_table(t, rd / "tracks" / "tracks_000000-000003.parquet")
        runs.append(rd)
    out = tmp_path / "out"
    pkg.package_chunk(start_event=0, end_event=5, start_run=0, start_local=0, end_run=1, end_local=2,
                      run_dirs=runs, run_size=3, objects=["tracks"], out_base=out,
                      dataset_name_dot="c.d.v23", row_group_size=None, compression="snappy",
                      sort_rows=True, shared_hits_from_hit_ids=True)
    f = next((out / "reco" / "tracks").glob("*.parquet"))
    t = pq.read_table(f)
    assert t.column("event_id").to_pylist() == [0, 1, 2, 3, 4, 5]
    assert _shared(t) == [[1, 1]] * 6


def test_v23_package_configs_enable_both():
    import yaml
    for camp in ("hard_scatter", "full_pileup"):
        c = yaml.safe_load(open(REPO / f"configs_production/{camp}/ttbar/v23/package_parquet_config.yaml"))
        assert c["sort_by_event_id"] is True and c["shared_hits_from_hit_ids"] is True
        assert c["track_covariance"] is True
        assert "vertices" in c["objects"] and "generator_particles" in c["objects"]
        assert "staging" in c["output_base_dir"] and "staging" in c["input_base_dir"]
    c = yaml.safe_load(open(REPO / "configs_production/full_pileup/ttbar/v23/package_parquet_config.yaml"))
    assert c["job_config"]["n_runs"] * c["chunk_size"] == 100_000
