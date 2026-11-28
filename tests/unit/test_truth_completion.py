"""D8: completed PU200 truth keeps every Release 1 row and adds only the rows the native-particle map holds;
packager: an incomplete candidate never overwrites the shared chunk manifest; a replacement goes through the
promotion journal in the run_stage attempt directory."""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "postprocessing"))
from utils import truth_completion as tc  # noqa: E402

pytest.importorskip("scipy")
pkg = pytest.importorskip("package_native_parquet")


def _obj(*arrs):
    o = np.empty(len(arrs), dtype=object)
    for i, a in enumerate(arrs):
        o[i] = np.asarray(a)
    return o


def _mc():
    # one entry, 4 MCParticles; particle 3's parent is 1
    f8 = lambda *v: np.array(v, dtype=np.float64)
    return {"PDG": _obj(np.array([2212, 211, -211, 22], np.int32)),
            "simulatorStatus": _obj(np.array([0, 0, 1 << 30, 0], np.int32)),
            "charge": _obj(np.array([1, 1, -1, 0], np.float32)), "time": _obj(np.array([0, .1, .2, .3], np.float32)),
            "mass": _obj(f8(.938, .1396, .1396, 0)), "vx": _obj(f8(0, 1, 2, 3)), "vy": _obj(f8(0, 1, 2, 3)),
            "vz": _obj(f8(0, 1, 2, 3)), "px": _obj(f8(1, 2, 3, 4)), "py": _obj(f8(0, 0, 0, 0)), "pz": _obj(f8(1, 1, 1, 1)),
            "parents_begin": _obj(np.array([0, 0, 0, 0], np.uint32)), "parents_end": _obj(np.array([0, 0, 0, 1], np.uint32)),
            "parents": _obj(np.array([1], np.int32))}


def _base(ids, event=10):
    n = len(ids)
    cols = {"event_id": pa.array([event], pa.uint32())}
    for f in list(tc.V1_SCHEMA)[1:]:
        vals = np.asarray(ids, dtype=np.uint64) if f.name == "particle_id" else \
            np.full(n, 7, dtype=tc.NP_TYPES[f.name]) if f.name != "primary" else np.ones(n, bool)
        cols[f.name] = pa.array([vals], type=f.type)
    return pa.table(cols, schema=tc.V1_SCHEMA)


def _native(ids):
    # remapped native table: particle_id = MCParticle index; kinematics as float32 of the MC values
    mc = _mc()
    cols = {"event_id": pa.array([0], pa.uint32()), "particle_id": pa.array([np.asarray(ids, np.uint64)])}
    for c in tc.KEY:
        cols[c] = pa.array([mc[c][0][ids].astype(np.float32)])
    cols["vertex_primary"] = pa.array([np.arange(len(ids), dtype=np.uint16) + 1])
    for c in ("perigee_d0", "perigee_z0"):
        cols[c] = pa.array([np.full(len(ids), np.nan, np.float32)])
    return pa.table(cols)


def _acts(ids):
    mc = _mc()
    d = {c: mc[c][0][ids].astype(np.float32) for c in tc.KEY}
    d["perigee_d0"] = np.arange(len(ids), dtype=np.float32) * 0.5
    d["perigee_z0"] = np.arange(len(ids), dtype=np.float32) * 2.0
    return {0: d}


def test_complete_window_keeps_base_and_adds_missing():
    base = _base([0, 1, 3])
    nat_ids = [0, 1, 2, 3]                       # the map holds MCParticle 2, Release 1 lacks it
    table, added = tc.complete_window(base, _native(nat_ids), _mc(), 0, _acts(nat_ids), 10, 0, 1)
    assert table.schema == tc.V1_SCHEMA
    assert table.column("particle_id")[0].as_py() == [0, 1, 2, 3]
    for c in tc.LIST_COLS:
        got = table.column(c)[0].as_py()
        want = base.column(c)[0].as_py()
        assert [got[i] for i in (0, 1, 3)] == want, c
    row = {c: table.column(c)[0].as_py()[2] for c in tc.LIST_COLS}
    assert row["pdg_id"] == -211 and row["primary"] is False and row["parent_id"] == -1
    assert row["energy"] == pytest.approx(float(np.float32(np.sqrt(9 + 1 + 0.1396 ** 2))))
    assert row["vertex_primary"] == 3 and row["perigee_d0"] == 1.0 and row["perigee_z0"] == 4.0
    assert added == {10: [{"particle_id": 2, "pdg_id": -211, "cause": "acts_build_difference"}]}


def test_complete_window_no_addition_is_identity():
    base = _base([0, 1, 2, 3])
    table, added = tc.complete_window(base, _native([0, 1, 2, 3]), _mc(), 0, _acts([0, 1, 2, 3]), 10, 0, 1)
    assert added == {} and table.equals(base)


def test_complete_window_refuses_misaligned_root():
    base = _base([0, 1])
    acts = _acts([0, 1, 2, 3]); acts[0]["px"] = acts[0]["px"][::-1].copy()
    with pytest.raises(RuntimeError, match="container order"):
        tc.complete_window(base, _native([0, 1, 2, 3]), _mc(), 0, acts, 10, 0, 1)


def test_column_agreement_formula_on_existing_rows():
    # Release 1 rows built by the formulas themselves agree in every column
    nat_ids = [0, 1, 2, 3]
    empty = _base([])
    full, _ = tc.complete_window(empty, _native(nat_ids), _mc(), 0, _acts(nat_ids), 10, 0, 1)
    ag = tc.column_agreement(full, _native(nat_ids), _mc(), 0, _acts(nat_ids), 10, 0, 1)
    assert all(ag[c] == [4, 0] for c in tc.LIST_COLS), ag


def _tracks(events):
    n = len(events)
    return pa.table({"event_id": pa.array(events, pa.uint32()),
                     "hit_ids": pa.array([[[1, 2]]] * n, pa.list_(pa.list_(pa.uint64()))),
                     "hit_outlier": pa.array([[[False, False]]] * n, pa.list_(pa.list_(pa.bool_())))})


def test_incomplete_candidate_keeps_shared_manifest(tmp_path, monkeypatch):
    rd = tmp_path / "runs" / "0"; (rd / "tracks").mkdir(parents=True)
    pq.write_table(_tracks([0, 1]), rd / "tracks" / "tracks_000000-000002.parquet")
    man = tmp_path / "v" / "provenance" / "package_parquet" / "chunk0_inputs.json"
    kw = dict(start_event=0, end_event=1, start_run=0, start_local=0, end_run=0, end_local=1, run_dirs=[rd],
              run_size=2, objects=["tracks"], out_base=tmp_path / "v" / "parquet", dataset_name_dot="c.d.v23",
              row_group_size=None, compression="snappy", sort_rows=True, shared_hits_from_hit_ids=True,
              manifest_path=man)
    pkg.package_chunk(**kw)
    accepted = man.read_text()
    shard = next((tmp_path / "v" / "parquet" / "reco" / "tracks").glob("*.parquet"))
    shard_bytes = shard.read_bytes()
    # second attempt: the run lost an event -> incomplete
    pq.write_table(_tracks([0]), rd / "tracks" / "tracks_000000-000002.parquet")
    att = tmp_path / "v" / ".attempts" / "package_parquet" / "0.a2"
    monkeypatch.setenv("COLLIDERML_ATTEMPT_DIR", str(att))
    with pytest.raises(RuntimeError, match="incomplete"):
        pkg.package_chunk(**kw)
    assert man.read_text() == accepted and shard.read_bytes() == shard_bytes
    assert json.loads((att / "incomplete_manifest.json").read_text())["status"] == "incomplete"
    # third attempt (explicit replacement, complete) goes through the journal in its attempt dir
    pq.write_table(_tracks([1, 0]), rd / "tracks" / "tracks_000000-000002.parquet")
    att3 = att.with_name("0.a3")
    monkeypatch.setenv("COLLIDERML_ATTEMPT_DIR", str(att3))
    pkg.package_chunk(**kw)
    j = json.loads((att3 / "promotion.json").read_text())
    assert j["state"] == "complete" and j["manifest_backup"] == str(att3 / "replaced" / man.name)
    assert (att3 / "replaced" / shard.name).read_bytes() == shard_bytes
    assert (att3 / "replaced" / man.name).read_text() == accepted
    m = json.loads(man.read_text())
    assert m["status"] == "ok" and m["replaced_backup"] == str(att3 / "replaced")
