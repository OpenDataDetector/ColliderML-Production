"""v23 production controller (E2, E3, E7) and attempt isolation: unit enumeration, barriers, input audit,
reconcile, writer exclusion, promotion."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "cli"))
sys.path.insert(0, str(REPO / "scripts" / "production"))
sys.path.insert(0, str(REPO / "scripts" / "postprocessing"))
import attempt_isolation as ai  # noqa: E402
import run_provenance as rp  # noqa: E402
import v23_control as vc  # noqa: E402


def test_chunk_contributors_agree_with_the_packager():
    from package_native_parquet import release_chunk_window
    for release, chunk, rs in ((100000, 100, 64), (10240, 1000, 1280)):
        n = (release + chunk - 1) // chunk
        covered = []
        three = 0
        for i in range(n):
            assert vc.chunk_window(i, chunk, release) == release_chunk_window(i, chunk, release)
            c = vc.contributors(i, chunk, release, rs)
            three += len(c) == 3
            for x in c:
                covered += [x["run"] * rs + k for k in range(x["first"], x["last"] + 1)]
        assert covered == list(range(release)), "chunks tile the release exactly, run-local ranges included"
        with pytest.raises(ValueError):
            vc.chunk_window(n, chunk, release)
    assert vc.contributors(999, 100, 100000, 64) == [{"run": 1560, "first": 60, "last": 63},
                                                     {"run": 1561, "first": 0, "last": 63},
                                                     {"run": 1562, "first": 0, "last": 31}]
    assert vc.contributors(10, 1000, 10240, 1280) == [{"run": 7, "first": 1040, "last": 1279}]
    assert vc.contributors(12, 100, 100000, 64)[0]["run"] == 18 and len(vc.contributors(12, 100, 100000, 64)) == 3


def test_attempt_isolation_promotion(tmp_path):
    run = tmp_path / "runs" / "5"
    run.mkdir(parents=True)
    (run / "sim.root").write_text("input")
    (run / "timing_summary.txt").write_text("tracking timing\n")
    (run / "reco_parts").mkdir()
    (run / "reco_parts" / "old.root").write_text("old")
    work = ai.attempt_root(tmp_path, "pandora_reco", 5, 2) / "5"
    assert ai.prepare(run, work, ai.top_names(["sim.root", "absent.root"])) == ["sim.root"]
    assert (work / "sim.root").is_symlink()
    (work / "reco_parts").mkdir()
    (work / "reco_parts" / "reco_part_000.root").write_text("new")
    (work / "pandora_slices.json").write_text("{}")
    (work / "timing_summary.txt").write_text("pandora timing\n")
    plan = ai.plan(work, run, ai.top_names(["pandora_slices.json", "reco_parts/reco_part_*.root|reco_edm4hep.root"]))
    assert {m["name"] for m in plan["moves"]} == {"reco_parts", "pandora_slices.json"}
    assert plan["embed"] == ["timing_summary.txt"]
    done = ai.promote(work, run, plan, work.parent / "replaced")
    assert done["replaced"] == ["reco_parts"] and done["embedded"]["timing_summary.txt"] == "pandora timing\n"
    assert (run / "reco_parts" / "reco_part_000.root").read_text() == "new"
    assert (work.parent / "replaced" / "reco_parts" / "old.root").read_text() == "old"
    assert (run / "timing_summary.txt").read_text() == "tracking timing\n", "another stage's file is untouched"
    assert (run / "sim.root").read_text() == "input"
    assert not ai.cleanup(work.parent), "the replaced outputs are kept"
    assert not (work / "sim.root").exists()
    with pytest.raises(ValueError):
        ai.purge(tmp_path / "runs")


def test_lock_first_and_freeze(tmp_path):
    lock = rp.acquire_unit_lock(tmp_path / "prov", "digitization")
    assert json.loads(lock.read_text())["pid"] == os.getpid()
    with pytest.raises(FileExistsError):
        rp.acquire_unit_lock(tmp_path / "prov", "digitization")
    assert rp.frozen_error(tmp_path) is None
    (tmp_path / ".freeze").write_text("finalize")
    assert "frozen" in rp.frozen_error(tmp_path)
    (tmp_path / ".freeze").unlink()
    (tmp_path / ".git_commit_success").write_text("x")
    assert "finalized" in rp.frozen_error(tmp_path)


def test_finalizer_refuses_with_a_held_unit_lock(tmp_path):
    base = tmp_path / "base"
    v = base / "hard_scatter" / "ttbar" / "v23"
    (v / "runs" / "0" / "provenance").mkdir(parents=True)
    (v / "runs" / "0" / "provenance" / "reco_tables.lock").write_text("{}")
    cfg = tmp_path / "pkg.yaml"
    cfg.write_text(yaml.safe_dump({"campaign": "hard_scatter", "dataset": "ttbar", "version": "v23",
                                   "common": {"output_base_dir": str(base)}, "release_events": 10,
                                   "chunk_size": 10, "objects": ["tracks"]}))
    out = subprocess.run([sys.executable, str(REPO / "scripts/production/finalize_version.py"), str(cfg),
                          "--approved-sha", "0" * 40, "--dry-run"], capture_output=True, text=True)
    assert out.returncode == 1 and "unit lock" in out.stdout
    assert not (v / ".freeze").exists(), "the freeze is released after a refusal"
    (v / ".freeze").write_text("another holder")
    out = subprocess.run([sys.executable, str(REPO / "scripts/production/finalize_version.py"), str(cfg),
                          "--approved-sha", "0" * 40, "--dry-run"], capture_output=True, text=True)
    assert out.returncode == 1 and "exists" in out.stdout and (v / ".freeze").read_text() == "another holder"


SHA = "a" * 40
IMG = {"sim": "1" * 64, "reco": "2" * 64}


def _rec_file(prov, stem, k, **kw):
    prov.mkdir(parents=True, exist_ok=True)
    rec = {"status": "complete", "return_code": 0, "git": {"sha": SHA}, "effective_config_sha256": "cfg",
           "image_id": IMG["sim"]} | kw
    p = prov / f"{stem}_attempt{k}.json"
    p.write_text(json.dumps(rec))
    return p


def _out(base, rel):
    p = base / rel
    return {"path": rel, "realpath": os.path.realpath(p), "size": p.stat().st_size, "mtime": p.stat().st_mtime,
            "sha256": rp.sha256_file(p)}


@pytest.fixture
def mini(tmp_path):
    """Two PU200-like runs of 4 events, chunks of 3 events over a 6-event release (chunk 1 spans both runs)."""
    v = tmp_path / "v23"
    parent = tmp_path / "parent"
    for r in (0, 1):
        rd = v / "runs" / str(r)
        rd.mkdir(parents=True)
        (parent / str(r)).mkdir(parents=True)
        g4 = parent / str(r) / "edm4hep.root"
        g4.write_text(f"geant4 {r}")
        os.chmod(g4, 0o444)
        os.symlink(g4, rd / "edm4hep.root")
    cr = {"version_dir": str(v), "n_runs": 2, "run_size": 4, "chunk_size": 3, "release_events": 6,
          "objects": ["tracks"], "parent_names": ["edm4hep.root"], "carried_objects": [], "carried": {},
          "stages": {s: {"effective_config_sha256": "cfg", "container": "sim" if s != "pandora_reco" else "reco"}
                     for s in vc.STAGES}}
    cr["parents"] = vc._parents(cr, ["edm4hep.root"], [0, 1])
    rec = {"approved_sha": SHA, "images": {"sim": {"id": IMG["sim"]}, "reco": {"id": IMG["reco"]}},
           "campaigns": {"full_pileup": cr}}
    return rec, cr, v


def _accept_tracking(v, r, sha=SHA):
    rd = v / "runs" / str(r)
    (rd / "tracks").mkdir(exist_ok=True)
    (rd / "tracks" / "t.parquet").write_text(f"tracks {r}")
    (rd / "sim_with_tracks.root").write_text(f"sim {r}")
    return _rec_file(rd / "provenance", "digitization", 1, git={"sha": sha},
                     inputs=[_out(rd, "edm4hep.root")], outputs=[_out(rd, "tracks/t.parquet"), _out(rd, "sim_with_tracks.root")])


def test_barrier_audit_and_reconcile(mini):
    rec, cr, v = mini
    jobs = set()
    assert vc.unit_state(rec, cr, "digitization", 0, jobs)["state"] == "missing"
    assert vc.upstream_problems(rec, cr, "digitization", 0, jobs, {}) == []
    _accept_tracking(v, 0)
    assert vc.unit_state(rec, cr, "digitization", 0, jobs)["state"] == "accepted"
    assert vc.audit_unit(rec, cr, "digitization", 0, jobs, {}) == []
    assert vc.upstream_problems(rec, cr, "pandora_reco", 0, jobs, {}) == []
    assert "missing" in vc.upstream_problems(rec, cr, "pandora_reco", 1, jobs, {})[0]
    # a consumer whose recorded input differs from the accepted producer output is not bound
    rd = v / "runs" / "0"
    good = _out(rd, "sim_with_tracks.root")
    _rec_file(rd / "provenance", "pandora_reco", 1, image_id=IMG["reco"], inputs=[good], outputs=[])
    assert vc.audit_unit(rec, cr, "pandora_reco", 0, jobs, {}) == []
    _rec_file(rd / "provenance", "pandora_reco", 2, image_id=IMG["reco"], inputs=[good | {"sha256": "0" * 64}], outputs=[])
    assert "differs from the accepted digitization" in vc.audit_unit(rec, cr, "pandora_reco", 0, jobs, {})[0]
    # wrong image or SHA: not accepted for production
    _rec_file(rd / "provenance", "pandora_reco", 3, image_id="3" * 64, inputs=[good], outputs=[])
    assert vc.unit_state(rec, cr, "pandora_reco", 0, jobs)["state"] == "mismatch"
    # a changed upstream output stops the barrier
    (rd / "sim_with_tracks.root").write_text("changed sim")
    assert "changed" in vc.upstream_problems(rec, cr, "pandora_reco", 0, jobs, {})[0]
    # chunk 1 (events 3-5) needs runs 0 and 1: run 1 missing blocks it
    probs = vc.upstream_problems(rec, cr, "package_parquet", 1, jobs, {})
    assert any("digitization run 1: missing" in p for p in probs)
    # a stale lock (owner job ended) is reconciled: record interrupted, lock removed
    p1 = v / "runs" / "1" / "provenance"
    p1.mkdir(parents=True)
    (p1 / "digitization.lock").write_text(json.dumps({"host": "nid0", "pid": 1, "slurm_job_id": "999999999"}))
    (p1 / "digitization_attempt1.json").write_text(json.dumps({"status": "started"}))
    assert vc.unit_state(rec, cr, "digitization", 1, jobs)["state"] == "stale_lock"
    assert vc.unit_state(rec, cr, "digitization", 1, {"999999999"})["state"] == "running"


def test_audit_chunk_inputs(mini):
    rec, cr, v = mini
    for r in (0, 1):
        _accept_tracking(v, r)
    pd = v / "provenance" / "package_parquet"

    def chunk_rec(k, inputs, start=3):
        return _rec_file(pd, "chunk1", k, container_record={"start_event": start, "end_event": 5, "inputs": inputs,
                                                             "outputs": {}})
    inp = {}
    for r in (0, 1):
        rd = v / "runs" / str(r)
        e = _out(rd, "tracks/t.parquet")
        inp[str(rd / "tracks/t.parquet")] = {"realpath": e["realpath"], "size": e["size"], "sha256": e["sha256"]}
        st = vc.stat_of(rd / "edm4hep.root")
        inp[str(rd / "edm4hep.root")] = {k: st[k] for k in ("realpath", "size", "mtime_ns", "inode")}
    chunk_rec(1, inp)
    assert vc.audit_unit(rec, cr, "package_parquet", 1, set(), {}) == []
    bad = dict(inp)
    k0 = str(v / "runs/1/tracks/t.parquet")
    bad[k0] = dict(bad[k0], sha256="0" * 64)
    chunk_rec(2, bad)
    assert any("differs from the accepted digitization" in p for p in vc.audit_unit(rec, cr, "package_parquet", 1, set(), {}))
    missing = {k: x for k, x in inp.items() if "/runs/1/" not in k}
    chunk_rec(3, missing)
    assert any("contributing run 1" in p for p in vc.audit_unit(rec, cr, "package_parquet", 1, set(), {}))
    chunk_rec(4, inp, start=0)
    assert any("manifest window" in p for p in vc.audit_unit(rec, cr, "package_parquet", 1, set(), {}))


def test_launch_record_digest(tmp_path):
    rec = {"schema": vc.SCHEMA, "a": 1}
    rec["digest"] = vc.record_digest(rec)
    p = tmp_path / "l.json"
    p.write_text(json.dumps(rec))
    assert vc.load_record(p)["a"] == 1
    rec["a"] = 2
    p.write_text(json.dumps(rec))
    with pytest.raises(SystemExit):
        vc.load_record(p)


def test_reconcile_one_pass_clears_tmp_of_a_stale_chunk_lock(mini, monkeypatch):
    rec, cr, v = mini
    monkeypatch.setattr(vc, "live_jobs", lambda: set())
    pd = v / "provenance" / "package_parquet"
    pd.mkdir(parents=True)
    (pd / "chunk1.lock").write_text(json.dumps({"host": "nidX", "pid": 1, "slurm_job_id": "424242"}))
    (pd / "chunk1_attempt1.json").write_text(json.dumps({"status": "started"}))
    d = v / "parquet" / "reco" / "tracks"
    d.mkdir(parents=True)
    tmp = d / ".full_pileup.ttbar.v23.reco.tracks.events3-5.parquet.abc.tmp"
    tmp.write_text("partial")
    live_tmp = d / ".full_pileup.ttbar.v23.reco.tracks.events0-2.parquet.def.tmp"
    live_tmp.write_text("partial")
    (pd / "chunk0.lock").write_text(json.dumps({"host": "nidX", "pid": 1, "slurm_job_id": "777"}))
    monkeypatch.setattr(vc, "live_jobs", lambda: {"777"})
    assert vc.reconcile(rec, "full_pileup", apply=True, purge_failed=False) == 0
    assert not tmp.exists() and not (pd / "chunk1.lock").exists()
    assert json.loads((pd / "chunk1_attempt1.json").read_text())["status"] == "interrupted"
    assert live_tmp.exists() and (pd / "chunk0.lock").exists(), "a live chunk keeps its lock and temporary shard"
