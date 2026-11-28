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
    assert not (work / "sim.root").is_symlink() and os.stat(work / "sim.root").st_ino == os.stat(run / "sim.root").st_ino, \
        "a produced input is pinned by a hard link to its inode"
    (work / "reco_parts").mkdir()
    (work / "reco_parts" / "reco_part_000.root").write_text("new")
    (work / "pandora_slices.json").write_text("{}")
    (work / "timing_summary.txt").write_text("pandora timing\n")
    plan = ai.plan(work, run, ai.top_names(["pandora_slices.json", "reco_parts/reco_part_*.root|reco_edm4hep.root"]),
                   inputs=["sim.root"])
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

    n = [0]

    def chunk_rec(_k, inputs, start=3, end=5):     # attempts numbered in call order (the latest is the last call)
        n[0] += 1
        return _rec_file(pd, "chunk1", n[0], container_record={"start_event": start, "end_event": end, "inputs": inputs,
                                                             "outputs": {}})
    inp = {}
    for r in (0, 1):
        rd = v / "runs" / str(r)
        e = _out(rd, "tracks/t.parquet")
        inp[str(rd / "tracks/t.parquet")] = {"realpath": e["realpath"], "size": e["size"], "sha256": e["sha256"]}
    chunk_rec(1, inp)
    assert vc.audit_unit(rec, cr, "package_parquet", 1, set(), {}) == []
    st = vc.stat_of(v / "runs/0/edm4hep.root")      # an input outside the required set (no remap configured)
    chunk_rec(5, dict(inp, **{str(v / "runs/0/edm4hep.root"): {k: st[k] for k in ("realpath", "size", "mtime_ns", "inode")}}))
    assert any("outside the required set" in p for p in vc.audit_unit(rec, cr, "package_parquet", 1, set(), {}))
    bad = dict(inp)
    k0 = str(v / "runs/1/tracks/t.parquet")
    bad[k0] = dict(bad[k0], sha256="0" * 64)
    chunk_rec(2, bad)
    assert any("differs from the accepted digitization" in p for p in vc.audit_unit(rec, cr, "package_parquet", 1, set(), {}))
    missing = {k: x for k, x in inp.items() if "/runs/1/" not in k}
    chunk_rec(3, missing)
    assert any("not in the manifest" in p for p in vc.audit_unit(rec, cr, "package_parquet", 1, set(), {}))
    chunk_rec(4, inp, start=0)
    assert any("manifest window" in p for p in vc.audit_unit(rec, cr, "package_parquet", 1, set(), {}))
    chunk_rec(7, inp, end=99999)                     # the end of the window is checked too
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


def _pub_fixture(tmp_path, monkeypatch):
    v = tmp_path / "staging" / "hard_scatter" / "ttbar" / "v23"
    f = v / "parquet" / "reco" / "tracks" / "hard_scatter.ttbar.v23.reco.tracks.events0-9.parquet"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"shard bytes")
    (v / ".freeze").write_text("finalized")
    rec = {"digest": "L" * 64, "campaigns": {"hard_scatter": {"version_dir": str(v)}}}
    man = {"campaign": "hard_scatter", "launch_record_digest": rec["digest"],
           "files": {str(f.relative_to(v / "parquet")): {"sha256": rp.sha256_file(f), "kind": "produced"}}}
    man["digest"] = vc._manifest_digest(man)
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(man))
    monkeypatch.setattr(vc, "release_preflight", lambda *a, **k: [])
    return rec, v, f, man, mp, tmp_path / "published"


def test_publish_transaction_and_rollback(tmp_path, monkeypatch):
    rec, v, f, man, mp, root = _pub_fixture(tmp_path, monkeypatch)
    tgt = root / "hard_scatter" / "ttbar" / "v23"
    bad = dict(man, campaign="full_pileup"); bad["digest"] = vc._manifest_digest(bad)
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    assert vc.release_publish(rec, "hard_scatter", tmp_path / "bad.json", root, 0, 2) == 1, "wrong-campaign manifest"
    edited = dict(man, release_events=1)
    (tmp_path / "edited.json").write_text(json.dumps(edited))
    assert vc.release_publish(rec, "hard_scatter", tmp_path / "edited.json", root, 0, 2) == 1, "digest of an edited manifest"
    extra = v / "parquet" / "reco" / "extra.parquet"
    extra.write_text("x")
    assert vc.release_publish(rec, "hard_scatter", mp, root, 0, 2) == 1, "a file outside the manifest"
    extra.unlink()
    real_rename, calls = os.rename, []

    def failing(a, b):      # the publication point fails after parquet/ moved into the hidden assembly dir
        calls.append((a, b))
        if len(calls) == 2:
            raise OSError("simulated failure at the publication point")
        return real_rename(a, b)
    monkeypatch.setattr(vc.os, "rename", failing)
    with pytest.raises(OSError):
        vc.release_publish(rec, "hard_scatter", mp, root, 0, 2)
    monkeypatch.setattr(vc.os, "rename", real_rename)
    assert f.exists() and not tgt.exists() and not list(tgt.parent.glob(".v23.publishing.*")), "restored"
    assert vc.release_publish(rec, "hard_scatter", mp, root, 0, 2) == 0
    assert not (v / "parquet").exists() and (tgt / "parquet").is_dir()
    assert vc.release_verify_bytes(rec, "hard_scatter", root, None, 2) == 1, "the portal check is required"
    assert vc.release_verify_bytes(rec, "hard_scatter", root, None, 2, no_portal=True) == 0
    pf = tgt / "parquet" / f.relative_to(v / "parquet")
    os.chmod(pf, 0o644); pf.write_bytes(b"shard bytez")
    assert vc.release_verify_bytes(rec, "hard_scatter", root, None, 2, no_portal=True) == 1, "a corrupted byte"
    assert vc.release_rollback(rec, "hard_scatter", root) == 1, "restored, but the rehash finds the corrupted byte"
    assert (v / "parquet").is_dir() and not tgt.exists()
    os.chmod(f, 0o644); f.write_bytes(b"shard bytes")
    assert vc.release_publish(rec, "hard_scatter", mp, root, 0, 2) == 0
    assert vc.release_rollback(rec, "hard_scatter", root) == 0, "a clean rollback verifies the restored tree"


def test_rollback_of_an_interrupted_publication(tmp_path, monkeypatch):
    rec, v, f, man, mp, root = _pub_fixture(tmp_path, monkeypatch)
    tgt = root / "hard_scatter" / "ttbar" / "v23"
    hidden = tgt.parent / ".v23.publishing.T1"
    hidden.mkdir(parents=True)
    (hidden / ".published").write_text(json.dumps({"from": str(v)}))
    os.rename(v / "parquet", hidden / "parquet")      # killed between the move and the publication point
    assert vc.release_rollback(rec, "hard_scatter", root) == 0
    assert f.exists() and not hidden.exists()


def test_audit_rejects_another_output_of_the_same_producer(mini):
    rec, cr, v = mini
    for r in (0, 1):
        _accept_tracking(v, r)
    rd = v / "runs" / "1"
    (rd / "truth_tracks").mkdir()
    (rd / "truth_tracks" / "tt.parquet").write_text("truth tracks 1")
    p = sorted((rd / "provenance").glob("digitization_attempt*.json"))[-1]
    r_ = json.loads(p.read_text()); r_["outputs"].append(_out(rd, "truth_tracks/tt.parquet")); p.write_text(json.dumps(r_))
    pd = v / "provenance" / "package_parquet"
    inp = {}
    for r in (0, 1):
        e = _out(v / "runs" / str(r), "tracks/t.parquet")
        inp[str(v / "runs" / str(r) / "tracks/t.parquet")] = {"realpath": e["realpath"], "size": e["size"], "sha256": e["sha256"]}
    _rec_file(pd, "chunk1", 1, container_record={"start_event": 3, "end_event": 5, "inputs": dict(inp), "outputs": {}})
    assert vc.audit_unit(rec, cr, "package_parquet", 1, set(), {}) == []
    sub = _out(rd, "truth_tracks/tt.parquet")
    inp[str(rd / "tracks/t.parquet")] = {"realpath": sub["realpath"], "size": sub["size"], "sha256": sub["sha256"]}
    _rec_file(pd, "chunk1", 2, container_record={"start_event": 3, "end_event": 5, "inputs": inp, "outputs": {}})
    assert any("not that output" in x for x in vc.audit_unit(rec, cr, "package_parquet", 1, set(), {}))


def test_parent_sha_is_frozen_and_compared(mini):
    rec, cr, v = mini
    f = cr["parents"]["0"]["edm4hep.root"]["files"]["."]
    assert "sha256" in f, "a small parent file is frozen with its sha256"


# ------------------------------------------------------------------------------------- Codex re-review (round 7)
def test_f_authorization_outside_repo_and_bound_decisions(tmp_path):
    dec = tmp_path / "plan.md"
    dec.write_text("signed plan")
    good = {"approved_sha": "b" * 40, "authorized_by": "lead", "date": "2026-11-28",
            "decision_records": {str(dec): rp.sha256_file(dec)}}
    f = tmp_path / "auth.json"
    f.write_text(json.dumps(good))
    auth, probs = vc.load_authorization(str(f))
    assert probs == [] and auth["approved_sha"] == "b" * 40
    inside = REPO / "configs_production" / ".auth_test_tmp.json"
    try:
        inside.write_text(json.dumps(good))
        assert vc.load_authorization(str(inside))[1], "an authorization inside the repo (would dirty the tree) is refused"
    finally:
        inside.unlink()
    f.write_text(json.dumps(dict(good, approved_sha="FILL_AT_SIGN_OFF")))
    assert any("40 hex" in x for x in vc.load_authorization(str(f))[1])
    dec.write_text("edited after signing")
    f.write_text(json.dumps(good))
    assert any("decision record" in x for x in vc.load_authorization(str(f))[1])
    # the tracked spec keeps its placeholder: freezing never needs an edit of it
    spec = yaml.safe_load((REPO / "configs_production/v23_launch_spec.yaml").read_text())
    assert spec["approved_sha"] == "FILL_AT_SIGN_OFF" and spec["campaigns"]["full_pileup"]["carried"] == []


def test_f_frozen_archive_is_what_the_driver_loads():
    # Codex: the pub record bound the 20261106 tarball to a61ba905 through inherited defaults
    raw = {"common": {"container": "localhost/colliderml/sw:x", "container_tarball": "/img/sw.tar",
                      "stage_containers": {"pandora_reco": {"container": "ghcr.io/opendatadetector/colliderml-reco:20261127"}}}}
    assert vc.driver_tarball(raw, "pandora_reco") is None, "an override without its own archive loads none"
    assert vc.driver_tarball(raw, "digitization") == "/img/sw.tar"
    raw["common"]["stage_containers"]["pandora_reco"]["container_tarball"] = "/img/reco.tar"
    assert vc.driver_tarball(raw, "pandora_reco") == "/img/reco.tar"


def test_f_plan_never_lists_a_unit_dispatch_refuses(mini):
    rec, cr, v = mini
    p0 = v / "runs" / "0" / "provenance"
    _rec_file(p0, "digitization", 1, status="incomplete", return_code=4)
    _rec_file(p0, "digitization", 2, status="incomplete", return_code=4)
    probs = vc.unit_dispatch_problems(rec, cr, "digitization", 0, set(), {})
    assert any("2 failed attempts" in x for x in probs), "twice failed: stop for a decision, also in plan"
    p1 = v / "runs" / "1" / "provenance"
    _accept_tracking(v, 1)
    _rec_file(p1, "digitization", 2, status="interrupted", return_code=137)
    assert vc.unit_state(rec, cr, "digitization", 1, set())["state"] == "replacement_incomplete"
    assert any("replacement_incomplete" in x for x in vc.unit_dispatch_problems(rec, cr, "digitization", 1, set(), {}))


def test_f_provenance_required_refuses_before_any_write(tmp_path):
    base = tmp_path / "base"
    v = base / "full_pileup" / "ttbar" / "v23"
    v.mkdir(parents=True)
    (v / ".provenance_required").write_text("")
    cfg = {"campaign": "full_pileup", "dataset": "ttbar", "version": "v23", "stage": "digitization",
           "common": {"output_base_dir": str(base)}}
    assert "provenance_required" in rp.provenance_required_error(cfg)
    assert rp.provenance_required_error(dict(cfg, provenance={"record": True})) is None
    cf = tmp_path / "c.yaml"
    cf.write_text(yaml.safe_dump(cfg))
    before = sorted(p_.name for p_ in v.iterdir())
    out = subprocess.run([sys.executable, str(REPO / "scripts/cli/provenance_preflight.py"), str(cf)],
                         capture_output=True, text=True)
    assert out.returncode == 1 and "provenance_required" in out.stdout
    assert sorted(p_.name for p_ in v.iterdir()) == before, "the driver preflight wrote nothing"


def test_b_parent_sha_substitution_and_pinned_generation(mini, tmp_path):
    rec, cr, v = mini
    for r in (0, 1):
        _accept_tracking(v, r)
    # Codex substitution 1: the recorded parent (generator-like small file) SHA-256 replaced by 64 zeroes
    cr["package_flags"] = {"remap_particle_ids": True}
    cr["objects"] = ["tracks"]
    rd = {r: v / "runs" / str(r) for r in (0, 1)}
    for r in (0, 1):
        (rd[r] / "particles").mkdir()
        (rd[r] / "particles" / "p.parquet").write_text(f"particles {r}")
        pth = sorted((rd[r] / "provenance").glob("digitization_attempt*.json"))[-1]
        r_ = json.loads(pth.read_text()); r_["outputs"].append(_out(rd[r], "particles/p.parquet")); pth.write_text(json.dumps(r_))
    inp = {}
    for r in (0, 1):
        for rel in ("tracks/t.parquet", "particles/p.parquet"):
            e = _out(rd[r], rel)
            inp[str(rd[r] / rel)] = {"realpath": e["realpath"], "size": e["size"], "sha256": e["sha256"]}
        st = vc.stat_of(rd[r] / "edm4hep.root")
        inp[str(rd[r] / "edm4hep.root")] = {k: st[k] for k in ("realpath", "size", "mtime_ns", "inode")} | \
            {"sha256": cr["parents"][str(r)]["edm4hep.root"]["files"]["."]["sha256"]}
    pd = v / "provenance" / "package_parquet"
    _rec_file(pd, "chunk1", 1, container_record={"start_event": 3, "end_event": 5, "inputs": inp, "outputs": {}})
    assert vc.audit_unit(rec, cr, "package_parquet", 1, set(), {}) == []
    bad = json.loads(json.dumps(inp))
    bad[str(rd[0] / "edm4hep.root")]["sha256"] = "0" * 64
    _rec_file(pd, "chunk1", 2, container_record={"start_event": 3, "end_event": 5, "inputs": bad, "outputs": {}})
    assert any("differs from the frozen parent" in x for x in vc.audit_unit(rec, cr, "package_parquet", 1, set(), {}))
    # concurrent upstream replacement: the consumer pinned the old inode; the new accepted generation has other bytes
    run = tmp_path / "r" / "runs" / "3"
    run.mkdir(parents=True)
    (run / "sim_with_tracks.root").write_text("generation 1")
    work = ai.attempt_root(tmp_path / "r", "pandora_reco", 3, 1) / "3"
    ai.prepare(run, work, ["sim_with_tracks.root"])
    os.rename(run / "sim_with_tracks.root", tmp_path / "old_gen")         # the replacement moves the old entry aside
    (run / "sim_with_tracks.root").write_text("generation 2")
    assert (work / "sim_with_tracks.root").read_text() == "generation 1", "the attempt still reads what it pinned"
    pins = ai.pinned_inputs(work.parent)
    assert pins["files"]["sim_with_tracks.root"] == os.path.realpath(run / "sim_with_tracks.root")


def test_e_promotion_journal_rollback_restores_one_generation(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "5"
    run.mkdir(parents=True)
    for n in ("a.root", "b.root"):
        (run / n).write_text(f"old {n}")
    adir = ai.attempt_root(tmp_path, "digitization", 5, 2)
    work = adir / "5"
    work.mkdir(parents=True)
    for n in ("a.root", "b.root", "c.root"):
        (work / n).write_text(f"new {n}")
    plan = ai.plan(work, run, ["a.root", "b.root", "c.root"])
    real, n = os.rename, [0]

    def kill_after_three(a_, b_):
        n[0] += 1
        if n[0] == 4:
            raise KeyboardInterrupt("killed mid-promotion")
        return real(a_, b_)
    for kill_at in range(1, 6):
        for f in ("a.root", "b.root"):
            assert (run / f).read_text() == f"old {f}"
        assert not (run / "c.root").exists()
        n[0] = 0
        monkeypatch.setattr(ai.os, "rename", lambda a_, b_, k=kill_at: (_ for _ in ()).throw(KeyboardInterrupt())
                            if (n.__setitem__(0, n[0] + 1) or n[0] == k) else real(a_, b_))
        try:
            ai.promote(work, run, plan, adir / "replaced", journal_dir=adir)
            interrupted = False
        except KeyboardInterrupt:
            interrupted = True
        monkeypatch.setattr(ai.os, "rename", real)
        assert rp.read_journal(adir)["state"] == "started"
        ai.rollback(adir)
        assert rp.read_journal(adir)["state"] == "rolled_back"
        for f in ("a.root", "b.root"):
            assert (run / f).read_text() == f"old {f}", f"kill after rename {kill_at}: previous generation restored"
        assert not (run / "c.root").exists()
        assert sorted(x.name for x in work.iterdir()) == ["a.root", "b.root", "c.root"], "new bytes kept for a rerun"
        (adir / rp.JOURNAL).unlink()
        assert interrupted or kill_at == 5


def test_e_reconcile_rolls_back_a_started_journal_and_purge_parses_every_layout(mini, monkeypatch):
    rec, cr, v = mini
    monkeypatch.setattr(vc, "live_jobs", lambda: set())
    _accept_tracking(v, 0)
    rd = v / "runs" / "0"
    adir = ai.attempt_root(v, "digitization", 0, 2)
    (adir / "0").mkdir(parents=True)
    (adir / "0" / "sim_with_tracks.root").write_text("new sim")
    rp.write_journal(adir, {"state": "started", "moves": [[str(adir / "0" / "sim_with_tracks.root"),
                                                              str(rd / "sim_with_tracks.root"),
                                                              str(adir / "replaced" / "sim_with_tracks.root")]],
                            "manifest": None, "manifest_backup": None})
    (adir / "replaced").mkdir()
    os.rename(rd / "sim_with_tracks.root", adir / "replaced" / "sim_with_tracks.root")   # killed after the backup
    _rec_file(rd / "provenance", "digitization", 2, status="promoting", return_code=None)
    # an old-layout packaging backup and a token that starts with "a"
    (v / ".attempts" / "package_parquet" / "events0-2.a1b2c3").mkdir(parents=True)
    assert vc.reconcile(rec, "full_pileup", apply=True, purge_failed=True) in (0, 1)
    assert (rd / "sim_with_tracks.root").read_text() == "sim 0", "previous generation back in place"
    assert rp.read_journal(adir)["state"] == "rolled_back"
    assert json.loads((rd / "provenance" / "digitization_attempt2.json").read_text())["status"] == "interrupted"
    assert (v / ".attempts" / "package_parquet" / "events0-2.a1b2c3").exists(), "never parsed as an attempt, kept"


def test_d_reconcile_serialized_with_writers(mini, monkeypatch):
    rec, cr, v = mini
    pd = v / "provenance" / "package_parquet"
    pd.mkdir(parents=True)
    (pd / "chunk1.lock").write_text(json.dumps({"host": "nidX", "pid": 1, "slurm_job_id": "424242"}))
    tmp = v / "parquet" / "reco" / "tracks" / ".full_pileup.ttbar.v23.reco.tracks.events3-5.parquet.abc.tmp"
    tmp.parent.mkdir(parents=True)
    tmp.write_text("partial")
    # first query: job gone (stale); second query right before the reclaim: the job is live again
    monkeypatch.setattr(vc, "live_jobs", lambda: set())
    monkeypatch.setattr(vc, "_job_running", lambda jid: True)
    assert vc.reconcile(rec, "full_pileup", apply=True, purge_failed=False) == 1
    assert (pd / "chunk1.lock").exists() and tmp.exists(), "a newly live job keeps its lock and its temporary bytes"
    # a writer cannot take a unit lock while reconcile holds the control lock, and two reconcilers exclude each other
    ctl = rp.ControlLock(v, "test reconcile", timeout_s=0)
    assert ctl.acquire()
    try:
        assert not rp.ControlLock(v, "writer", timeout_s=0).acquire()
        monkeypatch.setenv("COLLIDERML_RECONCILE_WAIT_S", "0")
        assert vc.reconcile(rec, "full_pileup", apply=True, purge_failed=False) == 1
        assert (pd / "chunk1.lock").exists() and tmp.exists()
    finally:
        ctl.release()
    # the stale lock is unlinked only if byte-identical to the one judged stale (no rename-and-restore)
    monkeypatch.setattr(vc, "_job_running", lambda jid: False)
    assert vc.reconcile(rec, "full_pileup", apply=True, purge_failed=False) == 0
    assert not (pd / "chunk1.lock").exists() and not tmp.exists()
    assert not list(pd.glob("*.reclaim.*"))


def _report(rec, cr, i, shards, **meta_over):
    w = list(vc.chunk_window(i, cr["chunk_size"], cr["release_events"]))
    cdir = (Path(rec["verifier"]["dir"]) / "contracts").resolve()
    checks_ = ["chunk_view", "inventory", "event_identity", "schema", "links", "coverage", "shared_hits", "track_numeric_d6"]
    meta = {"cmd": {"chunk": i, "prefix": cr["prefix"]}, "window": w, "launch_digest": rec["digest"],
            "verifier_sha256": rec["verifier"]["digests"], "objects": cr["objects"] + cr["carried_objects"],
            "carried": cr["carried_objects"], "contracts_dir": str(cdir),
            "contracts_sha256": {"files": {"particles_pu200.json": "c" * 64}}, "summary_sha256": "s" * 64,
            "shards": shards} | meta_over
    return {"meta": meta, "all_pass": True, "results": [{"check": c, "pass": True} for c in checks_]}


def test_a_verification_report_must_match_the_frozen_contract(tmp_path, mini):
    rec, cr, v = mini
    rec["digest"] = "L" * 64
    rec["verifier"] = {"dir": str(tmp_path / "verify"), "digests": {"contracts/particles_pu200.json": "c" * 64, "checks.py": "d" * 64}}
    cr["prefix"] = "full_pileup.ttbar.v23"
    cr["carried_objects"] = ["particles"]
    carried = tmp_path / "carried.events3-5.parquet"
    carried.write_text("v1 truth")
    cr["carried"] = {"particles": {str(carried): vc.stat_of(carried)}}
    shard = "/x/full_pileup.ttbar.v23.reco.tracks.events3-5.parquet"
    good = _report(rec, cr, 1, {shard: "h" * 64, os.path.realpath(carried): "t" * 64})
    assert vc.verification_report_problems(rec, cr, 1, good, {shard: "h" * 64}) == []
    # Codex: omitting carried particles from --objects removes the truth lookup that found D8
    no_truth = _report(rec, cr, 1, {shard: "h" * 64}, objects=["tracks"], carried=[])
    assert vc.verification_report_problems(rec, cr, 1, no_truth, {shard: "h" * 64})
    other_c = _report(rec, cr, 1, {shard: "h" * 64, os.path.realpath(carried): "t" * 64}, contracts_dir="/elsewhere")
    assert any("contracts dir" in x for x in vc.verification_report_problems(rec, cr, 1, other_c, {shard: "h" * 64}))
    edited_c = _report(rec, cr, 1, {shard: "h" * 64, os.path.realpath(carried): "t" * 64},
                       contracts_sha256={"files": {"particles_pu200.json": "0" * 64}})
    assert any("contract hashes" in x for x in vc.verification_report_problems(rec, cr, 1, edited_c, {shard: "h" * 64}))
    no_sum = _report(rec, cr, 1, {shard: "h" * 64, os.path.realpath(carried): "t" * 64}, summary_sha256=None)
    assert any("summary" in x for x in vc.verification_report_problems(rec, cr, 1, no_sum, {shard: "h" * 64}))
    missing_check = dict(good, results=[r for r in good["results"] if r["check"] != "links"])
    assert any("links absent" in x for x in vc.verification_report_problems(rec, cr, 1, missing_check, {shard: "h" * 64}))
    cr["objects"] = ["tracks", "pfos"]
    no_res = _report(rec, cr, 1, {shard: "h" * 64, os.path.realpath(carried): "t" * 64})
    assert any("resources" in x for x in vc.verification_report_problems(rec, cr, 1, no_res, {shard: "h" * 64}))


def test_a_p5_and_decisions_bind(tmp_path):
    rec = {"digest": "L" * 64, "verifier": {"digests": {"a.py": "1" * 64}},
           "frozen_tracking_sample": {"full_pileup": [0, 50, 1562]}}
    tb = tmp_path / "tracking_bound.json"
    good = {"all_pass": True, "meta": {"campaign": "full_pileup", "runs": [0, 50, 1562], "launch_digest": "L" * 64,
                                        "verifier_sha256": {"a.py": "1" * 64}},
            "results": [{"check": "production_tracking_bound", "pass": True, "info": {"n_conclusive_p5_bins": 7}}]}
    tb.write_text(json.dumps(good))
    assert vc.tracking_bound_problems(rec, "full_pileup", tb) == []
    # Codex: any JSON with all_pass true passed as P5 (e.g. a copied chunk report)
    tb.write_text(json.dumps({"all_pass": True, "meta": {"launch_digest": "L" * 64}, "results": [{"check": "links", "pass": True}]}))
    assert vc.tracking_bound_problems(rec, "full_pileup", tb)
    zero = json.loads(json.dumps(good)); zero["results"][0]["info"]["n_conclusive_p5_bins"] = 0
    tb.write_text(json.dumps(zero))
    assert any("conclusive" in x for x in vc.tracking_bound_problems(rec, "full_pileup", tb))
    one_run = json.loads(json.dumps(good)); one_run["meta"]["runs"] = [1562]
    tb.write_text(json.dumps(one_run))
    assert any("frozen sample" in x for x in vc.tracking_bound_problems(rec, "full_pileup", tb))
    # Codex: any readable file satisfied --decision
    plan = tmp_path / "PRODUCTION_PLAN_v3.md"
    plan.write_text("unsigned plan")
    assert vc.decision_problems(rec, [str(plan)])[1], "a decision the launch record does not name"
    rec["approved_decision_records"] = {str(plan): rp.sha256_file(plan)}
    assert vc.decision_problems(rec, [str(plan)])[1] == []
    plan.write_text("edited plan")
    assert any("changed" in x for x in vc.decision_problems(rec, [str(plan)])[1])


def test_i_budget_zero_limit_failed_report_and_nonterminal_states(mini, monkeypatch, tmp_path):
    with pytest.raises(ValueError):
        vc.parse_time_limit("00:00:00")
    with pytest.raises(ValueError):
        vc.parse_time_limit("1:00")
    assert vc.parse_time_limit("01:30:00") == 1.5
    rec, cr, v = mini
    rec["digest"] = "L" * 64
    rec["verifier"] = {"dir": str(tmp_path), "digests": {}}
    cr["prefix"] = "full_pileup.ttbar.v23"
    cr["verification_dir"] = str(tmp_path / "verification")
    (tmp_path / "verification").mkdir()
    (tmp_path / "verification" / "chunk0.json").write_text(json.dumps({"all_pass": False, "meta": {}, "results": []}))
    (tmp_path / "verification" / "chunk1.json").write_text("{truncated")
    assert vc.verification_current(rec, cr, 0)[0] is False, "a failed report does not retire the forecast"
    assert vc.verification_current(rec, cr, 1)[0] is False, "nor an unreadable one"
    rec["budget"] = {"cap_node_hours": 100, "retry_allowance": 0.0, "ledger": str(tmp_path / "ledger.jsonl"),
                     "rates_node_hours_per_unit": {"full_pileup": {s: 1.0 for s in vc.STAGES + ["verify"]}}}
    entries = [{"kind": "reservation", "reservation": "r1", "nodes": 10, "time_limit_h": 1.0},
               {"kind": "submitted", "reservation": "r1", "job_id": "123"}]
    monkeypatch.setattr(vc, "live_jobs", lambda: set())
    for state, charged in (("COMPLETING", 10.0), ("CONFIGURING", 10.0), ("COMPLETED", 0.5)):
        monkeypatch.setattr(vc, "_sacct", lambda ids, st=state: {"123": {"state": st, "elapsed_s": 180, "nodes": 10}})
        ok, rep_ = vc.budget(rec, 0.0, verbose=False, entries=entries)
        assert rep_["consumed"] == charged, state
    assert rep_["detail"]["full_pileup/verify"]["units_left"] == 2


def test_g_publish_interruptions_manifest_bytes_and_listing(tmp_path, monkeypatch):
    rec, v, f, man, mp, root = _pub_fixture(tmp_path, monkeypatch)
    (v / "README.md").write_text("dataset card")
    (v / "configs").mkdir()
    (v / "configs" / "package.yaml").write_text("cfg")
    man["meta_files"] = {"README.md": rp.sha256_file(v / "README.md"), "configs/package.yaml": rp.sha256_file(v / "configs/package.yaml")}
    man["digest"] = vc._manifest_digest(man)
    mp.write_text(json.dumps(man))
    raw = mp.read_bytes()
    tgt = root / "hard_scatter" / "ttbar" / "v23"
    # the manifest file is changed after validation: the validated bytes are the ones published
    real_check = vc._check_tree_against

    def swap(*a, **k):
        mp.write_text(json.dumps(dict(man, written_utc="later")))
        return real_check(*a, **k)
    monkeypatch.setattr(vc, "_check_tree_against", swap)
    assert vc.release_publish(rec, "hard_scatter", mp, root, 0, 2) == 0
    monkeypatch.setattr(vc, "_check_tree_against", real_check)
    assert (tgt / "RELEASE_MANIFEST.json").read_bytes() == raw
    assert (tgt / "README.md").read_text() == "dataset card" and (tgt / "configs" / "package.yaml").exists()
    listing = {r: m["sha256"] for r, m in man["files"].items()}
    lp = tmp_path / "listing.json"
    lp.write_text(json.dumps(listing))
    assert vc.release_verify_bytes(rec, "hard_scatter", root, str(lp), 2) == 0
    lp.write_text(json.dumps(dict(listing, **{"reco/tracks/stray.parquet": "x"})))
    assert vc.release_verify_bytes(rec, "hard_scatter", root, str(lp), 2) == 1, "an extra portal entry fails"
    assert vc.release_rollback(rec, "hard_scatter", root) == 0
    # killed after mkdir of the hidden dir and before .published: found through the intent file, rolled back
    mp.write_bytes(raw)
    real_mkdir = os.mkdir

    def mkdir_then_die(p_, *a):          # killed right after mkdir of the hidden dir, before .published
        real_mkdir(p_, *a)
        if ".publishing." in str(p_):
            raise KeyboardInterrupt("killed")
    monkeypatch.setattr(vc.os, "mkdir", mkdir_then_die)
    with pytest.raises(KeyboardInterrupt):
        vc.release_publish(rec, "hard_scatter", mp, root, 0, 2)
    monkeypatch.setattr(vc.os, "mkdir", real_mkdir)
    hidden = [p_ for p_ in tgt.parent.glob(".v23.publishing.*") if p_.is_dir()]
    intents = list(tgt.parent.glob(".v23.publishing.*.intent"))
    assert hidden and intents and not (hidden[0] / ".published").exists(), "the kill left a hidden dir without .published"
    assert vc.release_preflight.__name__  # (preflight is stubbed in this fixture)
    assert vc.release_rollback(rec, "hard_scatter", root) == 0
    assert not [p_ for p_ in tgt.parent.glob(".v23.publishing.*")], "no skeleton or intent left"
    assert f.exists()


def test_d8_truth_base_is_frozen_and_bound_by_sha(mini, tmp_path):
    rec, cr, v = mini
    for r in (0, 1):
        _accept_tracking(v, r)
    base = tmp_path / "v1truth"
    base.mkdir()
    for a_, b_ in ((0, 2), (3, 5)):
        (base / f"full_pileup.ttbar.v1.truth.particles.events{a_}-{b_}.parquet").write_text(f"v1 {a_}")
        os.chmod(base / f"full_pileup.ttbar.v1.truth.particles.events{a_}-{b_}.parquet", 0o444)
    tb, probs = vc._truth_base({"truth_base": {"object": "particles_completed", "dir": str(base)}}, True)
    assert probs == [] and all("sha256" in m for m in tb["shards"].values())
    cr["truth_base"] = tb
    cr["objects"] = ["particles_completed"]
    cr["package_flags"] = {"remap_particle_ids": True}
    assert vc._layout(cr, "particles_completed") == ("truth", "particles")
    rd = {r: v / "runs" / str(r) for r in (0, 1)}
    inp = {}
    for r in (0, 1):
        (rd[r] / "particles").mkdir()
        (rd[r] / "particles" / "p.parquet").write_text(f"particles {r}")
        pth = sorted((rd[r] / "provenance").glob("digitization_attempt*.json"))[-1]
        r_ = json.loads(pth.read_text()); r_["outputs"].append(_out(rd[r], "particles/p.parquet")); pth.write_text(json.dumps(r_))
        e = _out(rd[r], "particles/p.parquet")
        inp[str(rd[r] / "particles/p.parquet")] = {"realpath": e["realpath"], "size": e["size"], "sha256": e["sha256"]}
        # the ACTS particles.root (perigee of the added rows; r4 rehearsal: missing from the expected set)
        (rd[r] / "particles.root").write_text(f"acts particles {r}")
        r_ = json.loads(pth.read_text()); r_["outputs"].append(_out(rd[r], "particles.root")); pth.write_text(json.dumps(r_))
        e = _out(rd[r], "particles.root")
        inp[str(rd[r] / "particles.root")] = {"realpath": e["realpath"], "size": e["size"], "sha256": e["sha256"]}
        st = vc.stat_of(rd[r] / "edm4hep.root")
        inp[str(rd[r] / "edm4hep.root")] = {k: st[k] for k in ("realpath", "size", "mtime_ns", "inode")} | \
            {"sha256": cr["parents"][str(r)]["edm4hep.root"]["files"]["."]["sha256"]}
    f35 = str(base / "full_pileup.ttbar.v1.truth.particles.events3-5.parquet")
    m35 = tb["shards"][f35]
    good = dict(inp, **{"/any/name/the/packager/used.parquet": {"realpath": m35["realpath"], "size": m35["size"],
                                                               "inode": m35["inode"], "sha256": m35["sha256"]}})
    pd = v / "provenance" / "package_parquet"
    _rec_file(pd, "chunk1", 1, container_record={"start_event": 3, "end_event": 5, "inputs": good, "outputs": {}})
    assert vc.audit_unit(rec, cr, "package_parquet", 1, set(), {}) == []
    _rec_file(pd, "chunk1", 2, container_record={"start_event": 3, "end_event": 5, "inputs": inp, "outputs": {}})
    assert any("not in the manifest" in x for x in vc.audit_unit(rec, cr, "package_parquet", 1, set(), {})), "base shard missing"
    bad = json.loads(json.dumps(good)); bad["/any/name/the/packager/used.parquet"]["sha256"] = "0" * 64
    _rec_file(pd, "chunk1", 3, container_record={"start_event": 3, "end_event": 5, "inputs": bad, "outputs": {}})
    assert any("truth base" in x for x in vc.audit_unit(rec, cr, "package_parquet", 1, set(), {}))
    nopr = {k: m for k, m in good.items() if not k.endswith("particles.root")}
    _rec_file(pd, "chunk1", 4, container_record={"start_event": 3, "end_event": 5, "inputs": nopr, "outputs": {}})
    assert any("not in the manifest" in x for x in vc.audit_unit(rec, cr, "package_parquet", 1, set(), {})), \
        "particles.root missing"
    assert vc.upstream_problems(rec, cr, "package_parquet", 1, set(), {}) == []


def test_a_finalizer_needs_the_launch_record_in_production(tmp_path):
    base = tmp_path / "base"
    v = base / "hard_scatter" / "ttbar" / "v23"
    v.mkdir(parents=True)
    (v / ".provenance_required").write_text("")
    cfg = tmp_path / "pkg.yaml"
    cfg.write_text(yaml.safe_dump({"campaign": "hard_scatter", "dataset": "ttbar", "version": "v23",
                                   "common": {"output_base_dir": str(base)}, "release_events": 10,
                                   "chunk_size": 10, "objects": ["tracks"]}))
    out = subprocess.run([sys.executable, str(REPO / "scripts/production/finalize_version.py"), str(cfg),
                          "--approved-sha", "0" * 40], capture_output=True, text=True)
    assert out.returncode == 1 and "--launch-record is required" in out.stdout
    assert not (v / ".freeze").exists() and not (v / ".git_commit_success").exists()
    (v / rp.CONTROL_LOCK).write_text("{}")           # a writer or reconcile holds the version
    (v / ".provenance_required").unlink()
    out = subprocess.run([sys.executable, str(REPO / "scripts/production/finalize_version.py"), str(cfg),
                          "--approved-sha", "0" * 40, "--dry-run"], capture_output=True, text=True,
                         env=dict(os.environ, COLLIDERML_CONTROL_WAIT_S="0"))
    assert out.returncode == 1 and "control.lock" in out.stdout and not (v / ".freeze").exists()
