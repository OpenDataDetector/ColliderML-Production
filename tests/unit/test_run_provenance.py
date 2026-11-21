"""Provenance records, the pushed-commit gate, and run_stage exiting with the task's return code."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "cli"))
rp = pytest.importorskip("run_provenance")


def _git(repo, *a):
    subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin", "HOME": str(repo)})


@pytest.fixture
def repos(tmp_path):
    origin = tmp_path / "origin.git"
    work = tmp_path / "work"
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    subprocess.run(["git", "init", "-q", str(work)], check=True)
    (work / "a.txt").write_text("a\n")
    _git(work, "add", "a.txt")
    _git(work, "commit", "-q", "-m", "a")
    _git(work, "remote", "add", "origin", str(origin))
    return origin, work


def test_gate_unpushed_dirty_pushed(repos):
    origin, work = repos
    err = rp.pushed_commit_error(rp.git_state(work))
    assert err and "no remote-tracking branch" in err
    _git(work, "push", "-q", "origin", "HEAD:refs/heads/release/v23")
    _git(work, "fetch", "-q", "origin")
    st = rp.git_state(work)
    assert rp.pushed_commit_error(st) is None
    assert any("release/v23" in r for r in st["remote_branches_containing_head"])
    (work / "a.txt").write_text("changed\n")
    assert "dirty" in rp.pushed_commit_error(rp.git_state(work))
    (work / "a.txt").write_text("a\n")
    (work / "new.py").write_text("x = 1\n")
    assert "dirty" in rp.pushed_commit_error(rp.git_state(work)), "untracked code counts as dirty"


def test_redacted_sha_ignores_secret_lines(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("x: 1\ntoken: abc\n")
    b.write_text("x: 1\ntoken: different\n")
    assert rp.redacted_sha256(a) == rp.redacted_sha256(b)
    b.write_text("x: 2\ntoken: abc\n")
    assert rp.redacted_sha256(a) != rp.redacted_sha256(b)
    assert rp.redacted_sha256(tmp_path / "missing.yaml") is None


def test_hash_paths_and_record(tmp_path, repos):
    _, work = repos
    run = tmp_path / "runs" / "3"
    (run / "tracks").mkdir(parents=True)
    (run / "tracks" / "t.parquet").write_bytes(b"xyz")
    (run / "digi_provenance.json").write_text(json.dumps({"seed_finder_bFieldInZ_T": 3.0}))
    target = tmp_path / "edm.root"
    target.write_bytes(b"edm")
    (run / "edm4hep.root").symlink_to(target)
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("stage: digitization\n")
    config = {"stage": "digitization", "campaign": "c", "dataset": "d", "version": "v23",
              "provenance": {"record": True, "inputs": ["edm4hep.root"],
                             "outputs": ["tracks/*.parquet", "missing.root"],
                             "container_json": "digi_provenance.json"}}
    rec = rp.build_record(config=config, config_path=cfg_file, unit=3, rc=0, t_start=0, t_end=1,
                          repo=work, base=run, container=None, env_setup_path=tmp_path / "none.yaml")
    assert rec["inputs"][0]["is_link"] and rec["inputs"][0]["realpath"] == str(target)
    assert rec["inputs"][0]["sha256"] == rp.sha256_file(target)
    assert rec["outputs"][0]["size"] == 3
    assert rec["outputs"][1] == {"pattern": "missing.root", "missing": True}
    assert rec["container_record"]["seed_finder_bFieldInZ_T"] == 3.0
    assert rec["return_code"] == 0 and rec["git"]["sha"]
    p1 = rp.record_path(config, tmp_path / "runs", tmp_path, 3, postprocessing=False)
    rp.write_record(p1, rec)
    p2 = rp.record_path(config, tmp_path / "runs", tmp_path, 3, postprocessing=False)
    assert p1.name == "digitization_attempt1.json" and p2.name == "digitization_attempt2.json"
    assert json.loads(p1.read_text())["unit"] == 3


def test_run_stage_exits_with_task_code(tmp_path, monkeypatch):
    """Validation on and the guardian says OK: the task's own non-zero code must still win."""
    rs = pytest.importorskip("run_stage")
    import cli_utils
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    monkeypatch.setattr(cli_utils, "create_necessary_directories", lambda c: {"run_dir": run_dir})
    monkeypatch.setattr(cli_utils, "build_stage_command",
                        lambda **kw: {"full_command": "exit 7", "use_shifter": False, "env_setup_commands": ["true"]})
    monkeypatch.setattr(rs, "run_validation", lambda *a, **k: {"status": "OK"})
    monkeypatch.setattr(rs, "run_guardian", lambda *a, **k: {"action": "OK", "exit_code": 0, "reason": "t"})
    config = {"stage": "digitization", "campaign": "c", "dataset": "d", "version": "v23",
              "validation_config": {"enabled": True}}
    with pytest.raises(SystemExit) as e:
        rs.run_interactive(config, tmp_path / "c.yaml", tmp_path / "s.py", run_id=0)
    assert e.value.code == 7
    # and the guardian's failure still fails a successful task
    monkeypatch.setattr(cli_utils, "build_stage_command",
                        lambda **kw: {"full_command": "true", "use_shifter": False, "env_setup_commands": ["true"]})
    monkeypatch.setattr(rs, "run_guardian", lambda *a, **k: {"action": "FAIL", "exit_code": 1, "reason": "t"})
    with pytest.raises(SystemExit) as e:
        rs.run_interactive(config, tmp_path / "c.yaml", tmp_path / "s.py", run_id=0)
    assert e.value.code == 1


def test_run_stage_writes_record_for_failed_attempt(tmp_path, monkeypatch):
    rs = pytest.importorskip("run_stage")
    import cli_utils
    run_dir = tmp_path / "runs"
    (run_dir / "5").mkdir(parents=True)
    monkeypatch.setattr(cli_utils, "create_necessary_directories", lambda c: {"run_dir": run_dir})
    monkeypatch.setattr(cli_utils, "get_version_directory", lambda c: tmp_path)
    monkeypatch.setattr(cli_utils, "build_stage_command",
                        lambda **kw: {"full_command": "exit 2", "use_shifter": False, "env_setup_commands": ["true"]})
    cfg = tmp_path / "c.yaml"
    cfg.write_text("x: 1\n")
    config = {"stage": "digitization", "campaign": "c", "dataset": "d", "version": "v23",
              "validation_config": {"enabled": False}, "provenance": {"record": True, "inputs": ["edm4hep.root"]}}
    with pytest.raises(SystemExit) as e:
        rs.run_interactive(config, cfg, tmp_path / "s.py", run_id=5)
    assert e.value.code == 2
    rec = json.loads((run_dir / "5" / "provenance" / "digitization_attempt1.json").read_text())
    assert rec["return_code"] == 2 and rec["inputs"][0]["missing"] is True
