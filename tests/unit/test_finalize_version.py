"""finalize_version: the marker is written only after every release chunk has an accepted record."""
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "scripts" / "production" / "finalize_version.py"


def _setup(tmp_path, n_ok, sha2=None):
    cfg = tmp_path / "package_parquet_config.yaml"
    cfg.write_text(yaml.safe_dump({"campaign": "c", "dataset": "d", "version": "v23", "release_events": 250,
                                   "chunk_size": 100, "common": {"output_base_dir": str(tmp_path)}}))
    pdir = tmp_path / "c" / "d" / "v23" / "provenance" / "package_parquet"
    pdir.mkdir(parents=True)
    for i in range(n_ok):
        sha = sha2 if (sha2 and i == n_ok - 1) else "a" * 40
        (pdir / f"chunk{i}_attempt1.json").write_text(json.dumps({"status": "incomplete", "return_code": 4}))
        (pdir / f"chunk{i}_attempt2.json").write_text(json.dumps({"status": "complete", "return_code": 0,
                                                                  "git": {"sha": sha}}))
        (pdir / f"chunk{i}_inputs.json").write_text(json.dumps({"status": "ok"}))
    return cfg, tmp_path / "c" / "d" / "v23" / ".git_commit_success"


def _run(cfg):
    return subprocess.run([sys.executable, str(TOOL), str(cfg)], capture_output=True, text=True)


def test_missing_chunk_writes_nothing(tmp_path):
    cfg, marker = _setup(tmp_path, 2)          # 3 chunks expected (250 events / 100)
    r = _run(cfg)
    assert r.returncode == 1 and "chunk 2" in r.stdout and not marker.exists()


def test_mixed_sha_refused(tmp_path):
    cfg, marker = _setup(tmp_path, 3, sha2="b" * 40)
    assert _run(cfg).returncode == 1 and not marker.exists()


def test_complete_writes_once_readonly(tmp_path):
    cfg, marker = _setup(tmp_path, 3)
    assert _run(cfg).returncode == 0
    assert marker.exists() and (marker.stat().st_mode & 0o222) == 0
    assert _run(cfg).returncode == 1, "never overwritten"
