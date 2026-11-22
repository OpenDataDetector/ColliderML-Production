"""finalize_version: the marker is written only for a complete, unchanged, approved release candidate."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "scripts" / "production" / "finalize_version.py"
SHA = "a" * 40


def _setup(tmp_path, attempts=2):
    cfg = tmp_path / "package_parquet_config.yaml"
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg.write_text(yaml.safe_dump({"campaign": "c", "dataset": "d", "version": "v23", "release_events": 250,
                                   "chunk_size": 100, "objects": ["tracks", "vertices"],
                                   "common": {"output_base_dir": str(tmp_path)}}))
    v = tmp_path / "c" / "d" / "v23"
    pdir = v / "provenance" / "package_parquet"
    pdir.mkdir(parents=True)
    for i, (s, e) in enumerate([(0, 99), (100, 199), (200, 249)]):
        outs = {}
        for obj in ("tracks", "vertices"):
            f = v / "parquet" / "reco" / obj / f"c.d.v23.reco.{obj}.events{s}-{e}.parquet"
            f.parent.mkdir(parents=True, exist_ok=True)
            import pyarrow as pa
            import pyarrow.parquet as pq
            pq.write_table(pa.table({"event_id": pa.array(list(range(s, e + 1)), type=pa.uint32())}), f)
            outs[str(f)] = {"size": f.stat().st_size, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        man = {"status": "ok", "start_event": s, "end_event": e, "objects": ["tracks", "vertices"], "outputs": outs}
        for k in range(1, attempts):
            (pdir / f"chunk{i}_attempt{k}.json").write_text(json.dumps({"status": "incomplete", "return_code": 4}))
        (pdir / f"chunk{i}_attempt{attempts}.json").write_text(json.dumps(
            {"status": "complete", "return_code": 0, "git": {"sha": SHA}, "container_record": man}))
    return cfg, v


def _run(cfg, sha=SHA):
    return subprocess.run([sys.executable, str(TOOL), str(cfg), "--approved-sha", sha], capture_output=True, text=True)


def test_intact_candidate_writes_once_readonly(tmp_path):
    cfg, v = _setup(tmp_path)
    r = _run(cfg)
    assert r.returncode == 0, r.stdout
    m = v / ".git_commit_success"
    assert m.exists() and (m.stat().st_mode & 0o222) == 0
    assert _run(cfg).returncode == 1, "never overwritten"


def test_deleted_or_changed_shard_fails(tmp_path):
    cfg, v = _setup(tmp_path / "a")
    next((v / "parquet/reco/tracks").glob("*events100-*")).unlink()
    assert _run(cfg).returncode == 1 and not (v / ".git_commit_success").exists()
    cfg2, v2 = _setup(tmp_path / "b")
    f = next((v2 / "parquet/reco/vertices").glob("*events200-*"))
    f.write_bytes(b"x" * f.stat().st_size)  # same size, different bytes
    assert _run(cfg2).returncode == 1 and not (v2 / ".git_commit_success").exists()


def test_later_unresolved_attempt_and_numeric_order(tmp_path):
    cfg, v = _setup(tmp_path, attempts=9)
    (v / "provenance" / "package_parquet" / "chunk1_attempt10.json").write_text(json.dumps({"status": "started"}))
    r = _run(cfg)
    assert r.returncode == 1 and "attempt 10" in r.stdout, "attempt 10 is later than attempt 9"


def test_wrong_sha_and_stray_file_fail(tmp_path):
    cfg, v = _setup(tmp_path)
    assert _run(cfg, sha="b" * 40).returncode == 1
    (v / "parquet/reco/tracks/c.d.v23.reco.tracks.events250-349.parquet").write_bytes(b"x")
    r = _run(cfg)
    assert r.returncode == 1 and "unexpected file" in r.stdout


def test_manifest_naming_another_chunks_shard_fails(tmp_path):
    """Codex round 3: both manifests authenticate chunk 0, chunk 1's bytes are never read."""
    cfg, v = _setup(tmp_path)
    pdir = v / "provenance" / "package_parquet"
    rec0 = json.loads((pdir / "chunk0_attempt2.json").read_text())
    rec1 = json.loads((pdir / "chunk1_attempt2.json").read_text())
    rec1["container_record"]["outputs"] = rec0["container_record"]["outputs"]
    (pdir / "chunk1_attempt2.json").write_text(json.dumps(rec1))
    next((v / "parquet/reco/tracks").glob("*events100-*")).write_bytes(b"corrupt")
    r = _run(cfg)
    assert r.returncode == 1 and "not exactly this chunk" in r.stdout


def test_empty_object_list_fails(tmp_path):
    cfg, v = _setup(tmp_path)
    c = yaml.safe_load(cfg.read_text()); c["objects"] = []
    cfg.write_text(yaml.safe_dump(c))
    assert _run(cfg).returncode == 1
