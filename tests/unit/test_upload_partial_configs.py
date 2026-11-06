"""HF upload: a config folder that holds fewer shards than local files is re-uploaded, not skipped.

dihiggs_pu200_tracker_hits kept 295 of 1000 shards after an interrupted upload; the old check only
looked for the folder (ColliderML-Production#53).
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "dataset"))

upload = pytest.importorskip("upload_to_hf_unified")


class FakeApi:
    def __init__(self, files):
        self.files = files

    def list_repo_files(self, repo_id, repo_type):
        return self.files


def local_config(tmp_path, name, n_files):
    d = tmp_path / name
    d.mkdir()
    for i in range(n_files):
        (d / f"{name}.events{i}.parquet").write_bytes(b"")
    return {"config_name": name, "path": d}


def test_existing_configs_count_shards(monkeypatch):
    files = ["README.md", "data/a/train-00000-of-00002.parquet", "data/a/train-00001-of-00002.parquet",
             "data/b/train-00000-of-00003.parquet", "data/b/notes.txt"]
    monkeypatch.setattr(upload, "HfApi", lambda token=None: FakeApi(files))
    assert upload.get_existing_configs("repo", "token") == {"a": 2, "b": 1}


def test_partial_config_is_not_complete(tmp_path):
    cfg = local_config(tmp_path, "dihiggs_pu200_tracker_hits", 4)
    assert not upload.is_complete_on_hub(cfg, {"dihiggs_pu200_tracker_hits": 1})
    assert not upload.is_complete_on_hub(cfg, {})
    assert upload.is_complete_on_hub(cfg, {"dihiggs_pu200_tracker_hits": 4})


def test_empty_local_config_is_never_complete(tmp_path):
    cfg = local_config(tmp_path, "empty", 0)
    assert not upload.is_complete_on_hub(cfg, {"empty": 0})
