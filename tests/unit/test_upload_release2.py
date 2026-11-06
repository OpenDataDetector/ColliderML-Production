"""HF upload, Release 2: per-dataset versions, truth/reco object layout, and the per-file schema gate."""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "dataset"))

upload = pytest.importorskip("upload_to_hf_unified")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def write(path, table):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def test_object_subdir():
    assert upload.object_subdir("particles") == "truth/particles"
    assert upload.object_subdir("tracker_simhits") == "truth/tracker_simhits"
    assert upload.object_subdir("pfos") == "reco/pfos"


def test_dataset_version_overrides_default(tmp_path):
    t = pa.table({"event_id": pa.array([0], pa.uint32())})
    write(tmp_path / "hard_scatter/ttbar/v22/parquet/reco/pfos/a.events0-0.parquet", t)
    write(tmp_path / "hard_scatter/ggf/v1/parquet/reco/pfos/a.events0-0.parquet", t)
    config = {
        "data": {"base_dir": str(tmp_path), "version": "v1", "format_subdir": "parquet"},
        "object_types": ["pfos"],
        "campaigns": [{"campaign_name": "hard_scatter", "pileup": 0, "pileup_label": "pu0",
                       "datasets": [{"name": "ttbar", "version": "v22"}, {"name": "ggf"}]}],
    }
    got = {c["config_name"]: c["version"] for c in upload.build_config_list(config)}
    assert got == {"ttbar_pu0_pfos": "v22", "ggf_pu0_pfos": "v1"}


def test_schema_gate(tmp_path):
    good = pa.table({"event_id": pa.array([0, 1], pa.uint32()), "x": [1.0, 2.0]})
    write(tmp_path / "f0.parquet", good)
    write(tmp_path / "f1.parquet", good)
    assert upload.validate_parquet_files([tmp_path / "f0.parquet", tmp_path / "f1.parquet"], "c")

    write(tmp_path / "drift.parquet", pa.table({"event_id": pa.array([2], pa.uint32())}))
    assert not upload.validate_parquet_files([tmp_path / "f0.parquet", tmp_path / "drift.parquet"], "c")

    write(tmp_path / "empty.parquet", good.slice(0, 0))
    assert not upload.validate_parquet_files([tmp_path / "f0.parquet", tmp_path / "empty.parquet"], "c")

    write(tmp_path / "noid.parquet", pa.table({"x": [1.0]}))
    assert not upload.validate_parquet_files([tmp_path / "noid.parquet"], "c")


def test_readme_path_relative_to_config(tmp_path):
    cfg = tmp_path / "sub" / "config.yaml"
    assert upload.resolve_relative("CARD.md", str(cfg)) == tmp_path / "sub" / "CARD.md"
    assert upload.resolve_relative("/abs/CARD.md", str(cfg)) == Path("/abs/CARD.md")
