"""Offline checks for the reco_tables stage wiring (no data, no container)."""

import importlib.util
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_stage_is_registered_everywhere():
    cli = _load("cli_utils_under_test", "scripts/cli/cli_utils.py")
    assert cli.STAGE_SCRIPT_MAP["reco_tables"] == "postprocessing/convert_reco_tables.py"
    assert (REPO / "scripts" / cli.STAGE_SCRIPT_MAP["reco_tables"]).is_file()
    assert "reco_tables" in cli.SIMULATION_STAGES, "must be per-run (simulation-shaped) for --output-subdir"
    assert "reco_tables" in cli.SHIFTER_STAGES, "runs in the reco image"
    assert "reco_tables" not in cli.POSTPROCESSING_STAGES


def test_stage_container_override_is_honoured():
    cli = _load("cli_utils_under_test2", "scripts/cli/cli_utils.py")
    cfg = {"common": {"container": "sim:1", "container_tarball": "/sim.tar",
                      "stage_containers": {"reco_tables": {"container": "reco:1",
                                                           "container_tarball": "/reco.tar"}}}}
    assert cli.resolve_stage_container(cfg, "reco_tables") == ("reco:1", "/reco.tar")
    assert cli.resolve_stage_container(cfg, "digitization") == ("sim:1", "/sim.tar")


def test_packager_routes_release2_tables():
    sys.path.insert(0, str(REPO / "scripts" / "postprocessing"))
    pkg = _load("package_native_parquet_under_test", "scripts/postprocessing/package_native_parquet.py")
    assert pkg.OBJECT_LAYOUT["calo_cells"] == ("reco", "calo_cells")
    assert pkg.OBJECT_LAYOUT["calo_clusters"] == ("reco", "calo_clusters")
    assert pkg.OBJECT_LAYOUT["pfos"] == ("reco", "pfos")
    for obj in ("calo_cells", "calo_clusters", "pfos"):
        assert obj not in pkg.TIME_COLUMNS, "k4ODD times are already in ns; no mm->ns conversion"


def test_validation_rules_cover_the_stage():
    rules = yaml.safe_load((REPO / "scripts/simulation/validation/validation_rules.yaml").read_text())
    stage = rules["stages"]["reco_tables"] if "stages" in rules else rules["reco_tables"]
    patterns = {p["pattern"]: p for p in stage["file_patterns"]}
    for name in ("calo_cells", "calo_clusters", "pfos"):
        assert f"{name}/*.parquet" in patterns
        assert patterns[f"{name}/*.parquet"]["required"] is True


def test_native_event_count_and_config_defaults(tmp_path):
    sys.path.insert(0, str(REPO / "scripts" / "postprocessing"))
    pytest.importorskip("uproot")
    mod = _load("convert_reco_tables_under_test", "scripts/postprocessing/convert_reco_tables.py")
    run = tmp_path / "0"
    (run / "particles").mkdir(parents=True)
    pq.write_table(pa.table({"event_id": pa.array([0, 1, 2], pa.uint32())}),
                   run / "particles" / "particles_000000-000003.parquet")
    assert mod.event_count_of_native_table(run) == 3
    assert mod.event_count_of_native_table(tmp_path / "missing") is None

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"objects": ["calo_cells"], "events": 5}))
    args = mod.load_config(mod.build_parser().parse_args(["--config", str(cfg), "--output", str(tmp_path)]))
    assert args.objects == ["calo_cells"]
    assert args.events == 5
    assert args.reco_input_name == "reco_edm4hep.root"
    args = mod.load_config(mod.build_parser().parse_args(["--output", str(tmp_path)]))
    assert args.objects == list(mod.ALL_OBJECTS)
    assert args.events == -1

    # missing producer output fails loudly, naming the stage to run
    with pytest.raises(FileNotFoundError, match="pandora_reco"):
        mod.resolve_inputs(run, args)
