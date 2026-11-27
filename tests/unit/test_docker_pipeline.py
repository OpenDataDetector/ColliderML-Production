"""Docker pipeline: stage list, development configs, vertex-link switches, stage timer."""

import csv
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs_development" / "docker_test"
PIPELINE = REPO / "scripts" / "cli" / "run_pipeline_docker.sh"
sys.path.insert(0, str(REPO / "scripts" / "cli"))
sys.path.insert(0, str(REPO / "scripts" / "postprocessing"))

import stage_timer  # noqa: E402


def _load_sim_util(name):
    # by path: scripts/postprocessing/utils is also a package called "utils"
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        f"sim_utils_{name}", REPO / "scripts" / "simulation" / "utils" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


want_vertex_track_links = _load_sim_util("vertex_link_support").want_vertex_track_links

CHANNELS = sorted(p.name for p in CONFIGS.iterdir() if p.is_dir())
COMMON_STAGES = ["simulation", "digitization", "particles_full", "pandora_reco", "reco_tables", "package_parquet"]


def _pipeline(*args):
    return subprocess.run(["bash", str(PIPELINE), *args], capture_output=True, text=True)


def _stage_table(channel):
    """(stage, script, config) rows of a channel, read from the pipeline script."""
    text = PIPELINE.read_text()
    rows = re.findall(r"(?:^|\")([a-z_]+)\|([\w/.]+\.py)\|(\w+\.yaml)", text, flags=re.M)
    names = _pipeline("--channel", channel, "--list-stages").stdout.split()
    by_name = {r[0]: r for r in rows}
    return [by_name[n] for n in names]


def test_channels_present():
    assert CHANNELS == ["higgs_portal", "ttbar"]


@pytest.mark.parametrize("channel,generation", [
    ("ttbar", ["madgraph_init", "madgraph_generation", "pythia_generation"]),
    ("higgs_portal", ["pythia_generation"]),
])
def test_stage_order(channel, generation):
    result = _pipeline("--channel", channel, "--list-stages")
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == generation + COMMON_STAGES


@pytest.mark.parametrize("channel", CHANNELS)
def test_every_stage_has_script_and_config(channel):
    for stage, script, config in _stage_table(channel):
        assert (REPO / "scripts" / script).is_file(), script
        cfg = yaml.safe_load((CONFIGS / channel / config).read_text())
        assert cfg["stage"] == stage
        assert cfg["dataset"] == channel


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*/*.yaml")), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_configs_have_no_site_settings(path):
    text = path.read_text()
    for word in ("/global/", "/pscratch", "nersc", "slurm", "qos", "account", "container_tarball", "job_config"):
        assert word not in text.lower(), f"{path.name} mentions {word!r}"
    for value in re.findall(r"(?<![\w.])/[\w./-]+", re.sub(r"#.*", "", text)):
        assert value.startswith(("/output", "/cache", "/opt/", "/tmp/")), f"{path.name}: host path {value}"


@pytest.mark.parametrize("channel", CHANNELS)
def test_packaging_and_digitisation_agree_on_links(channel):
    digi = yaml.safe_load((CONFIGS / channel / "digitization_config.yaml").read_text())
    package = yaml.safe_load((CONFIGS / channel / "package_parquet_config.yaml").read_text())
    assert digi["vertexing_track_links"] == "auto"
    assert package["vertex_links"] == "auto"
    # the vertex truth links read truth/particles, which must be written first
    assert package["objects"].index("particles_full") < package["objects"].index("vertices")
    assert package["run_size"] == package["chunk_size"] == digi["events"]


def test_unknown_stage_and_channel_are_rejected():
    assert _pipeline("--channel", "ttbar", "--from", "nonsense").returncode != 0
    assert _pipeline("--channel", "nonsense").returncode != 0
    assert _pipeline("--channel", "ttbar", "--events", "ten").returncode != 0


@pytest.mark.parametrize("setting,available,expected", [
    (True, False, True), (False, True, False), (None, True, False),
    ("auto", True, True), ("auto", False, False), ("true", False, True), ("False", True, False),
])
def test_want_vertex_track_links(setting, available, expected):
    assert want_vertex_track_links(setting, available=available) is expected


def test_want_vertex_track_links_rejects_other_strings():
    with pytest.raises(ValueError):
        want_vertex_track_links("sometimes", available=True)


def test_resolve_vertex_links(tmp_path):
    pytest.importorskip("scipy")
    pkg = pytest.importorskip("package_native_parquet", reason="needs pyarrow")
    runs = [tmp_path / "0", tmp_path / "1"]
    for r in runs:
        r.mkdir()
    assert pkg.resolve_vertex_links(True, runs) is True
    assert pkg.resolve_vertex_links(False, runs) is False
    assert pkg.resolve_vertex_links("auto", runs) is False
    (runs[0] / pkg.VERTEX_LINKS_CSV).write_text("")
    with pytest.raises(RuntimeError):
        pkg.resolve_vertex_links("auto", runs)
    (runs[1] / pkg.VERTEX_LINKS_CSV).write_text("")
    assert pkg.resolve_vertex_links("auto", runs) is True
    with pytest.raises(ValueError):
        pkg.resolve_vertex_links("maybe", runs)


def test_stage_timer_appends_rows(tmp_path):
    record = tmp_path / "out" / "timing.csv"
    assert stage_timer.main(["--record", str(record), "--stage", "ok", "--", sys.executable, "-c", "pass"]) == 0
    assert stage_timer.main(["--record", str(record), "--stage", "bad", "--", sys.executable, "-c", "raise SystemExit(3)"]) == 3
    rows = list(csv.DictReader(record.open()))
    assert [r["stage"] for r in rows] == ["ok", "bad"]
    assert [r["exit_code"] for r in rows] == ["0", "3"]
    assert list(rows[0]) == stage_timer.COLUMNS
    assert float(rows[0]["max_process_rss_mb"]) > 0


def test_stage_timer_cgroup_peak(tmp_path):
    peak = tmp_path / "memory.peak"
    peak.write_text(str(3 * 1024**2) + "\n")
    assert stage_timer.cgroup_peak_mb(peak) == "3"
    assert stage_timer.cgroup_peak_mb(tmp_path / "missing") == ""
