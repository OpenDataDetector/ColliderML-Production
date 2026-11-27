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
sim_output_check = _load_sim_util("sim_output_check")

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
    assert package["strict_joins"] is True


def test_invalid_options_are_rejected():
    for args in (["--from", "nonsense"], ["--events", "ten"], ["--events", "0"],
                 ["--events", "00"], ["--pandora-processes", "0"], ["--threads", ""], ["--events"],
                 ["--from", "digitization", "--to", "simulation"]):
        result = _pipeline("--channel", "ttbar", *args)
        assert result.returncode != 0, args
        assert result.stderr.startswith("ERROR"), args
    assert _pipeline("--channel", "nonsense").returncode != 0


def test_single_thread_is_rejected():
    result = _pipeline("--channel", "ttbar", "--threads", "1")
    assert result.returncode != 0
    assert "2 or more" in result.stderr
    for channel in CHANNELS:
        sim = yaml.safe_load((CONFIGS / channel / "simulation_config.yaml").read_text())
        assert sim["threads"] >= 2
        assert sim["verify_output"] is True


def test_output_directory_with_a_run_is_refused(tmp_path):
    (tmp_path / "runs").mkdir()
    result = _pipeline("--channel", "ttbar", "--output", str(tmp_path), "--runtime", "true")
    assert result.returncode != 0
    assert "already holds a run" in result.stderr


def test_stages_reach_the_runtime_as_separate_arguments(tmp_path):
    """With `echo` as the runtime, each stage prints its container command line."""
    result = _pipeline("--channel", "higgs_portal", "--output", str(tmp_path / "my out"), "--cache",
                       str(tmp_path / "cache"), "--runtime", "echo", "--events", "3", "--pileup", "5",
                       "--pandora-processes", "2")
    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if " stage sim " in line or " stage reco " in line]
    tails = [line.split(" stage ", 1)[1] for line in lines]
    assert tails[0] == ("sim pythia_generation simulation/pythia_gen.py "
                        "configs_development/docker_test/higgs_portal/pythia_config.yaml 0 42 --events 3 --pileup 5")
    assert tails[1].endswith("simulation_config.yaml 0 42 --events 3 --threads 2")
    assert tails[3].endswith("particles_full_config.yaml 0 42")
    assert tails[4].startswith("reco pandora_reco ") and tails[4].endswith("--processes 2")
    assert tails[5].startswith("reco reco_tables ")
    assert tails[6].endswith("package_parquet_config.yaml 0 42 --run-size 3 --chunk-size 3")
    assert f"-v {tmp_path / 'my out'}:/output" in result.stdout


def test_partial_rerun_keeps_the_event_count_of_the_run(tmp_path):
    out = tmp_path / "out"
    first = _pipeline("--channel", "higgs_portal", "--output", str(out), "--cache", str(tmp_path / "c"),
                      "--runtime", "echo", "--events", "007", "--pileup", "3")
    assert first.returncode == 0, first.stderr
    assert "--events 7 --pileup 3" in first.stdout
    assert (out / "pipeline_settings").read_text() == "channel=higgs_portal\nevents=7\npileup=3\n"
    (out / "parquet" / "reco").mkdir(parents=True)

    again = _pipeline("--channel", "higgs_portal", "--output", str(out), "--cache", str(tmp_path / "c"),
                      "--runtime", "echo", "--from", "package_parquet")
    assert again.returncode == 0, again.stderr
    assert "--run-size 7 --chunk-size 7" in again.stdout
    assert not (out / "parquet").exists()            # set aside before packaging again
    assert len(list(out.glob("parquet.replaced.*"))) == 1


def test_run_id_must_be_a_plain_name():
    import os
    result = subprocess.run(
        ["bash", str(REPO / "scripts/cli/run_docker.sh"), "simulation/pythia_gen.py",
         "configs_development/docker_test/ttbar/pythia_config.yaml"],
        capture_output=True, text=True, env={**os.environ, "RUN_ID": "../x", "CONTAINER_RUNTIME": "echo"})
    assert result.returncode != 0
    assert "RUN_ID must be a plain name" in result.stderr


def test_ttbar_pythia_reads_the_madgraph_output_in_place():
    cfg = yaml.safe_load((CONFIGS / "ttbar" / "pythia_config.yaml").read_text())
    # madgraph_gen.py stages the unsplit file of run 0 here
    assert cfg["hard_scatter_file"] == "/output/runs/all/0/events.hepmc.gz"
    assert "hard_process" not in cfg


def test_duplicated_event_numbers():
    assert sim_output_check.duplicated_event_numbers([0, 2, 1, 3]) == []
    assert sim_output_check.duplicated_event_numbers([0, 0, 1, 1, 2]) == [0, 1]
    assert sim_output_check.duplicated_event_numbers([]) == []


class _FakeSimFile:
    """Stands in for uproot.open(<ddsim file>): one array per entry, as podio stores EventHeader."""

    def __init__(self, numbers):
        self._numbers = numbers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getitem__(self, key):
        assert key in ("events", sim_output_check.EVENT_NUMBER_BRANCH)
        return self

    def array(self, library):
        import numpy as np
        out = np.empty(len(self._numbers), dtype=object)
        for i, n in enumerate(self._numbers):
            out[i] = np.array(n if isinstance(n, list) else [n], dtype=np.uint64)
        return out


@pytest.mark.parametrize("numbers,requested,message", [
    ([0, 0, 1, 1], 2, "4 entries but 2 distinct"),
    ([], 2, "no events"),
    ([0, 1], 3, "2 events, 3 requested"),
    ([0, 1, 7], 3, "are not 0 to 2"),
    ([0, [1, 2]], None, "exactly one EventHeader"),
])
def test_check_sim_output_failures(monkeypatch, numbers, requested, message):
    uproot = pytest.importorskip("uproot")
    monkeypatch.setattr(uproot, "open", lambda path: _FakeSimFile(numbers))
    with pytest.raises(sim_output_check.SimOutputError, match=message):
        sim_output_check.check_sim_output("edm4hep.root", requested)


def test_check_sim_output_accepts_complete_files(monkeypatch):
    uproot = pytest.importorskip("uproot")
    monkeypatch.setattr(uproot, "open", lambda path: _FakeSimFile([0, 2, 1]))
    assert sim_output_check.check_sim_output("edm4hep.root", 3) == 3
    assert sim_output_check.check_sim_output("edm4hep.root") == 3
    assert sim_output_check.check_sim_output("edm4hep.root", -1) == 3


def test_check_sim_output_reports_unreadable_files(tmp_path):
    pytest.importorskip("uproot")
    with pytest.raises(sim_output_check.SimOutputError, match="cannot read"):
        sim_output_check.check_sim_output(tmp_path / "missing.root", 2)


def test_particles_full_requires_the_status_words():
    pytest.importorskip("scipy")
    pf = pytest.importorskip("convert_particles_full", reason="needs pyarrow and uproot")
    assert set(pf.OPTIONAL_INT_COLUMNS) == {"color_a", "color_b"}
    assert {"generator_status", "simulator_status"} <= set(pf.INT_BRANCHES) - set(pf.OPTIONAL_INT_COLUMNS)


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
