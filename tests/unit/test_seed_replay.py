"""Historical digitisation seed replay (v23 D2) and the v23 tracking configs that use it."""

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SIM_UTILS = REPO / "scripts" / "simulation" / "utils"


def _load(name):
    # by path: scripts/postprocessing/utils is also a package called "utils"
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"sim_utils_{name}", SIM_UTILS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_sr = _load("seed_replay")
SeedReplayError, replay_seed_string = _sr.SeedReplayError, _sr.replay_seed_string
hash_seed_string = _load("config").hash_seed_string

PU200 = [{"runs": [0, 15], "seed_string": "ttbar_v20_run{run}"},
         {"runs": [16, 1562], "seed_string": "ttbar_v22_run{run}"}]


def test_ranges():
    assert replay_seed_string(PU200, "0") == "ttbar_v20_run0"
    assert replay_seed_string(PU200, 15) == "ttbar_v20_run15"
    assert replay_seed_string(PU200, "16") == "ttbar_v22_run16"
    assert replay_seed_string(PU200, "1562") == "ttbar_v22_run1562"


@pytest.mark.parametrize("run", ["1563", "-1", None, "all", "3.0", " 7x"])
def test_uncovered_or_bad_run_raises(run):
    with pytest.raises(SeedReplayError):
        replay_seed_string(PU200, run)


def test_overlap_and_malformed_raise():
    with pytest.raises(SeedReplayError, match="overlap"):
        replay_seed_string([{"runs": [0, 10], "seed_string": "a{run}"},
                            {"runs": [10, 20], "seed_string": "b{run}"}], 5)
    with pytest.raises(SeedReplayError):
        replay_seed_string([{"runs": [0, 10], "seed_string": "no_placeholder"}], 5)
    with pytest.raises(SeedReplayError):
        replay_seed_string([], 5)


def test_hash_matches_historical_logs():
    # values from the v20 / v22 tracking logs (utils.config "hashed to seed")
    assert hash_seed_string("ttbar_v20_run0") == 184970890
    assert hash_seed_string("ttbar_v20_run15") == 25170514
    assert hash_seed_string("ttbar_v20_run7") == 57859072
    assert hash_seed_string("ttbar_v22_run16") == 795355875
    assert hash_seed_string("ttbar_v22_run1562") == 399302094


def test_v23_tracking_configs():
    expect = {"hard_scatter": {0: "ttbar_v20_run0", 7: "ttbar_v20_run7"},
              "full_pileup": {0: "ttbar_v20_run0", 15: "ttbar_v20_run15", 16: "ttbar_v22_run16",
                              1562: "ttbar_v22_run1562"}}
    for camp, cases in expect.items():
        c = yaml.safe_load(open(REPO / f"configs_production/{camp}/ttbar/v23/digitization_config.yaml"))
        assert c["version"] == "v23"
        assert c["seeding_bfield_tesla"] == 3.0
        assert c["seed_min_pt_gev"] == 0.65
        assert c["vertexing"] is True
        assert "staging" in c["common"]["output_base_dir"]
        for run, s in cases.items():
            assert replay_seed_string(c["digitization_seed_replay"], run) == s
        n = c["job_config"]["n_runs"]
        replay_seed_string(c["digitization_seed_replay"], n - 1)
        with pytest.raises(SeedReplayError):
            replay_seed_string(c["digitization_seed_replay"], n)
