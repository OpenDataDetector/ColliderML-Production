"""
Config generation for simulation pipelines.

Generates YAML config files from templates, applying user overrides
(events, pileup, seed, etc.). Mirrors the structure of configs in
configs_development/docker_test/.
"""

import copy
from pathlib import Path

import yaml

# Channel -> list of pipeline stages
CHANNEL_STAGES = {
    "higgs_portal": [
        {"name": "Pythia Generation", "stage": "pythia_generation",
         "script": "simulation/pythia_gen.py", "config": "pythia_config.yaml"},
        {"name": "Detector Simulation", "stage": "simulation",
         "script": "simulation/ddsim_run.py", "config": "simulation_config.yaml"},
        {"name": "Digitization & Reconstruction", "stage": "digitization",
         "script": "simulation/digi_and_reco.py", "config": "digitization_config.yaml"},
        {"name": "Parquet Conversion", "stage": "convert_all",
         "script": "postprocessing/convert_all.py", "config": "convert_all.yaml"},
    ],
    "ttbar": [
        {"name": "MadGraph Init", "stage": "madgraph_init",
         "script": "simulation/madgraph_init.py", "config": "madgraph_init_config.yaml"},
        {"name": "MadGraph Generation", "stage": "madgraph_generation",
         "script": "simulation/madgraph_gen.py", "config": "madgraph_generation_config.yaml"},
        {"name": "Pythia Generation", "stage": "pythia_generation",
         "script": "simulation/pythia_gen.py", "config": "pythia_config.yaml"},
        {"name": "Detector Simulation", "stage": "simulation",
         "script": "simulation/ddsim_run.py", "config": "simulation_config.yaml"},
        {"name": "Digitization & Reconstruction", "stage": "digitization",
         "script": "simulation/digi_and_reco.py", "config": "digitization_config.yaml"},
        {"name": "Parquet Conversion", "stage": "convert_all",
         "script": "postprocessing/convert_all.py", "config": "convert_all.yaml"},
    ],
}

# Presets file path (relative to repo root or bundled in package)
PRESETS_PATHS = [
    Path(__file__).parent.parent / "configs_production" / "presets.yaml",
    Path(__file__).parent / "presets.yaml",
]


def get_channel_stages(channel):
    """Get the pipeline stages for a channel."""
    if channel not in CHANNEL_STAGES:
        raise ValueError(
            f"Unknown channel '{channel}'. "
            f"Available: {sorted(CHANNEL_STAGES.keys())}"
        )
    return CHANNEL_STAGES[channel]


def find_config_dir(channel):
    """Find the config directory for a channel.

    Searches in order:
    1. configs_development/docker_test/{channel}/
    2. configs_production/templates/
    """
    repo_root = Path(__file__).parent.parent

    # Docker test configs (complete per-channel configs)
    docker_test = repo_root / "configs_development" / "docker_test" / channel
    if docker_test.is_dir():
        return docker_test

    # Production templates (shared templates)
    templates = repo_root / "configs_production" / "templates"
    if templates.is_dir():
        return templates

    raise FileNotFoundError(
        f"No config directory found for channel '{channel}'. "
        f"Searched: {docker_test}, {templates}"
    )


def generate_configs(channel, events, pileup, seed=42, output_base_dir="/output"):
    """Generate config dicts for all stages of a channel.

    Reads the template configs from the repo, applies overrides for
    events, pileup, and seed, and returns a list of config dicts.

    Args:
        channel: Physics channel name (e.g., "higgs_portal", "ttbar").
        events: Number of events to generate.
        pileup: Pileup level (0-200).
        seed: Random seed.
        output_base_dir: Base output directory inside the container.

    Returns:
        list of dict: One config dict per stage, in pipeline order.
    """
    config_dir = find_config_dir(channel)
    stages = get_channel_stages(channel)

    configs = []
    for stage_info in stages:
        config_file = config_dir / stage_info["config"]
        if not config_file.exists():
            # Skip stages with missing configs (e.g., single_muon has no madgraph)
            continue

        with open(config_file) as f:
            config = yaml.safe_load(f)

        # Apply overrides
        config["events"] = events
        config["common"]["output_base_dir"] = output_base_dir

        # Apply pileup to Pythia stage
        if stage_info["stage"] == "pythia_generation" and "pileup" in config:
            config["pileup"] = pileup

        configs.append({
            "stage_info": stage_info,
            "config": config,
        })

    return configs


def load_presets():
    """Load presets from presets.yaml."""
    for path in PRESETS_PATHS:
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f)
            return data.get("presets", {})
    return {}


def resolve_preset(preset_name):
    """Resolve a preset name to channel + parameters.

    Returns:
        dict with keys: channel, events, pileup, description
    """
    presets = load_presets()
    if preset_name not in presets:
        available = ", ".join(sorted(presets.keys()))
        raise ValueError(
            f"Unknown preset '{preset_name}'. Available: {available}"
        )
    return presets[preset_name]
