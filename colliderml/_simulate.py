"""
Public simulation API.

Provides the colliderml.simulate() function that orchestrates config
generation, Docker container lifecycle, and output collection.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import yaml


class SimulationResult:
    """Result of a simulation run.

    Attributes:
        output_dir: Path to the output directory.
        run_dir: Path to the specific run directory (e.g., output/runs/0/).
        channel: Physics channel that was simulated.
        events: Number of events requested.
        pileup: Pileup level.
        stages: List of stage results with name and returncode.
    """

    def __init__(self, output_dir, run_dir, channel, events, pileup, stages):
        self.output_dir = Path(output_dir)
        self.run_dir = Path(run_dir)
        self.channel = channel
        self.events = events
        self.pileup = pileup
        self.stages = stages

    @property
    def particles(self):
        """Load particles table as a pyarrow Table."""
        return self._load_parquet("particles")

    @property
    def tracks(self):
        """Load tracks table as a pyarrow Table."""
        return self._load_parquet("tracks")

    @property
    def tracker_hits(self):
        """Load tracker hits table as a pyarrow Table."""
        return self._load_parquet("tracker_hits")

    @property
    def calo_hits(self):
        """Load calorimeter hits table as a pyarrow Table."""
        return self._load_parquet("calo_hits")

    def _load_parquet(self, table_name):
        """Load a Parquet file from the run output."""
        import pyarrow.parquet as pq

        # Search for parquet files in the output
        patterns = [
            self.run_dir / f"{table_name}.parquet",
            self.run_dir / table_name / "*.parquet",
        ]

        for pattern in patterns:
            path = Path(str(pattern))
            if path.exists():
                return pq.read_table(str(path))

        # Try glob for directory-based output
        parquet_dir = self.run_dir / table_name
        if parquet_dir.is_dir():
            files = sorted(parquet_dir.glob("*.parquet"))
            if files:
                import pyarrow as pa
                tables = [pq.read_table(str(f)) for f in files]
                return pa.concat_tables(tables)

        raise FileNotFoundError(
            f"No {table_name} parquet file found in {self.run_dir}. "
            f"Available files: {list(self.run_dir.glob('*'))}"
        )

    def list_files(self):
        """List all output files in the run directory."""
        if not self.run_dir.exists():
            return []
        return sorted(self.run_dir.rglob("*"))

    def __repr__(self):
        n_files = len(self.list_files())
        return (
            f"SimulationResult(channel='{self.channel}', events={self.events}, "
            f"pileup={self.pileup}, files={n_files}, "
            f"output='{self.run_dir}')"
        )


def simulate(
    channel=None,
    events=None,
    pileup=None,
    preset=None,
    seed=42,
    output_dir=None,
    image=None,
    remote=False,
    run_id="0",
):
    """Run the ColliderML simulation pipeline.

    Generates events using the full pipeline: Pythia/MadGraph -> DDSim (Geant4)
    -> Digitization + Reconstruction -> Parquet conversion. Everything runs
    inside a Docker container.

    Args:
        channel: Physics channel ("higgs_portal", "ttbar", etc.).
                 Not needed if preset is specified.
        events: Number of events to simulate.
                Not needed if preset is specified.
        pileup: Pileup level (0-200). Defaults to 0 or preset value.
        preset: Preset name (e.g., "ttbar-quick"). Overrides channel/events/pileup.
        seed: Random seed for reproducibility.
        output_dir: Host directory for output files.
                    Defaults to ./colliderml_output/{channel}_pu{pileup}_{events}evt/
        image: Docker image override.
        remote: If True, submit to NERSC instead of running locally.
                Requires HF token for authentication.
        run_id: Run subdirectory name (default "0").

    Returns:
        SimulationResult with access to output files and data.

    Examples:
        # Quick local simulation
        result = colliderml.simulate(channel="higgs_portal", events=10, pileup=10)
        print(result.particles.to_pandas().head())

        # Using a preset
        result = colliderml.simulate(preset="ttbar-quick")

        # Remote simulation (no Docker needed)
        result = colliderml.simulate(channel="ttbar", events=10000, remote=True)
    """
    from colliderml._config import resolve_preset, get_channel_stages, find_config_dir

    # Resolve preset
    if preset is not None:
        preset_config = resolve_preset(preset)
        channel = channel or preset_config["channel"]
        events = events or preset_config.get("events", 10)
        pileup = pileup if pileup is not None else preset_config.get("pileup", 0)
    else:
        if channel is None:
            raise ValueError("Must specify either 'channel' or 'preset'.")
        if events is None:
            raise ValueError("Must specify 'events' (or use a preset).")
        if pileup is None:
            pileup = 0

    # Remote mode
    if remote:
        return _simulate_remote(channel, events, pileup, seed)

    # Local Docker mode
    from colliderml._docker import (
        check_docker_available,
        pull_image,
        ensure_cache,
        run_pipeline,
        DEFAULT_IMAGE,
    )

    image = image or DEFAULT_IMAGE

    # Check Docker
    check_docker_available()
    if not pull_image(image):
        raise RuntimeError("Docker image not available. Cannot proceed.")

    # Determine output directory
    if output_dir is None:
        output_dir = Path.cwd() / "colliderml_output" / f"{channel}_pu{pileup}_{events}evt"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find repo root (where scripts/ and configs_development/ live)
    repo_root = Path(__file__).parent.parent.resolve()

    # Ensure cache has ODD, MG5 interface, etc.
    cache_dir = ensure_cache(repo_root)

    # Get pipeline stages and config paths
    stages = get_channel_stages(channel)
    config_dir = find_config_dir(channel)

    pipeline_stages = []
    for stage_info in stages:
        config_file = config_dir / stage_info["config"]
        if not config_file.exists():
            continue
        # Config path relative to repo root
        config_rel = config_file.relative_to(repo_root)
        pipeline_stages.append({
            "name": stage_info["name"],
            "stage": stage_info["stage"],
            "script": stage_info["script"],
            "config_path": str(config_rel),
        })

    total = len(pipeline_stages)

    def on_start(i, name):
        print(f"  [{i+1}/{total}] {name}...", flush=True)

    def on_end(i, name, rc):
        status = "done" if rc == 0 else f"FAILED (exit {rc})"
        print(f"  [{i+1}/{total}] {name} - {status}", flush=True)

    print(f"Running {channel} pipeline ({events} events, pileup={pileup})...")

    result = run_pipeline(
        repo_root=str(repo_root),
        output_dir=str(output_dir),
        stages=pipeline_stages,
        seed=seed,
        run_id=run_id,
        image=image,
        on_stage_start=on_start,
        on_stage_end=on_end,
    )

    run_dir = Path(result["run_dir"])
    print(f"Done. Output: {run_dir}")

    # List output files
    if run_dir.exists():
        files = sorted(run_dir.glob("*"))
        for f in files:
            if f.is_file():
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  {f.name} ({size_mb:.1f} MB)")

    return SimulationResult(
        output_dir=str(output_dir),
        run_dir=str(run_dir),
        channel=channel,
        events=events,
        pileup=pileup,
        stages=result["stages"],
    )


def _simulate_remote(channel, events, pileup, seed):
    """Submit simulation to remote NERSC service.

    TODO: Implement in Phase 2. For now, raise a helpful error.
    """
    raise NotImplementedError(
        "Remote simulation (remote=True) is not yet available.\n"
        "This feature will be added in a future release.\n"
        "For now, use local Docker simulation:\n"
        "  colliderml.simulate(channel='{channel}', events={events}, pileup={pileup})\n"
        "Or load pre-generated data:\n"
        "  colliderml.load('{{channel}}_pu{{pileup}}')"
    )
