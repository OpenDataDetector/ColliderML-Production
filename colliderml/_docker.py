"""
Docker container management for ColliderML simulation.

Wraps the Docker SDK (docker-py) to run simulation stages inside the
ODD software container. Handles image pulling, volume mounts, cache
directories, and environment setup.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_IMAGE = "ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0"


def check_docker_available():
    """Check if Docker is available on the system."""
    if shutil.which("docker") is None:
        raise RuntimeError(
            "Docker is not installed or not in PATH.\n"
            "Install Docker: https://docs.docker.com/get-docker/\n"
            "Or use remote simulation: colliderml.simulate(..., remote=True)"
        )

    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Docker daemon is not running.\n"
                "Start Docker and try again.\n"
                f"Error: {result.stderr.strip()}"
            )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Docker daemon is not responding (timed out).")


def check_image_available(image=DEFAULT_IMAGE):
    """Check if the container image is available locally."""
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def pull_image(image=DEFAULT_IMAGE, interactive=True):
    """Pull the container image, optionally prompting the user first.

    Args:
        image: Docker image to pull.
        interactive: If True, prompt before pulling (image is ~10 GB).

    Returns:
        True if image is now available.
    """
    if check_image_available(image):
        return True

    if interactive and sys.stdin.isatty():
        print(f"\nDocker image {image} not found locally.")
        print("This is a ~10 GB download.")
        try:
            response = input("Continue? [Y/n]: ").strip().lower()
            if response in ("n", "no"):
                print("Aborted. Use remote=True to run on NERSC instead.")
                return False
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return False

    print(f"Pulling {image}...")
    result = subprocess.run(
        ["docker", "pull", image],
        timeout=3600,  # 1 hour timeout for large image
    )
    return result.returncode == 0


def ensure_cache(repo_root):
    """Ensure the .cache directory exists with required contents.

    Clones ODD and MG5aMC_PY8_interface on the host (where network works).
    These are mounted into the container at /cache.

    Args:
        repo_root: Path to the ColliderML-Production repo root.

    Returns:
        Path to the cache directory.
    """
    cache_dir = repo_root / ".cache"
    cache_dir.mkdir(exist_ok=True)

    # Clone ODD v4.0.4 if not present
    odd_xml = cache_dir / "odd-v4" / "xml" / "OpenDataDetector.xml"
    if not odd_xml.exists():
        print("Cloning OpenDataDetector v4.0.4...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", "v4.0.4",
                 "https://gitlab.cern.ch/acts/OpenDataDetector.git",
                 str(cache_dir / "odd-v4")],
                capture_output=True, timeout=300,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("WARNING: Failed to clone ODD. Simulation stages will fail.")

    # Clone MG5aMC_PY8_interface if not present
    mg5_cc = cache_dir / "MG5aMC_PY8_interface" / "MG5aMC_PY8_interface.cc"
    if not mg5_cc.exists():
        print("Cloning MG5aMC_PY8_interface...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/mg5amcnlo/MG5aMC_PY8_interface.git",
                 str(cache_dir / "MG5aMC_PY8_interface")],
                capture_output=True, timeout=120,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("WARNING: Failed to clone MG5aMC_PY8_interface.")

    return cache_dir


def run_stage(repo_root, output_dir, cache_dir, stage_script, config_path,
              seed=42, run_id="0", image=DEFAULT_IMAGE, extra_args=None):
    """Run a single pipeline stage inside the Docker container.

    This mirrors the logic of scripts/cli/run_docker.sh.

    Args:
        repo_root: Path to the ColliderML-Production repo root.
        output_dir: Host directory for output files.
        cache_dir: Host directory for cached data (ODD, G4, pip).
        stage_script: Script path relative to scripts/ (e.g., "simulation/pythia_gen.py").
        config_path: Config file path relative to repo root.
        seed: Random seed.
        run_id: Run subdirectory name.
        image: Docker image to use.
        extra_args: Additional arguments to pass to the stage script.

    Returns:
        subprocess.CompletedProcess from the docker run command.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    script_dir = os.path.dirname(stage_script)
    script_name = os.path.basename(stage_script)

    extra = " ".join(extra_args) if extra_args else ""

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{repo_root}:/workspace",
        "-v", f"{output_dir}:/output",
        "-v", f"{cache_dir}:/cache",
        "-e", "COLLIDERML_CACHE=/cache",
        image,
        "-c",
        (
            f"source /workspace/scripts/cli/setup_container_env.sh && "
            f"cd /workspace/scripts/{script_dir} && "
            f"python3 {script_name} "
            f"--config /workspace/{config_path} "
            f"--output /output/runs "
            f"--output-subdir {run_id} "
            f"--seed {seed} "
            f"{extra}"
        ).strip(),
    ]

    return subprocess.run(cmd, timeout=7200)  # 2 hour timeout per stage


def run_pipeline(repo_root, output_dir, stages, seed=42, run_id="0",
                 image=DEFAULT_IMAGE, on_stage_start=None, on_stage_end=None):
    """Run a full pipeline (multiple stages) inside Docker.

    Args:
        repo_root: Path to the ColliderML-Production repo root.
        output_dir: Host directory for output.
        stages: List of dicts with keys: name, script, config_path.
        seed: Random seed.
        run_id: Run subdirectory.
        image: Docker image to use.
        on_stage_start: Callback(stage_index, stage_name) called before each stage.
        on_stage_end: Callback(stage_index, stage_name, returncode) called after each stage.

    Returns:
        dict with output_dir and per-stage results.
    """
    cache_dir = ensure_cache(Path(repo_root))

    results = []
    for i, stage in enumerate(stages):
        if on_stage_start:
            on_stage_start(i, stage["name"])

        result = run_stage(
            repo_root=repo_root,
            output_dir=output_dir,
            cache_dir=cache_dir,
            stage_script=stage["script"],
            config_path=stage["config_path"],
            seed=seed,
            run_id=run_id,
            image=image,
        )

        if on_stage_end:
            on_stage_end(i, stage["name"], result.returncode)

        results.append({
            "stage": stage["name"],
            "returncode": result.returncode,
        })

        if result.returncode != 0:
            raise RuntimeError(
                f"Stage '{stage['name']}' failed with exit code {result.returncode}. "
                f"Check Docker output above for details."
            )

        # Special handling: ttbar copies MadGraph output for Pythia
        if stage.get("stage") == "madgraph_generation":
            _copy_madgraph_output(output_dir, run_id)

    return {
        "output_dir": str(output_dir),
        "run_dir": str(Path(output_dir) / "runs" / run_id),
        "stages": results,
    }


def _copy_madgraph_output(output_dir, run_id):
    """Copy MadGraph HepMC output to where Pythia expects it.

    MadGraph stages files to runs/all/0/, but Pythia looks in runs/{run_id}/.
    """
    output_dir = Path(output_dir)
    src = output_dir / "runs" / "all" / "0" / "events.hepmc.gz"
    dst_dir = output_dir / "runs" / run_id
    dst = dst_dir / "events.hepmc.gz"

    if src.exists() and not dst.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
