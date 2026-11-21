"""Per-run provenance records and the pushed-commit gate for production stages (v23).

Config block (all keys optional, off unless ``record`` / ``require_pushed_commit`` is true):

    provenance:
      record: true                  # write one JSON record per run (or chunk) and attempt
      require_pushed_commit: true   # refuse to run from a dirty tree or a commit on no remote branch
      commit_marker: false          # write <version>/.git_commit_success (packaging stage only in v23)
      inputs:  [edm4hep.root, ...]  # run-relative paths or globs, hashed (sha256) after the stage
      outputs: [...]                # same, for outputs
      image_files: [/opt/...]       # files inside the stage image, hashed inside the container
      container_json: digi_provenance.json   # run-relative JSON written by the stage, embedded

Record path: <runs>/<N>/provenance/<stage>_attempt<K>.json for per-run stages, and
<version>/provenance/<stage>/chunk<N>_attempt<K>.json for packaging. Attempts are numbered
1, 2, ... in the order they finish; a record is written for failed attempts too.

Pure python + git/podman-hpc subprocess calls; runs on the host (run_stage), not in the image.
"""
from __future__ import annotations

import datetime
import glob
import hashlib
import json
import os
import re
import socket
import subprocess
from pathlib import Path

SCHEMA = "colliderml.provenance/1"
_SECRET = re.compile(r"(?i)(token|passw|secret|api[_-]?key|credential)")


def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def git_state(repo) -> dict:
    """HEAD sha, branch, dirty flag (tracked or untracked, ignored files excluded) and the
    remote-tracking branches that contain HEAD (local refs; run `git fetch` beforehand)."""
    sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    status = _git(repo, "status", "--porcelain")
    remotes = _git(repo, "branch", "-r", "--contains", "HEAD").stdout.split()
    remotes = [r for r in remotes if r != "->"]
    return {
        "sha": sha,
        "branch": branch,
        "dirty": bool(status.stdout.strip()) or status.returncode != 0,
        "dirty_paths": status.stdout.splitlines()[:20],
        "remote_branches_containing_head": remotes,
    }


def pushed_commit_error(state: dict) -> str | None:
    """Reason the code state is not acceptable for a production run, or None."""
    if not state.get("sha"):
        return "not a git checkout"
    if state["dirty"]:
        return f"working tree is dirty: {state['dirty_paths']}"
    if not state["remote_branches_containing_head"]:
        return f"HEAD {state['sha'][:10]} is on no remote-tracking branch (push it, then git fetch)"
    return None


def redacted_sha256(path) -> str | None:
    """sha256 of a site config with secret-looking lines removed (None if absent)."""
    p = Path(path)
    if not p.exists():
        return None
    lines = [l for l in p.read_text().splitlines(keepends=True) if not _SECRET.search(l)]
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def effective_config_sha256(config: dict) -> str:
    clean = {k: v for k, v in config.items() if k != "env_setup"}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, default=str).encode()).hexdigest()


def hash_paths(base: Path, patterns) -> list[dict]:
    """sha256, size and realpath of every file matching the run-relative patterns."""
    out = []
    for pat in patterns or []:
        matches = sorted(glob.glob(str(base / pat)))
        if not matches:
            out.append({"pattern": pat, "missing": True})
            continue
        for m in matches:
            p = Path(m)
            real = os.path.realpath(p)
            rec = {"path": str(p.relative_to(base)) if str(p).startswith(str(base)) else str(p),
                   "realpath": real, "is_link": p.is_symlink()}
            try:
                rec["size"] = os.path.getsize(real)
                rec["sha256"] = sha256_file(real)
            except OSError as e:
                rec["error"] = str(e)
            out.append(rec)
    return out


def image_id(container: str | None) -> str | None:
    if not container:
        return None
    for cmd in (["podman-hpc", "inspect", "--format", "{{.Id}}", container],
                ["podman-hpc", "images", "--format", "{{.Repository}}:{{.Tag}} {{.ID}}"]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode != 0:
            continue
        if cmd[1] == "inspect" and r.stdout.strip():
            return r.stdout.strip()
        for line in r.stdout.splitlines():
            name, _, iid = line.partition(" ")
            if name == container or name == f"localhost/{container}":
                return iid
    return None


def image_file_hashes(container: str | None, files) -> dict:
    if not container or not files:
        return {}
    try:
        r = subprocess.run(["podman-hpc", "run", "--rm", "--entrypoint", "sha256sum", container, *files],
                           capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": str(e)}
    out = {f: None for f in files}
    for line in r.stdout.splitlines():
        h, _, f = line.partition("  ")
        out[f.strip()] = h
    if r.returncode != 0:
        out["error"] = r.stderr.strip()[-500:]
    return out


def record_path(config: dict, runs_dir: Path, version_dir: Path, unit: int, postprocessing: bool) -> Path:
    stage = config["stage"]
    if postprocessing:
        d = version_dir / "provenance" / stage
        stem = f"chunk{unit}"
    else:
        d = runs_dir / str(unit) / "provenance"
        stem = stage
    d.mkdir(parents=True, exist_ok=True)
    k = 1
    while (d / f"{stem}_attempt{k}.json").exists():
        k += 1
    return d / f"{stem}_attempt{k}.json"


def build_record(*, config: dict, config_path, unit: int, rc: int, t_start: float, t_end: float,
                 repo, base: Path, container: str | None, env_setup_path) -> dict:
    prov = config.get("provenance") or {}
    rec = {
        "schema": SCHEMA,
        "stage": config.get("stage"),
        "campaign": config.get("campaign"),
        "dataset": config.get("dataset"),
        "version": config.get("version"),
        "unit": unit,
        "return_code": rc,
        "start_utc": datetime.datetime.utcfromtimestamp(t_start).isoformat() + "Z",
        "end_utc": datetime.datetime.utcfromtimestamp(t_end).isoformat() + "Z",
        "host": socket.gethostname(),
        "slurm": {k: os.environ.get(k) for k in
                  ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_PROCID", "SLURM_NODEID", "SLURM_ARRAY_TASK_ID")},
        "config_path": str(config_path),
        "config_file_sha256": sha256_file(config_path) if config_path and Path(config_path).exists() else None,
        "effective_config_sha256": effective_config_sha256(config),
        "env_setup_redacted_sha256": redacted_sha256(env_setup_path),
        "git": git_state(repo) if repo else None,
        "container": container,
        "image_id": image_id(container),
        "inputs": hash_paths(base, prov.get("inputs")),
        "outputs": hash_paths(base, prov.get("outputs")),
        "image_files": image_file_hashes(container, prov.get("image_files")),
    }
    cj = prov.get("container_json")
    if cj:
        p = base / cj
        try:
            rec["container_record"] = json.loads(p.read_text())
        except (OSError, ValueError) as e:
            rec["container_record"] = {"error": f"{p}: {e}"}
    return rec


def write_record(path: Path, rec: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)
