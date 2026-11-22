"""Per-run provenance records and the pushed-commit gate for production stages (v23).

Config block (all keys optional, off unless ``record`` / ``require_pushed_commit`` is true):

    provenance:
      record: true                  # write one JSON record per run (or chunk) and attempt
      require_pushed_commit: true   # refuse to run from a dirty tree or a commit on no remote branch
      commit_marker: false          # write <version>/.git_commit_success (packaging stage only in v23)
      require_approved_sha: true    # HEAD must equal $COLLIDERML_APPROVED_SHA
      approved_remote: https://github.com/OpenDataDetector/ColliderML-Production   # origin URL (normalised, exact)
      approved_ref: origin/release/v23   # HEAD must be on this remote-tracking branch
      inputs:  [edm4hep.root, ...]  # run-relative paths or globs, hashed (sha256) after the stage
      outputs: [...]                # same, for outputs; "a|b" = at least one of the alternatives
      image_files: [/opt/...]       # files inside the stage image, hashed inside the container
      container_json: digi_provenance.json   # JSON written by the stage, embedded ({unit} = run/chunk)

Attempts are reserved before the stage starts (exclusive create of <stem>_attempt<K>.json with
status "started"), so concurrent retries get distinct numbers and a killed wrapper leaves a
"started" record behind. The final record has status "complete" only when every declared input
and output is present and hashed, the image id is known and the embedded JSON is readable;
otherwise "incomplete" with a list of problems, and run_stage turns a task exit 0 into 4.

Record path: <runs>/<N>/provenance/<stage>_attempt<K>.json for per-run stages, and
<version>/provenance/<stage>/chunk<N>_attempt<K>.json for packaging. A record is written for
failed attempts too.

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
        "origin_url": _git(repo, "remote", "get-url", "origin").stdout.strip(),
        "sha": sha,
        "branch": branch,
        "dirty": bool(status.stdout.strip()) or status.returncode != 0,
        "dirty_paths": status.stdout.splitlines()[:20],
        "remote_branches_containing_head": remotes,
    }


def _norm_url(u: str) -> str:
    """https://github.com/Org/Repo(.git)(/) and git@github.com:Org/Repo.git -> github.com/org/repo"""
    u = u.strip().rstrip("/").lower()
    if u.endswith(".git"):
        u = u[:-4]
    for pre in ("https://", "http://", "ssh://git@", "git@"):
        if u.startswith(pre):
            u = u[len(pre):]
    return u.replace(":", "/", 1) if u.startswith("github.com:") else u


def pushed_commit_error(state: dict, approved_sha: str | None = None,
                        approved_remote: str | None = None, approved_ref: str | None = None) -> str | None:
    """Reason the code state is not acceptable for a production run, or None."""
    if not state.get("sha"):
        return "not a git checkout"
    if state["dirty"]:
        return f"working tree is dirty: {state['dirty_paths']}"
    if approved_remote and _norm_url(state.get("origin_url", "")) != _norm_url(approved_remote):
        return f"origin {state.get('origin_url')!r} is not {approved_remote}"
    origin_branches = [r for r in state["remote_branches_containing_head"] if r.startswith("origin/")]
    if not origin_branches:
        return f"HEAD {state['sha'][:10]} is on no origin branch (push it, then git fetch)"
    if approved_ref and approved_ref not in origin_branches:
        return f"HEAD {state['sha'][:10]} is not on {approved_ref} (on {origin_branches})"
    if approved_sha is not None and state["sha"] != approved_sha:
        return f"HEAD {state['sha'][:10]} is not the approved commit {approved_sha[:10]}"
    return None


def gate_error(config: dict, repo) -> tuple[dict, str | None]:
    """Apply the provenance gate of a config: (git state, error or None)."""
    prov = config.get("provenance") or {}
    state = git_state(repo)
    approved = None
    if prov.get("require_approved_sha"):
        approved = os.environ.get("COLLIDERML_APPROVED_SHA")
        if not approved:
            return state, "provenance.require_approved_sha: COLLIDERML_APPROVED_SHA is not set"
    return state, pushed_commit_error(state, approved, prov.get("approved_remote"), prov.get("approved_ref"))


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
    """sha256, size and realpath of every file matching the run-relative patterns.
    "a|b" matches the first alternative that exists."""
    out = []
    for pat in patterns or []:
        matches = []
        for alt in pat.split("|"):
            matches = sorted(glob.glob(str(base / alt)))
            if matches:
                break
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
                rec["mtime"] = os.path.getmtime(real)
                rec["sha256"] = sha256_file(real)
            except OSError as e:
                rec["error"] = str(e)
            out.append(rec)
    return out


def parse_image_listing(stdout: str, container: str) -> str | None:
    """Full image id (64 hex, no 'sha256:') of `container` from `podman-hpc images --no-trunc` output lines
    '<repository>:<tag> <id>'; None if absent or ambiguous."""
    ids = set()
    for line in stdout.splitlines():
        name, _, iid = line.strip().partition(" ")
        if name in (container, f"localhost/{container}"):
            ids.add(iid.strip().removeprefix("sha256:"))
    return ids.pop() if len(ids) == 1 else None


def image_id(container: str | None) -> str | None:
    """Full image id. `podman-hpc inspect` fails on compute nodes for migrated images ('no such object'), so the
    no-trunc image listing is the fallback there (verified 2026-11-21, debug jobs 61773295 and 61773331)."""
    if not container:
        return None
    try:
        r = subprocess.run(["podman-hpc", "inspect", "--format", "{{.Id}}", container], capture_output=True,
                           text=True, timeout=120)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip().removeprefix("sha256:")
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        r = subprocess.run(["podman-hpc", "images", "--no-trunc", "--format", "{{.Repository}}:{{.Tag}} {{.ID}}"],
                           capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return parse_image_listing(r.stdout, container) if r.returncode == 0 else None


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


def reserve_attempt(config: dict, runs_dir: Path, version_dir: Path, unit: int, postprocessing: bool) -> Path:
    """Exclusively create the next <stem>_attempt<K>.json with status "started" and return it."""
    stage = config["stage"]
    if postprocessing:
        d = version_dir / "provenance" / stage
        stem = f"chunk{unit}"
    else:
        d = runs_dir / str(unit) / "provenance"
        stem = stage
    d.mkdir(parents=True, exist_ok=True)
    k = 1
    while True:
        path = d / f"{stem}_attempt{k}.json"
        try:
            with open(path, "x") as f:
                json.dump({"schema": SCHEMA, "stage": stage, "unit": unit, "status": "started",
                           "host": socket.gethostname(), "pid": os.getpid(),
                           "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                           "start_utc": datetime.datetime.utcnow().isoformat() + "Z"}, f)
            return path
        except FileExistsError:
            k += 1


record_path = reserve_attempt  # backwards-compatible name


def unit_lock(attempt_path: Path) -> Path:
    """Exclusive per-unit lock next to the attempt records: a second concurrent attempt of the same run or
    chunk is refused (FileExistsError). A lock left by a killed attempt stays until the operator removes it."""
    lock = attempt_path.parent / (attempt_path.name.rsplit("_attempt", 1)[0] + ".lock")
    with open(lock, "x") as f:
        f.write(f"{socket.gethostname()} {os.getpid()} {attempt_path.name}\n")
    return lock


def record_problems(rec: dict) -> list[str]:
    """Missing evidence in a record (empty list = complete). Outputs must be fresh: modified after the
    attempt started (a no-op retry cannot inherit an earlier attempt's files)."""
    probs = []
    t0 = rec.get("start_epoch")
    for kind in ("inputs", "outputs"):
        for e in rec.get(kind) or []:
            if e.get("missing"):
                probs.append(f"{kind}: {e['pattern']} missing")
            elif "sha256" not in e:
                probs.append(f"{kind}: {e.get('path')} not hashed ({e.get('error')})")
            elif kind == "outputs" and t0 is not None and e.get("mtime", t0) < t0:
                probs.append(f"outputs: {e.get('path')} older than the attempt (stale)")
    if rec.get("container") and not rec.get("image_id"):
        probs.append("image id unknown")
    imgf = rec.get("image_files") or {}
    if "error" in imgf or any(v is None for k, v in imgf.items() if k != "error"):
        probs.append(f"image files not hashed: {imgf}")
    cr = rec.get("container_record")
    if "container_record" in rec and not cr:
        probs.append("container record empty")
    if isinstance(cr, dict) and "error" in cr and len(cr) == 1:
        probs.append(f"container record: {cr['error']}")
    elif isinstance(cr, dict) and "status" in cr and cr["status"] != "ok":
        probs.append(f"container record status {cr['status']!r}")
    if not (rec.get("git") or {}).get("sha"):
        probs.append("no git sha")
    return probs


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
        "start_epoch": t_start,
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
        p = base / str(cj).format(unit=unit)
        try:
            rec["container_record"] = json.loads(p.read_text())
        except (OSError, ValueError) as e:
            rec["container_record"] = {"error": f"{p}: {e}"}
    return rec


def write_record(path: Path, rec: dict) -> None:
    rec = dict(rec)
    rec["problems"] = record_problems(rec)
    rec["status"] = "complete" if not rec["problems"] else "incomplete"
    tmp = path.with_name(f".{path.name}.{socket.gethostname()}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)
