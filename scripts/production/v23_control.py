#!/usr/bin/env python3
"""ttbar v23 production controller (PRODUCTION_PLAN.md section 8b: E2, E3, E5, E7).

Every production action reads one FROZEN launch record (``freeze``). Units, chunk contributors and the inputs
each unit consumes are enumerated here from the record's numbers (run size, run count, release size, chunk
size), independently of the stage code, and compared with what the attempt records and chunk manifests say.

  freeze       SPEC.yaml --out LAUNCH.json     write the launch record (exclusive create; never overwritten)
  check        LAUNCH.json                     record digest and live identity (git, env_setup, configs, images, verifier)
  expect       LAUNCH.json [--campaign C]      expected units per stage and the contributors of every chunk
  status       LAUNCH.json --campaign C --stage S [--units ...]
  plan         LAUNCH.json --campaign C --stage S --out FILE [--first I --last J]
               units to run: missing, failed or interrupted, with every upstream input accepted and in place
  audit        LAUNCH.json --campaign C --stage S
               every accepted unit consumed exactly the accepted upstream outputs and frozen parents
               (sha256 from the records and manifests; Geant4 files by frozen stat)
  reconcile    LAUNCH.json --campaign C [--apply] [--purge-failed]
               stale unit locks (owner job ended), interrupted attempts, temporary shards
  budget       LAUNCH.json [--add-node-hours X]   consumed (sacct of the ledger jobs) + forecast <= cap
  dispatch     LAUNCH.json --campaign C --stage S --units FILE --nodes N --time HH:MM:SS --per-wave W
               [--qos Q] [--submit]   (check, upstream audit, plan barrier and budget gate, then sbatch)
  release      {preflight,inventory,publish,verify-bytes,rollback} LAUNCH.json --campaign C ...

Exit status: 0 when the requested condition holds, 1 when it does not (the reasons are printed), 2 on usage
errors. Nothing writes into a version directory except reconcile --apply and the release steps.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import os
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "cli"))
import attempt_isolation  # noqa: E402
import run_provenance as rp  # noqa: E402

SCHEMA = "colliderml.v23.launch/1"
STAGES = ["digitization", "pandora_reco", "reco_tables", "package_parquet"]
RUN_STAGES = STAGES[:3]
GROUP = {"particles": "truth", "tracker_simhits": "truth", "generator_particles": "truth",
         "tracker_hits": "reco", "tracks": "reco", "truth_tracks": "reco", "calo_cells": "reco",
         "calo_clusters": "reco", "pfos": "reco", "vertices": "reco"}
RECO_TABLE_OBJECTS = {"calo_cells", "calo_clusters", "pfos"}
SIM_INPUT = "edm4hep.root"      # Geant4 file: stat-only identity (frozen in the record)


# ------------------------------------------------------------------------------------------------ helpers
def sha256_file(p) -> str:
    return rp.sha256_file(p)


def canonical(d) -> bytes:
    return json.dumps(d, sort_keys=True, separators=(",", ":"), default=str).encode()


def record_digest(rec: dict) -> str:
    return hashlib.sha256(canonical({k: v for k, v in rec.items() if k != "digest"})).hexdigest()


def stat_of(p) -> dict:
    real = os.path.realpath(p)
    st = os.stat(real)
    return {"realpath": real, "size": st.st_size, "mtime_ns": st.st_mtime_ns, "inode": st.st_ino,
            "owner_writable": bool(st.st_mode & 0o200)}


def chunk_window(i: int, chunk: int, release: int) -> tuple[int, int]:
    """Independent re-derivation of the release chunk window (the packager has its own)."""
    s = i * chunk
    if i < 0 or s >= release:
        raise ValueError(f"chunk {i} outside the release of {release} events")
    return s, min(release, s + chunk) - 1


def contributors(i: int, chunk: int, release: int, run_size: int) -> list[dict]:
    """Runs a chunk reads and the run-local entry range of each: [{"run", "first", "last"}]."""
    s, e = chunk_window(i, chunk, release)
    out = []
    for r in range(s // run_size, e // run_size + 1):
        lo, hi = max(s, r * run_size), min(e, (r + 1) * run_size - 1)
        out.append({"run": r, "first": lo - r * run_size, "last": hi - r * run_size})
    return out


def load_record(path) -> dict:
    rec = json.loads(Path(path).read_text())
    if rec.get("schema") != SCHEMA:
        raise SystemExit(f"{path}: not a {SCHEMA} launch record")
    if record_digest(rec) != rec.get("digest"):
        raise SystemExit(f"{path}: digest mismatch (the launch record was edited after freezing)")
    return rec


def camp(rec, c) -> dict:
    if c not in rec["campaigns"]:
        raise SystemExit(f"campaign {c} not in the launch record ({sorted(rec['campaigns'])})")
    return rec["campaigns"][c]


def units_of(cr: dict, stage: str) -> list[int]:
    """Expected units: every run / every release chunk, or the frozen rehearsal subset (unit_ranges)."""
    if stage in RUN_STAGES:
        full = range(cr["n_runs"])
    else:
        full = range((cr["release_events"] + cr["chunk_size"] - 1) // cr["chunk_size"])
    lo_hi = (cr.get("unit_ranges") or {}).get(stage)
    return [u for u in full if lo_hi is None or lo_hi[0] <= u <= lo_hi[1]]


def prov_dir(cr: dict, stage: str, unit: int) -> tuple[Path, str]:
    v = Path(cr["version_dir"])
    return rp.unit_paths(v / "runs", v, unit, stage == "package_parquet", stage)


def live_jobs() -> set[str] | None:
    try:
        out = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i"], capture_output=True,
                             text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return {x.split("_")[0] for x in out.stdout.split()}


def lock_owner(lock: Path) -> dict:
    t = lock.read_text(errors="replace").strip()
    try:
        return json.loads(t)
    except ValueError:      # pre-v23-E3 format "host pid name"
        parts = t.split()
        return {"host": parts[0] if parts else None, "pid": int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None}


def lock_alive(owner: dict, jobs: set[str] | None) -> bool | None:
    """True / False when it can be decided, None when not (no job id and another host)."""
    jid = owner.get("slurm_job_id")
    if jid:
        if jobs is None:
            return None
        if str(jid).split("_")[0] not in jobs:
            return False
        # the owner's job is still running: on the owner's node the owner pid decides (an attempt killed
        # inside a live allocation); elsewhere it counts as alive
        if owner.get("host") != socket.gethostname() or not owner.get("pid"):
            return True
    if owner.get("host") == socket.gethostname() and owner.get("pid"):
        try:
            os.kill(int(owner["pid"]), 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    return None


# ------------------------------------------------------------------------------------------------ freeze
def _effective_config(path: Path) -> dict:
    import cli_utils
    return cli_utils.load_and_process_config(str(path), REPO / "scripts" / "cli" / "env_setup.yaml")


def _tarball_sha256(tarball: str | None) -> str | None:
    if not tarball or not Path(tarball).exists():
        return None
    t = Path(tarball)
    for f in [Path(str(t) + ".sha256"), t.with_suffix(".sha256"), *t.parent.glob(t.name.split(".")[0] + "*.sha256")]:
        if f.exists():
            for line in f.read_text().splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1].lstrip("*") == t.name:
                    return parts[0]
    return sha256_file(t)


def _verifier_digests(vdir: str) -> dict:
    code = ("import json,sys; sys.path.insert(0, sys.argv[1]); import run_checks; "
            "print(json.dumps(run_checks.verifier_digests(), sort_keys=True))")
    out = subprocess.run([sys.executable, "-c", code, vdir], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def _parents(cr: dict, names: list[str], runs: list[int]) -> dict:
    """Frozen identity of the parent links of every run: realpath and stat of each terminal file."""
    runs_dir = Path(cr["version_dir"]) / "runs"

    def one(r):
        out = {}
        for name in names:
            p = runs_dir / str(r) / name
            if not p.is_symlink():
                out[name] = {"error": "not a link to a parent terminal"}
                continue
            real = Path(os.path.realpath(p))
            files = sorted(x for x in real.rglob("*") if x.is_file()) if real.is_dir() else [real]
            out[name] = {"realpath": str(real), "files": {str(f.relative_to(real)) if real.is_dir() else ".": stat_of(f)
                                                          for f in files}}
        return str(r), out
    with ThreadPoolExecutor(16) as ex:
        return dict(ex.map(one, runs))


def _carried(cr: dict, objects: list[str]) -> dict:
    out = {}
    for obj in objects:
        g = GROUP[obj]
        d = Path(cr["version_dir"]) / "parquet" / g / obj
        out[obj] = {str(f): stat_of(f) | {"is_link": f.is_symlink()} for f in sorted(d.glob("*.parquet"))}
    return out


def freeze(spec_path: str, out: str) -> int:
    spec = yaml.safe_load(Path(spec_path).read_text())
    state = rp.git_state(REPO)
    err = rp.pushed_commit_error(state, None, spec["approved_remote"], spec["approved_ref"])
    if err:
        print(f"freeze: {err}")
        return 1
    env_setup = REPO / "scripts" / "cli" / "env_setup.yaml"
    rec = {"schema": SCHEMA, "written_utc": datetime.datetime.utcnow().isoformat() + "Z",
           "decision": spec.get("decision"), "approved_sha": state["sha"], "approved_ref": spec["approved_ref"],
           "approved_remote": spec["approved_remote"], "env_setup_sha256": sha256_file(env_setup),
           "env_setup_redacted_sha256": rp.redacted_sha256(env_setup), "images": {},
           "verifier": {"dir": spec["verifier_dir"], "digests": _verifier_digests(spec["verifier_dir"])},
           "budget": {"cap_node_hours": spec.get("cap_node_hours", 1100), "retry_allowance": spec.get("retry_allowance", 0.15),
                      "rates_node_hours_per_unit": spec.get("rates_node_hours_per_unit", {}),
                      "ledger": str(Path(out).with_name(Path(out).stem + "_ledger.jsonl"))},
           "approved_resources": spec.get("approved_resources", {}), "campaigns": {}}
    problems = []
    for cname, cs in spec["campaigns"].items():
        cr = {"stages": {}}
        for stage in STAGES:
            p = Path(cs["configs"][stage]).resolve()
            raw = yaml.safe_load(p.read_text())
            eff = _effective_config(p)
            if eff.get("stage") != stage or eff.get("campaign") != cname:
                problems.append(f"{p}: stage/campaign {eff.get('stage')}/{eff.get('campaign')} != {stage}/{cname}")
            import cli_utils
            container, tarball = cli_utils.resolve_stage_container(eff, stage)
            override = ((eff.get("common") or {}).get("stage_containers") or {}).get(stage) or {}
            if override.get("container"):          # as run_runs_interactive.sh: the override's own archive or none
                tarball = override.get("container_tarball")
            cr["stages"][stage] = {"config": str(p), "config_sha256": sha256_file(p),
                                   "effective_config_sha256": rp.effective_config_sha256(eff),
                                   "container": container, "tarball": tarball}
            if container not in rec["images"]:
                iid = rp.image_id(container)
                if not iid or len(iid) != 64:
                    problems.append(f"{container}: full image id unknown ({iid})")
                rec["images"][container] = {"id": iid, "tarball": tarball, "tarball_sha256": _tarball_sha256(tarball),
                                            "registry_digest": (spec.get("registry_digests") or {}).get(container)}
            vdir = Path(cli_utils.get_version_directory(eff))
            cr.setdefault("version_dir", str(vdir))
            if str(vdir) != cr["version_dir"]:
                problems.append(f"{p}: version dir {vdir} != {cr['version_dir']}")
            if stage == "digitization":
                cr["n_runs"] = int(raw["job_config"]["n_runs"])
            if stage == "package_parquet":
                cr["run_size"] = int(raw["run_size"])
                cr["chunk_size"] = int(raw["chunk_size"])
                cr["release_events"] = int(raw["release_events"])
                cr["objects"] = list(raw["objects"])
                cr["prefix"] = f"{raw['campaign']}.{raw['dataset']}.{raw['version']}"
        if cr["n_runs"] * cr["run_size"] < cr["release_events"]:
            problems.append(f"{cname}: {cr['n_runs']} runs x {cr['run_size']} < release {cr['release_events']}")
        cr["parent_names"] = list(cs.get("parents", [SIM_INPUT]))
        cr["carried_objects"] = list(cs.get("carried", []))
        # rehearsal only: a subset of the units (production specs leave unit_ranges out)
        cr["unit_ranges"] = {k: [int(v[0]), int(v[1])] for k, v in (cs.get("unit_ranges") or {}).items()}
        cr["require_protected_parents"] = bool(cs.get("require_protected_parents", True))
        runs_needed = set(units_of(cr, "digitization"))
        for i in units_of(cr, "package_parquet"):
            runs_needed |= {c["run"] for c in contributors(i, cr["chunk_size"], cr["release_events"], cr["run_size"])}
        cr["parents"] = _parents(cr, cr["parent_names"], sorted(runs_needed))
        cr["carried"] = _carried(cr, cr["carried_objects"])
        for r, ps in cr["parents"].items():
            for name, e in ps.items():
                if "error" in e:
                    problems.append(f"{cname} run {r} {name}: {e['error']}")
                elif cr["require_protected_parents"] and any(f["owner_writable"] for f in e["files"].values()):
                    problems.append(f"{cname} run {r} {name}: parent terminal is writable (not protected)")
        for obj, fs in cr["carried"].items():
            if not fs:
                problems.append(f"{cname}: no carried {obj} shards")
            if cr["require_protected_parents"] and any(f["owner_writable"] for f in fs.values()):
                problems.append(f"{cname}: a carried {obj} shard is writable (not protected)")
        cr["expected_units"] = {s: [units_of(cr, s)[0], units_of(cr, s)[-1], len(units_of(cr, s))] for s in STAGES}
        rec["campaigns"][cname] = cr
    if problems:
        print("\n".join(problems))
        print(f"freeze: {len(problems)} problem(s); nothing written")
        return 1
    rec["digest"] = record_digest(rec)
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print(f"froze {out}: sha {state['sha']}, digest {rec['digest']}")
    return 0


def check(rec: dict, images: bool = True) -> list[str]:
    """Live identity against the frozen record (run before every dispatch and release step)."""
    probs = []
    state = rp.git_state(REPO)
    err = rp.pushed_commit_error(state, rec["approved_sha"], rec["approved_remote"], rec["approved_ref"])
    if err:
        probs.append(f"code: {err}")
    env_setup = REPO / "scripts" / "cli" / "env_setup.yaml"
    if sha256_file(env_setup) != rec["env_setup_sha256"]:
        probs.append("env_setup.yaml changed since freezing")
    try:
        if _verifier_digests(rec["verifier"]["dir"]) != rec["verifier"]["digests"]:
            probs.append("verifier files changed since freezing")
    except subprocess.CalledProcessError as e:
        probs.append(f"verifier digests: {e}")
    for cname, cr in rec["campaigns"].items():
        for stage, s in cr["stages"].items():
            p = Path(s["config"])
            if not p.exists() or sha256_file(p) != s["config_sha256"]:
                probs.append(f"{cname}/{stage}: config {p} changed since freezing")
                continue
            if rp.effective_config_sha256(_effective_config(p)) != s["effective_config_sha256"]:
                probs.append(f"{cname}/{stage}: effective config changed (site defaults?)")
    if images:
        for name, im in rec["images"].items():
            got = rp.image_id(name)
            if got != im["id"]:
                probs.append(f"image {name}: {got} != frozen {im['id']}")
    return probs


# ------------------------------------------------------------------------------------------------ unit state
def unit_state(rec: dict, cr: dict, stage: str, unit: int, jobs) -> dict:
    d, stem = prov_dir(cr, stage, unit)
    path, last = rp.latest_record(d, stem)
    lock = d / f"{stem}.lock"
    st = {"unit": unit, "record": str(path) if path else None}
    if lock.exists():
        owner = lock_owner(lock)
        alive = lock_alive(owner, jobs)
        st["lock"] = owner
        st["state"] = "running" if alive is not False else "stale_lock"
        if alive is None:
            st["state"] = "lock_undecided"
        return st
    if last is None:
        st["state"] = "missing"
        return st
    if rp.is_accepted(last, rec["approved_sha"]):
        fz = cr["stages"][stage]
        bad = []
        if last.get("effective_config_sha256") != fz["effective_config_sha256"]:
            bad.append("effective config differs from the frozen one")
        im = rec["images"].get(fz["container"], {})
        if last.get("image_id") != im.get("id"):
            bad.append(f"image {last.get('image_id')} is not the frozen {im.get('id')}")
        st["state"] = "accepted" if not bad else "mismatch"
        if bad:
            st["why"] = bad
        st["rec"] = last
        return st
    if rp.is_accepted(last):
        st["state"], st["why"] = "mismatch", [f"accepted on {(last.get('git') or {}).get('sha')}, not the approved SHA"]
        return st
    status = last.get("status")
    st["state"] = "interrupted" if status in ("started", "promoting", "interrupted") else "failed"
    st["why"] = [f"{status}/{last.get('return_code')}"]
    return st


def _output_index(rec_: dict, base: Path) -> dict:
    """realpath -> output entry of an accepted record."""
    out = {}
    for e in rec_.get("outputs") or []:
        if e.get("realpath"):
            out[e["realpath"]] = e
    return out


def _current(e: dict, base: Path) -> str | None:
    """Is a recorded output still in place (same terminal, size and mtime)? Reason if not."""
    p = base / e["path"]
    try:
        real = os.path.realpath(p)
        if real != e["realpath"]:
            return f"{e['path']} now resolves to {real}"
        if os.path.getsize(real) != e.get("size") or abs(os.path.getmtime(real) - e.get("mtime", -1)) > 1e-6:
            return f"{e['path']} changed (size or mtime) since its record"
    except OSError as x:
        return f"{e['path']}: {x}"
    return None


def _parent_current(cr: dict, run: int, name: str) -> list[str]:
    fz = cr["parents"].get(str(run), {}).get(name)
    if fz is None:
        return [f"run {run} {name}: not frozen"]
    p = Path(cr["version_dir"]) / "runs" / str(run) / name
    if os.path.realpath(p) != fz["realpath"]:
        return [f"run {run} {name}: link now resolves to {os.path.realpath(p)}"]
    probs = []
    for rel, f in fz["files"].items():
        q = Path(fz["realpath"]) if rel == "." else Path(fz["realpath"]) / rel
        try:
            now = stat_of(q)
        except OSError as x:
            probs.append(f"run {run} {name}/{rel}: {x}")
            continue
        if any(now[k] != f[k] for k in ("size", "mtime_ns", "inode")):
            probs.append(f"run {run} {name}/{rel}: stat differs from the frozen parent")
    return probs


def upstream_problems(rec: dict, cr: dict, stage: str, unit: int, jobs, cache: dict) -> list[str]:
    """Barrier: every upstream unit accepted (approved SHA, frozen config and image) and its outputs in place;
    parents and carried inputs as frozen."""
    def acc(s, u):
        key = (s, u)
        if key not in cache:
            cache[key] = unit_state(rec, cr, s, u, jobs)
        return cache[key]
    runs_dir = Path(cr["version_dir"]) / "runs"
    probs = []
    g4 = SIM_INPUT in cr["parent_names"]
    if stage == "digitization":
        return _parent_current(cr, unit, SIM_INPUT) if g4 else []
    if stage in ("pandora_reco", "reco_tables"):
        up = "digitization" if stage == "pandora_reco" else "pandora_reco"
        s = acc(up, unit)
        if s["state"] != "accepted":
            return [f"{up} run {unit}: {s['state']} {s.get('why', '')}"]
        for e in s["rec"].get("outputs") or []:
            why = _current(e, runs_dir / str(unit))
            if why:
                probs.append(f"{up} run {unit}: {why}")
        if stage == "reco_tables" and g4:
            probs += _parent_current(cr, unit, SIM_INPUT)
        return probs
    # packaging: every contributing run, both upstream stages, the parents, the carried truth
    need_rt = bool(RECO_TABLE_OBJECTS & set(cr["objects"]))
    for c in contributors(unit, cr["chunk_size"], cr["release_events"], cr["run_size"]):
        r = c["run"]
        for up in ["digitization"] + (["reco_tables"] if need_rt else []):
            s = acc(up, r)
            if s["state"] != "accepted":
                probs.append(f"{up} run {r}: {s['state']} {s.get('why', '')}")
                continue
            for e in s["rec"].get("outputs") or []:
                why = _current(e, runs_dir / str(r))
                if why:
                    probs.append(f"{up} run {r}: {why}")
        for name in cr["parent_names"]:
            probs += _parent_current(cr, r, name)
    s0, e0 = chunk_window(unit, cr["chunk_size"], cr["release_events"])
    for obj, fs in cr["carried"].items():
        cover = [(f, m) for f, m in fs.items() if _overlaps(f, s0, e0)]
        if not cover:
            probs.append(f"carried {obj}: no frozen shard covers events {s0}-{e0}")
        for f, m in cover:
            try:
                now = stat_of(f)
            except OSError as x:
                probs.append(f"carried {Path(f).name}: {x}")
                continue
            if any(now[k] != m[k] for k in ("realpath", "size", "mtime_ns", "inode")):
                probs.append(f"carried {Path(f).name}: differs from the frozen inventory")
    return probs


def _window_of(name: str):
    import re
    m = re.search(r"\.events(\d+)-(\d+)\.parquet$", name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _overlaps(path: str, s: int, e: int) -> bool:
    w = _window_of(Path(path).name)
    return bool(w) and w[0] <= e and w[1] >= s


# ------------------------------------------------------------------------------------------------ audit
def audit_unit(rec: dict, cr: dict, stage: str, unit: int, jobs, cache: dict) -> list[str]:
    """Exact input binding of an ACCEPTED unit: what its record (or chunk manifest) says it consumed equals the
    accepted upstream outputs (sha256), the frozen parents (stat for the Geant4 file, sha256 otherwise) and the
    frozen carried shards; every contributing run of a chunk appears among the inputs of every object."""
    st = unit_state(rec, cr, stage, unit, jobs)
    if st["state"] != "accepted":
        return [f"{stage} {unit}: not accepted ({st['state']})"]
    me = st["rec"]
    runs_dir = Path(cr["version_dir"]) / "runs"

    def up(s, u):
        key = (s, u)
        if key not in cache:
            cache[key] = unit_state(rec, cr, s, u, jobs)
        return cache[key]

    def producer_sha(realpath: str, runs: list[int], stages: list[str]):
        for r in runs:
            for s in stages:
                u = up(s, r)
                if u["state"] != "accepted":
                    continue
                e = _output_index(u["rec"], runs_dir / str(r)).get(realpath)
                if e:
                    return e.get("sha256"), f"{s} run {r}"
        return None, None

    def parent_entry(realpath: str, runs: list[int]):
        for r in runs:
            for name in cr["parent_names"]:
                fz = cr["parents"].get(str(r), {}).get(name) or {}
                for rel, f in (fz.get("files") or {}).items():
                    if f["realpath"] == realpath:
                        return f
        return None

    probs = []
    if stage in RUN_STAGES:
        upstage = {"pandora_reco": "digitization", "reco_tables": "pandora_reco"}.get(stage)
        for e in me.get("inputs") or []:
            real = e.get("realpath")
            if Path(e.get("path", "")).name == SIM_INPUT or e.get("path") == SIM_INPUT:
                f = parent_entry(real, [unit])
                if f is None or e.get("size") != f["size"]:
                    probs.append(f"{stage} run {unit}: consumed {e.get('path')} is not the frozen parent")
                continue
            sha, who = producer_sha(real, [unit], [upstage] if upstage else [])
            if sha is None:
                probs.append(f"{stage} run {unit}: input {e.get('path')} has no accepted producer")
            elif sha != e.get("sha256"):
                probs.append(f"{stage} run {unit}: input {e.get('path')} differs from the accepted {who} output")
        return probs
    man = me.get("container_record") or {}
    inputs = man.get("inputs") or {}
    contrib = contributors(unit, cr["chunk_size"], cr["release_events"], cr["run_size"])
    runs = [c["run"] for c in contrib]
    carried_index = {m["realpath"]: (f, m) for o in cr["carried"].values() for f, m in o.items()}
    seen_runs = {}
    for path, meta in inputs.items():
        real = meta.get("realpath") or os.path.realpath(path)
        parts = Path(path).parts
        r = int(parts[parts.index("runs") + 1]) if "runs" in parts else None
        if r is not None:
            sub = parts[parts.index("runs") + 2] if len(parts) > parts.index("runs") + 2 else ""
            seen_runs.setdefault(r, set()).add(sub)
        if real in carried_index:
            m = carried_index[real][1]
            if meta.get("size") != m["size"] or meta.get("inode") != m["inode"]:
                probs.append(f"chunk {unit}: carried {Path(path).name} differs from the frozen inventory")
            continue
        f = parent_entry(real, runs)
        if f is not None:
            if meta.get("size") != f["size"] or meta.get("inode") != f["inode"] or meta.get("mtime_ns") != f["mtime_ns"]:
                probs.append(f"chunk {unit}: parent {path} differs from the frozen parent")
            continue
        sha, who = producer_sha(real, [r] if r is not None else runs, ["digitization", "reco_tables"])
        if sha is None:
            probs.append(f"chunk {unit}: input {path} has no accepted producer, frozen parent or carried shard")
        elif sha != meta.get("sha256"):
            probs.append(f"chunk {unit}: input {path} differs from the accepted {who} output")
    src_of = {"vertices": "performance_vertexing.root", "generator_particles": "generator_particles"}
    for c in contrib:
        got = seen_runs.get(c["run"], set())
        for obj in cr["objects"]:
            if src_of.get(obj, obj) not in got:
                probs.append(f"chunk {unit}: no {obj} input of contributing run {c['run']} in the manifest")
    extra = sorted(set(seen_runs) - set(runs))
    if extra:
        probs.append(f"chunk {unit}: inputs of runs {extra} outside the chunk's contributors {runs}")
    if man.get("start_event") != chunk_window(unit, cr["chunk_size"], cr["release_events"])[0]:
        probs.append(f"chunk {unit}: manifest window {man.get('start_event')}-{man.get('end_event')} is not the chunk's")
    return probs


# ------------------------------------------------------------------------------------------------ reconcile
def reconcile(rec: dict, cname: str, apply: bool, purge_failed: bool) -> int:
    cr = camp(rec, cname)
    jobs = live_jobs()
    v = Path(cr["version_dir"])
    actions, undecided = [], []
    held = set()        # (stage, unit) whose lock stays: live or undecided owner
    for stage in STAGES:
        for u in units_of(cr, stage):
            d, stem = prov_dir(cr, stage, u)
            lock = d / f"{stem}.lock"
            path, last = rp.latest_record(d, stem)
            if lock.exists():
                owner = lock_owner(lock)
                alive = lock_alive(owner, jobs)
                if alive is None:
                    undecided.append(f"{lock}: owner {owner} cannot be checked from here")
                    held.add((stage, u))
                    continue
                if alive:
                    held.add((stage, u))
                    continue
                actions.append(("unlock", lock, path, last))
            elif last and last.get("status") in ("started", "promoting"):
                actions.append(("interrupt", None, path, last))
    for tmp in glob.glob(str(v / "parquet" / "*" / "*" / ".*.tmp")):
        w = _window_of(Path(tmp).name.rsplit(".", 2)[0].lstrip("."))
        if w is None:
            continue
        # a chunk whose stale lock is removed in this pass counts as unlocked (found by pilot_r2 T4 on aa37663:
        # the temporary shard of the interrupted chunk stayed until a second pass)
        if ("package_parquet", w[0] // cr["chunk_size"]) not in held:
            actions.append(("tmp", Path(tmp), None, None))
    for tmp in glob.glob(str(v / "runs" / "*" / "provenance" / ".*.tmp")) + glob.glob(str(v / "provenance" / "*" / ".*.tmp")):
        actions.append(("tmp", Path(tmp), None, None))
    if purge_failed:
        for ad in sorted(glob.glob(str(v / ".attempts" / "*" / "*.a*"))):
            stage = Path(ad).parent.name
            u, k = Path(ad).name.split(".a")
            d, stem = prov_dir(cr, stage, int(u))
            recf = d / f"{stem}_attempt{k}.json"
            r_ = json.loads(recf.read_text()) if recf.exists() else {}
            if (d / f"{stem}.lock").exists() or rp.is_accepted(r_):
                continue
            actions.append(("purge", Path(ad), recf, r_))
    now = datetime.datetime.utcnow().isoformat() + "Z"
    for kind, p, recp, last in actions:
        print(f"{'do' if apply else 'would'} {kind} {p or recp}")
        if not apply:
            continue
        if kind in ("unlock", "interrupt"):
            if recp is not None and last and last.get("status") in ("started", "promoting"):
                rp.set_status(recp, "interrupted", reconciled_utc=now,
                              reconciled_reason=f"{kind}: owner job ended without a final record")
            if p is not None:
                p.unlink()
        elif kind == "tmp":
            p.unlink(missing_ok=True)
        elif kind == "purge":
            attempt_isolation.purge(p)
    for u in undecided:
        print(f"undecided {u}")
    print(f"reconcile {cname}: {len(actions)} action(s){'' if apply else ' (dry run)'}, {len(undecided)} undecided")
    return 1 if undecided else 0


# ------------------------------------------------------------------------------------------------ budget
def _sacct(ids: list[str]) -> dict:
    if not ids:
        return {}
    out = subprocess.run(["sacct", "-X", "-n", "-P", "-j", ",".join(ids), "-o", "JobID,State,ElapsedRaw,NNodes,TimelimitRaw"],
                         capture_output=True, text=True, timeout=300)
    res = {}
    for line in out.stdout.splitlines():
        jid, state, el, nn, tl = (line.split("|") + [""] * 5)[:5]
        res[jid.split("_")[0]] = {"state": state, "elapsed_s": int(el or 0), "nodes": int(nn or 0),
                                  "limit_min": int(tl) if tl.isdigit() else None}
    return res


def budget(rec: dict, add_node_hours: float = 0.0, verbose: bool = True) -> tuple[bool, dict]:
    """Consumed (every ledger job: elapsed x nodes when finished, full time limit x nodes while pending or
    running) + forecast of the units not yet accepted (frozen per-unit rates x (1 + retry allowance))
    + the block about to be dispatched <= cap."""
    b = rec["budget"]
    ledger = Path(b["ledger"])
    entries = [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()] if ledger.exists() else []
    acct = _sacct([e["job_id"] for e in entries])
    consumed = 0.0
    for e in entries:
        a = acct.get(str(e["job_id"]).split("_")[0])
        if a and a["state"].split()[0] not in ("PENDING", "RUNNING", "REQUEUED", "SUSPENDED"):
            consumed += a["elapsed_s"] * a["nodes"] / 3600
        else:
            consumed += e["nodes"] * e["time_limit_h"]
    jobs = live_jobs()
    forecast, detail = 0.0, {}
    for cname, cr in rec["campaigns"].items():
        rates = b["rates_node_hours_per_unit"].get(cname, {})
        for stage in STAGES + ["verify"]:
            if stage == "verify":
                n_left = len(units_of(cr, "package_parquet"))
            else:
                n_left = sum(1 for u in units_of(cr, stage) if unit_state(rec, cr, stage, u, jobs)["state"] != "accepted")
            nh = n_left * float(rates.get(stage, 0.0)) * (1 + b["retry_allowance"])
            detail[f"{cname}/{stage}"] = {"units_left": n_left, "node_hours": round(nh, 2)}
            forecast += nh
    total = consumed + forecast + add_node_hours
    ok = total <= b["cap_node_hours"]
    rep = {"consumed": round(consumed, 2), "forecast": round(forecast, 2), "this_block": add_node_hours,
           "total": round(total, 2), "cap": b["cap_node_hours"], "ok": ok, "detail": detail}
    if verbose:
        print(json.dumps(rep, indent=1))
    return ok, rep


# ------------------------------------------------------------------------------------------------ release (E7)
def _published_dir(cr: dict, root: str) -> Path:
    v = Path(cr["version_dir"])
    return Path(root) / v.parent.parent.name / v.parent.name / v.name


def release_preflight(rec, cname, published_root, min_free_inodes) -> list[str]:
    cr = camp(rec, cname)
    v = Path(cr["version_dir"])
    probs = check(rec, images=False)
    if not v.is_dir():
        probs.append(f"{v} missing")
    if not (v / rp.MARKER).exists():
        probs.append(f"{v}: not finalized ({rp.MARKER} missing)")
    tgt = _published_dir(cr, published_root)
    if tgt.exists():
        probs.append(f"{tgt} already exists")
    if os.path.realpath(published_root) == os.path.realpath(v.parent.parent.parent):
        probs.append("published root equals the staging root")
    try:
        st = os.statvfs(published_root)
        if st.f_favail < min_free_inodes:
            probs.append(f"{published_root}: {st.f_favail} free inodes < {min_free_inodes}")
        if os.stat(published_root).st_dev != os.stat(v).st_dev:
            probs.append(f"{published_root} and {v} are on different file systems (no atomic rename)")
    except OSError as e:
        probs.append(f"{published_root}: {e}")
    return probs


def release_inventory(rec, cname, out, verification_dir, decision_files, jobs_n) -> int:
    """RELEASE_MANIFEST: every produced and carried shard of every release object (sha256 recomputed now and
    compared with the accepted chunk manifests / the frozen inventory), exact tiling of the release events per
    object, the verification results of every chunk and of the release-wide reductions (verifier digests equal
    to the frozen ones), the decision records, and where the run products are retained (not published)."""
    cr = camp(rec, cname)
    v = Path(cr["version_dir"])
    probs = []
    if not (v / rp.MARKER).exists():
        probs.append("version not finalized")
    jobs = live_jobs()
    n_chunks = len(units_of(cr, "package_parquet"))
    produced = {}
    for i in range(n_chunks):
        st = unit_state(rec, cr, "package_parquet", i, jobs)
        if st["state"] != "accepted":
            probs.append(f"chunk {i}: {st['state']}")
            continue
        for f, m in ((st["rec"].get("container_record") or {}).get("outputs") or {}).items():
            produced[f] = {"chunk": i, "size": m["size"], "sha256_manifest": m["sha256"]}
        probs += [f"audit: {p}" for p in audit_unit(rec, cr, "package_parquet", i, jobs, {})]
    carried = {f: {"object": o, **m} for o, fs in cr["carried"].items() for f, m in fs.items()}
    files = sorted(set(produced) | set(carried))

    def h(f):
        try:
            return f, sha256_file(f)
        except OSError as e:
            return f, f"error: {e}"
    with ThreadPoolExecutor(jobs_n) as ex:
        sha = dict(ex.map(h, files))
    for f, m in produced.items():
        if sha[f] != m["sha256_manifest"]:
            probs.append(f"{Path(f).name}: sha256 {sha[f]} != chunk manifest")
    for f, m in carried.items():
        now = stat_of(f)
        if any(now[k] != m[k] for k in ("realpath", "size", "mtime_ns", "inode")):
            probs.append(f"carried {Path(f).name}: changed since freezing")
    R = cr["release_events"]
    for obj in cr["objects"] + cr["carried_objects"]:
        d = v / "parquet" / GROUP[obj] / obj
        present = sorted(str(p) for p in d.iterdir()) if d.is_dir() else []
        wins = sorted(_window_of(Path(p).name) or (-1, -1) for p in present)
        pos = 0
        for a, b in wins:
            if a != pos:
                probs.append(f"{obj}: windows do not tile the release at event {pos} (next {a}-{b})")
                break
            pos = b + 1
        if pos != R:
            probs.append(f"{obj}: windows end at {pos - 1}, release is 0-{R - 1}")
        unknown = [p for p in present if p not in produced and p not in carried]
        if unknown:
            probs.append(f"{obj}: {len(unknown)} file(s) neither produced nor carried, e.g. {unknown[0]}")
    ver = {}
    vd = Path(verification_dir)
    for i in range(n_chunks):
        f = vd / f"chunk{i}.json"
        if not f.exists():
            probs.append(f"verification of chunk {i} missing")
            continue
        doc = json.loads(f.read_text())
        if not doc.get("all_pass"):
            probs.append(f"verification of chunk {i} failed")
        if (doc.get("meta") or {}).get("verifier_sha256") != rec["verifier"]["digests"]:
            probs.append(f"verification of chunk {i} ran another verifier")
        ver[str(i)] = sha256_file(f)
    red = vd / "release_reductions.json"
    if not red.exists() or not json.loads(red.read_text()).get("all_pass"):
        probs.append("release-wide reductions missing or failed")
    else:
        ver["release_reductions"] = sha256_file(red)
    dec = {}
    for f in decision_files:
        dec[str(f)] = sha256_file(f)
    if probs:
        print("\n".join(probs[:200]))
        print(f"inventory: {len(probs)} problem(s); no manifest written")
        return 1
    man = {"schema": "colliderml.v23.release/1", "campaign": cname, "launch_record_digest": rec["digest"],
           "approved_sha": rec["approved_sha"], "written_utc": datetime.datetime.utcnow().isoformat() + "Z",
           "release_events": R, "objects": cr["objects"] + cr["carried_objects"],
           "files": {str(Path(f).relative_to(v / "parquet")): {"sha256": sha[f], "size": os.path.getsize(f),
                                                               "kind": "produced" if f in produced else "carried",
                                                               "realpath": os.path.realpath(f)} for f in files},
           "verification_sha256": ver, "decisions_sha256": dec,
           "run_products_retained_privately": str(v / "runs")}
    man["digest"] = hashlib.sha256(canonical(man)).hexdigest()
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps(man, indent=1, sort_keys=True) + "\n")
    print(f"inventory: {len(files)} files, manifest {out} ({man['digest']})")
    return 0


def release_publish(rec, cname, manifest, published_root, min_free_inodes) -> int:
    """Read-only shards, then ONE rename of <staging version>/parquet into the published version dir (the
    publication point); the run products stay in staging. Any failure before the rename undoes the skeleton."""
    cr = camp(rec, cname)
    v = Path(cr["version_dir"])
    man = json.loads(Path(manifest).read_text())
    probs = release_preflight(rec, cname, published_root, min_free_inodes)
    if man.get("launch_record_digest") != rec["digest"]:
        probs.append("manifest belongs to another launch record")
    if probs:
        print("\n".join(probs))
        return 1
    tgt = _published_dir(cr, published_root)
    freeze = v / rp.FREEZE
    if not freeze.exists():
        print(f"{freeze} missing: the version must stay frozen after finalization")
        return 1
    for rel in man["files"]:
        p = v / "parquet" / rel
        if not p.is_symlink():
            os.chmod(p, 0o444)
    tgt.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(tgt)
    try:
        (tgt / "RELEASE_MANIFEST.json").write_text(Path(manifest).read_text())
        os.rename(v / "parquet", tgt / "parquet")       # publication point
    except BaseException:
        (tgt / "RELEASE_MANIFEST.json").unlink(missing_ok=True)
        os.rmdir(tgt)
        raise
    with open(tgt / ".published", "x") as f:
        json.dump({"from": str(v), "manifest_digest": man["digest"], "host": socket.gethostname(),
                   "utc": datetime.datetime.utcnow().isoformat() + "Z"}, f)
    print(f"published {tgt}")
    return 0


def release_verify_bytes(rec, cname, published_root, remote_listing, jobs_n) -> int:
    cr = camp(rec, cname)
    tgt = _published_dir(cr, published_root)
    man = json.loads((tgt / "RELEASE_MANIFEST.json").read_text())
    probs = []
    present = {str(p.relative_to(tgt / "parquet")) for p in (tgt / "parquet").rglob("*") if p.is_file() or p.is_symlink()}
    want = set(man["files"])
    if present != want:
        probs.append(f"published file set differs: {len(present - want)} extra, {len(want - present)} missing")
    with ThreadPoolExecutor(jobs_n) as ex:
        got = dict(ex.map(lambda r: (r, sha256_file(tgt / "parquet" / r)), sorted(want & present)))
    probs += [f"{r}: sha256 differs from the manifest" for r, s in got.items() if s != man["files"][r]["sha256"]]
    if remote_listing:
        remote = json.loads(Path(remote_listing).read_text())
        probs += [f"portal {r}: {remote.get(r)} != {m['sha256']}" for r, m in man["files"].items() if remote.get(r) != m["sha256"]]
    print("\n".join(probs[:100]) or f"verify-bytes: {len(got)} files equal to the manifest")
    return 1 if probs else 0


def release_rollback(rec, cname, published_root) -> int:
    cr = camp(rec, cname)
    v = Path(cr["version_dir"])
    tgt = _published_dir(cr, published_root)
    pub = tgt / ".published"
    if not pub.exists() or json.loads(pub.read_text()).get("from") != str(v):
        print(f"{tgt} was not published from {v} by this tool; refusing")
        return 1
    if (v / "parquet").exists():
        print(f"{v / 'parquet'} exists; refusing")
        return 1
    os.rename(tgt / "parquet", v / "parquet")
    (tgt / "RELEASE_MANIFEST.json").unlink()
    pub.unlink()
    os.rmdir(tgt)
    print(f"rolled back {tgt} -> {v / 'parquet'}")
    return 0


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("freeze"); p.add_argument("spec"); p.add_argument("--out", required=True)
    p = sub.add_parser("check"); p.add_argument("record"); p.add_argument("--no-images", action="store_true")
    p = sub.add_parser("expect"); p.add_argument("record"); p.add_argument("--campaign")
    for name in ("status", "plan", "audit"):
        p = sub.add_parser(name); p.add_argument("record"); p.add_argument("--campaign", required=True)
        p.add_argument("--stage", required=True, choices=STAGES)
        p.add_argument("--first", type=int); p.add_argument("--last", type=int)
        p.add_argument("--units", help="comma-separated units (default: all expected)")
        if name == "plan":
            p.add_argument("--out", required=True)
    p = sub.add_parser("reconcile"); p.add_argument("record"); p.add_argument("--campaign", required=True)
    p.add_argument("--apply", action="store_true"); p.add_argument("--purge-failed", action="store_true")
    p = sub.add_parser("budget"); p.add_argument("record"); p.add_argument("--add-node-hours", type=float, default=0.0)
    p = sub.add_parser("dispatch"); p.add_argument("record"); p.add_argument("--campaign", required=True)
    p.add_argument("--stage", required=True, choices=STAGES); p.add_argument("--units", required=True)
    p.add_argument("--nodes", type=int, required=True); p.add_argument("--time", required=True)
    p.add_argument("--per-wave", type=int, required=True); p.add_argument("--qos", default="regular")
    p.add_argument("--account", default="m4958"); p.add_argument("--submit", action="store_true")
    p = sub.add_parser("release"); p.add_argument("step", choices=["preflight", "inventory", "publish", "verify-bytes", "rollback"])
    p.add_argument("record"); p.add_argument("--campaign", required=True); p.add_argument("--published-root")
    p.add_argument("--manifest"); p.add_argument("--verification-dir"); p.add_argument("--decision", action="append", default=[])
    p.add_argument("--remote-listing"); p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--min-free-inodes", type=int, default=50000)
    a = ap.parse_args(argv)

    if a.cmd == "freeze":
        return freeze(a.spec, a.out)
    rec = load_record(a.record)
    if a.cmd == "check":
        probs = check(rec, images=not a.no_images)
        print("\n".join(probs) or f"launch record {rec['digest']}: live identity ok")
        return 1 if probs else 0
    if a.cmd == "expect":
        for cname, cr in rec["campaigns"].items():
            if a.campaign and cname != a.campaign:
                continue
            for s in STAGES:
                us = units_of(cr, s)
                print(f"{cname} {s}: {len(us)} units {us[0]}..{us[-1]}")
            cu = units_of(cr, "package_parquet")
            for i in sorted({cu[0], cu[min(1, len(cu) - 1)], cu[-2] if len(cu) > 1 else cu[0], cu[-1]}):
                w = chunk_window(i, cr["chunk_size"], cr["release_events"])
                print(f"  chunk {i} events {w[0]}-{w[1]}: {contributors(i, cr['chunk_size'], cr['release_events'], cr['run_size'])}")
        return 0
    if a.cmd in ("status", "plan", "audit"):
        cr = camp(rec, a.campaign)
        us = units_of(cr, a.stage)
        if a.units:
            want = [int(x) for x in a.units.split(",")]
            bad = [u for u in want if u not in us]
            if bad:
                print(f"units {bad} are not expected units of {a.stage}")
                return 2
            us = want
        if a.first is not None or a.last is not None:
            us = [u for u in us if (a.first is None or u >= a.first) and (a.last is None or u <= a.last)]
        jobs = live_jobs()
        cache = {}
        if a.cmd == "status":
            counts = {}
            for u in us:
                s = unit_state(rec, cr, a.stage, u, jobs)
                counts[s["state"]] = counts.get(s["state"], 0) + 1
                if s["state"] != "accepted":
                    print(f"{a.stage} {u}: {s['state']} {s.get('why', '')}")
            print(json.dumps(counts))
            return 0 if counts.get("accepted", 0) == len(us) else 1
        if a.cmd == "audit":
            n_bad = 0
            for u in us:
                probs = audit_unit(rec, cr, a.stage, u, jobs, cache)
                for x in probs:
                    print(x)
                n_bad += bool(probs)
            print(f"audit {a.campaign}/{a.stage}: {len(us) - n_bad} of {len(us)} units bound to accepted inputs")
            return 1 if n_bad else 0
        todo, blocked = [], []
        for u in us:
            s = unit_state(rec, cr, a.stage, u, jobs)
            if s["state"] in ("accepted", "running"):
                continue
            if s["state"] in ("stale_lock", "lock_undecided", "mismatch"):
                blocked.append(f"{a.stage} {u}: {s['state']} {s.get('why', '')} (reconcile or explicit replacement first)")
                continue
            probs = upstream_problems(rec, cr, a.stage, u, jobs, cache)
            if probs:
                blocked.append(f"{a.stage} {u}: upstream not ready: {probs[:3]}")
            else:
                todo.append(u)
        Path(a.out).write_text("".join(f"{u}\n" for u in todo))
        for b in blocked:
            print(b)
        print(f"plan {a.campaign}/{a.stage}: {len(todo)} unit(s) to run -> {a.out}; {len(blocked)} blocked")
        return 1 if blocked else 0
    if a.cmd == "reconcile":
        return reconcile(rec, a.campaign, a.apply, a.purge_failed)
    if a.cmd == "budget":
        ok, _ = budget(rec, a.add_node_hours)
        return 0 if ok else 1
    if a.cmd == "dispatch":
        cr = camp(rec, a.campaign)
        units = [int(x) for x in Path(a.units).read_text().split()]
        probs = check(rec)
        jobs = live_jobs()
        cache = {}
        upstage = {"pandora_reco": "digitization", "reco_tables": "pandora_reco"}.get(a.stage)
        for u in units:
            s = unit_state(rec, cr, a.stage, u, jobs)
            if s["state"] in ("accepted", "running", "stale_lock", "lock_undecided", "mismatch"):
                probs.append(f"{a.stage} {u}: {s['state']}: not a unit to dispatch")
            probs += upstream_problems(rec, cr, a.stage, u, jobs, cache)
            ups = [(upstage, u)] if upstage else []
            if a.stage == "package_parquet":
                ups = [(s_, c["run"]) for c in contributors(u, cr["chunk_size"], cr["release_events"], cr["run_size"])
                       for s_ in ("digitization", "reco_tables")]
            for s_, r in ups:
                probs += audit_unit(rec, cr, s_, r, jobs, cache)
        h, m, s_ = (int(x) for x in a.time.split(":"))
        block_nh = a.nodes * (h + m / 60 + s_ / 3600)
        ok, rep = budget(rec, block_nh, verbose=False)
        if not ok:
            probs.append(f"budget: consumed {rep['consumed']} + forecast {rep['forecast']} + block {block_nh:.1f} "
                         f"> cap {rep['cap']}")
        cfg = cr["stages"][a.stage]["config"]
        cmd = ["sbatch", "-A", a.account, "-C", "cpu", "-q", a.qos, "-N", str(a.nodes), "-t", a.time,
               "-J", f"v23-{a.campaign}-{a.stage}", "--wrap",
               f"export COLLIDERML_APPROVED_SHA={rec['approved_sha']}; bash {REPO}/scripts/cli/run_blocks_multinode.sh "
               f"{cfg} --units {Path(a.units).resolve()} {a.per_wave}"]
        if probs:
            print("\n".join(probs[:100]))
            print(f"dispatch refused: {len(probs)} problem(s)")
            return 1
        print(" ".join(cmd))
        if not a.submit:
            return 0
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            print(out.stderr)
            return 1
        jid = out.stdout.split()[-1]
        with open(rec["budget"]["ledger"], "a") as f:
            f.write(json.dumps({"job_id": jid, "campaign": a.campaign, "stage": a.stage, "n_units": len(units),
                                "nodes": a.nodes, "time_limit_h": block_nh / a.nodes,
                                "submitted_utc": datetime.datetime.utcnow().isoformat() + "Z"}) + "\n")
        print(f"submitted {jid}")
        return 0
    if a.cmd == "release":
        if a.step == "preflight":
            probs = release_preflight(rec, a.campaign, a.published_root, a.min_free_inodes)
            print("\n".join(probs) or "preflight ok")
            return 1 if probs else 0
        if a.step == "inventory":
            return release_inventory(rec, a.campaign, a.manifest, a.verification_dir, a.decision, a.jobs)
        if a.step == "publish":
            return release_publish(rec, a.campaign, a.manifest, a.published_root, a.min_free_inodes)
        if a.step == "verify-bytes":
            return release_verify_bytes(rec, a.campaign, a.published_root, a.remote_listing, a.jobs)
        return release_rollback(rec, a.campaign, a.published_root)
    return 2


if __name__ == "__main__":
    sys.exit(main())
