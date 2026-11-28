#!/usr/bin/env python3
"""ttbar v23 production controller (PRODUCTION_PLAN.md section 8b: E2, E3, E5, E7).

Every production action reads one FROZEN launch record (``freeze``). Units, chunk contributors and the inputs
each unit consumes are enumerated here from the record's numbers (run size, run count, release size, chunk
size), independently of the stage code, and compared with what the attempt records and chunk manifests say.

  freeze       SPEC.yaml --out LAUNCH.json [--authorization AUTH.json]
               write the launch record (exclusive create; never overwritten); AUTH.json (outside the repo) names
               the authorized commit and the approved decision records
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
               (--stage verify: chunk verification blocks, scripts/production/verify_blocks.sh)
  unit-check   LAUNCH.json --config CFG --units a,b   (worker side, run_runs_interactive.sh: this node's identity
               and the per-unit dispatch rule before any write)
  verify-chunk LAUNCH.json --campaign C --chunk K     (one chunk with the frozen verifier and arguments)
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
import re
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
SMALL_PARENT_BYTES = 1 << 30    # parent files up to this size are frozen with their sha256
VERIFY = "verify"               # budgeted verification stage (unit = release chunk)
# sacct states after which a job holds no more nodes (anything else keeps the reservation at its full limit)
TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL",
                   "DEADLINE", "PREEMPTED", "REVOKED"}
# published next to parquet/ (VERIFICATION_PLAN_v3_1 public contents): the dataset card, the configs, the marker
PUBLIC_META = ("README.md", "configs", rp.MARKER)


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


def _layout(cr: dict, obj: str) -> tuple[str, str]:
    """(group, published name) of a packaged object, as the packager lays it out (its OBJECT_LAYOUT; the D8 truth
    object of truth_base is published as truth/particles)."""
    tb = cr.get("truth_base") or {}
    if obj == tb.get("object"):
        return "truth", "particles"
    try:
        sys.path.insert(0, str(REPO / "scripts" / "postprocessing"))
        from package_native_parquet import OBJECT_LAYOUT
        if obj in OBJECT_LAYOUT:
            return OBJECT_LAYOUT[obj]
    except Exception:  # noqa: BLE001  (pyarrow-less tooling: the controller's own map)
        pass
    return GROUP.get(obj, "truth"), obj


def obj_dir(cr: dict, obj: str) -> Path:
    g, name = _layout(cr, obj)
    return Path(cr["version_dir"]) / "parquet" / g / name


def driver_tarball(raw: dict, stage: str) -> str | None:
    """The image archive run_runs_interactive.sh loads for a stage, from the RAW config (site defaults are not
    applied there): the stage override's own archive when the override names a container, else common's."""
    common = raw.get("common") or {}
    over = (common.get("stage_containers") or {}).get(stage) or {}
    if over.get("container"):
        return over.get("container_tarball") or None
    return common.get("container_tarball") or None


def _truth_base(cs: dict, require_protected: bool) -> tuple[dict | None, list[str]]:
    """D8: the frozen base of the produced PU200 truth/particles (the Release 1 shards the packager completes):
    realpath, size, mtime, inode and sha256 of every shard."""
    tb = cs.get("truth_base")
    if not tb:
        return None, []
    d = Path(tb["dir"])
    files = sorted(d.glob("*.parquet"))
    probs = [] if files else [f"truth_base {d}: no shards"]

    def one(f):
        return str(f), stat_of(f) | {"sha256": sha256_file(f)}
    with ThreadPoolExecutor(16) as ex:
        shards = dict(ex.map(one, files))
    if require_protected and any(m["owner_writable"] for m in shards.values()):
        probs.append(f"truth_base {d}: a shard is writable (not protected)")
    return {"object": tb["object"], "dir": str(d), "shards": shards}, probs


def load_authorization(path: str | None) -> tuple[dict, list[str]]:
    """The lead's authorization (outside the repository, so the tracked spec is never edited to authorize a commit
    and the tree stays clean): {"approved_sha": 40 hex, "authorized_by", "date", "decision_records": {path: sha256}}.
    Every decision record must exist with that sha256."""
    if not path:
        return {}, []
    probs = []
    if Path(os.path.realpath(path)).is_relative_to(REPO):
        return {}, ["the authorization file must live outside the repository"]
    auth = json.loads(Path(path).read_text())
    if not re.fullmatch(r"[0-9a-f]{40}", str(auth.get("approved_sha", ""))):
        probs.append("authorization: approved_sha must be 40 hex digits")
    for k in ("authorized_by", "date"):
        if not auth.get(k):
            probs.append(f"authorization: {k} missing")
    recs = auth.get("decision_records") or {}
    if not recs:
        probs.append("authorization: no decision_records")
    for f, h in recs.items():
        if not Path(f).exists() or sha256_file(f) != h:
            probs.append(f"authorization: decision record {f} missing or not sha256 {str(h)[:12]}")
    return auth, probs


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
            def ent(f):
                e = stat_of(f)
                if e["size"] <= SMALL_PARENT_BYTES:     # e.g. generator_particles; the Geant4 file stays stat-only
                    e["sha256"] = sha256_file(f)
                return e
            out[name] = {"realpath": str(real), "files": {str(f.relative_to(real)) if real.is_dir() else ".": ent(f)
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


def freeze(spec_path: str, out: str, authorization: str | None = None) -> int:
    """authorization: a JSON file outside the repo {"approved_sha", "authorized_by", "date", "decision_records":
    {path: sha256}} (the lead's authorization); it takes precedence over the spec, so the tracked spec is never
    edited to authorize a commit (an edit would dirty the tree the freeze requires clean)."""
    spec = yaml.safe_load(Path(spec_path).read_text())
    auth, aprobs = load_authorization(authorization)
    if aprobs:
        print("\n".join(f"freeze: {x}" for x in aprobs))
        return 1
    if auth:
        spec["approved_sha"] = auth.get("approved_sha")
        spec["approved_decision_records"] = auth.get("decision_records") or {}
    state = rp.git_state(REPO)
    if not spec.get("approved_sha") or not re.fullmatch(r"[0-9a-f]{40}", str(spec.get("approved_sha"))):
        print("freeze: no authorized approved_sha (40 hex) in the authorization file or the spec")
        return 1
    err = rp.pushed_commit_error(state, spec["approved_sha"], spec["approved_remote"], spec["approved_ref"])
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
           "approved_resources": spec.get("approved_resources", {}), "campaigns": {},
           "accepted_shas": spec.get("accepted_shas") or {},
           "require_tracking_bound": bool(spec.get("require_tracking_bound", True)),
           "authorization": {k: auth.get(k) for k in ("approved_sha", "authorized_by", "date")} if auth else None,
           "authorization_sha256": sha256_file(authorization) if authorization else None,
           "approved_decision_records": spec.get("approved_decision_records") or {},
           "frozen_tracking_sample": spec.get("frozen_tracking_sample") or {}}
    problems = []
    import math
    for cname in spec["campaigns"]:
        rates = (spec.get("rates_node_hours_per_unit") or {}).get(cname) or {}
        for st_ in STAGES + ["verify"]:
            v = rates.get(st_)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0:
                problems.append(f"rate {cname}/{st_} = {v!r}: every stage needs a finite positive measured rate")
    for cname, cs in spec["campaigns"].items():
        cr = {"stages": {}}
        for stage in STAGES:
            p = Path(cs["configs"][stage]).resolve()
            raw = yaml.safe_load(p.read_text())
            eff = _effective_config(p)
            if eff.get("stage") != stage or eff.get("campaign") != cname:
                problems.append(f"{p}: stage/campaign {eff.get('stage')}/{eff.get('campaign')} != {stage}/{cname}")
            import cli_utils
            container, _ = cli_utils.resolve_stage_container(eff, stage)
            tarball = driver_tarball(raw, stage)       # what the driver loads (Codex re-review finding 6)
            cr["stages"][stage] = {"config": str(p), "config_sha256": sha256_file(p),
                                   "effective_config_sha256": rp.effective_config_sha256(eff),
                                   "container": container, "tarball": tarball}
            if container in rec["images"] and rec["images"][container]["tarball"] != tarball:
                problems.append(f"{container}: archive {tarball} of {cname}/{stage} differs from the archive "
                                f"{rec['images'][container]['tarball']} another stage loads for the same image")
            if container not in rec["images"]:
                iid = rp.image_id(container)
                want = (spec.get("required_image_ids") or {}).get(container)
                if not iid or len(iid) != 64:
                    problems.append(f"{container}: full image id unknown ({iid})")
                if want is None or iid != want:
                    problems.append(f"{container}: image id {iid} is not the required {want} (spec required_image_ids)")
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
                cr["package_flags"] = {k: raw.get(k) for k in ("track_covariance", "remap_particle_ids", "vertex_links",
                                                                "sim_input_name")}
                cr["package_flags"]["truth_completion_base"] = (raw.get("truth_completion") or {}).get("base_dir")
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
        cr["truth_base"], tprobs = _truth_base(cs, cr["require_protected_parents"])
        problems += [f"{cname}: {x}" for x in tprobs]
        if cr["truth_base"]:
            if cr["truth_base"]["object"] not in cr["objects"]:
                problems.append(f"{cname}: truth_base object {cr['truth_base']['object']} is not a packaged object")
            if "particles" in cr["carried_objects"]:
                problems.append(f"{cname}: truth/particles cannot be both carried and produced (truth_base)")
            tc = cr["package_flags"].get("truth_completion_base")
            if not tc or os.path.realpath(tc) != os.path.realpath(cr["truth_base"]["dir"]):
                problems.append(f"{cname}: the packager's truth_completion.base_dir {tc} is not the frozen truth_base")
        elif cr["package_flags"].get("truth_completion_base"):
            problems.append(f"{cname}: the packager completes truth from a base the spec does not freeze (truth_base)")
        cr["verification_dir"] = str(cs.get("verification_dir") or Path(cr["version_dir"]) / "verification")
        cr["verify"] = dict(cs.get("verify") or {})
        if not cr["verify"].get("particles_contract"):
            problems.append(f"{cname}: verify.particles_contract missing (the verifier's truth contract)")
        if rec["require_tracking_bound"] and not (rec["frozen_tracking_sample"] or {}).get(cname):
            problems.append(f"{cname}: frozen_tracking_sample missing (P5 run list)")
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
def accepted_shas(rec: dict, stage: str) -> list[str]:
    """Code SHAs whose attempts count as accepted for a stage: the approved SHA, plus the reuse the launch record
    names explicitly (rehearsal split-SHA reuse; the production spec names none)."""
    return [rec["approved_sha"]] + list((rec.get("accepted_shas") or {}).get(stage, []))


def unit_state(rec: dict, cr: dict, stage: str, unit: int, jobs) -> dict:
    d, stem = prov_dir(cr, stage, unit)
    paths = rp.attempt_records(d, stem)
    path, last = (paths[-1], rp.latest_record_at(paths[-1]) or {"status": "unreadable"}) if paths else (None, None)
    lock = d / f"{stem}.lock"
    st = {"unit": unit, "record": str(path) if path else None, "n_attempts": len(paths)}
    st["n_failed"] = sum(1 for p in paths if not rp.is_accepted(rp.latest_record_at(p)))
    if lock.exists():
        owner = lock_owner(lock)
        alive = lock_alive(owner, jobs)
        st["lock"] = owner
        st["state"] = "running" if alive else ("lock_undecided" if alive is None else "stale_lock")
        return st
    if last is None:
        st["state"] = "missing"
        return st
    if rp.is_accepted(last):
        fz = cr["stages"][stage]
        bad = []
        if (last.get("git") or {}).get("sha") not in accepted_shas(rec, stage):
            bad.append(f"accepted on {(last.get('git') or {}).get('sha')}, not an approved SHA")
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
    earlier = [p for p in paths[:-1] if rp.is_accepted(rp.latest_record_at(p))]
    status = last.get("status")
    if earlier:
        # an explicit replacement of an accepted unit did not complete: blocked until the operator replaces it
        # again (COLLIDERML_REPLACE_ACCEPTED) or restores the kept generation
        st["state"], st["why"] = "replacement_incomplete", [f"{Path(earlier[-1]).name} accepted, then {status}"]
        return st
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
    tb = cr.get("truth_base")
    if tb and tb["object"] in cr["objects"]:
        cover = [(f, m) for f, m in tb["shards"].items() if _overlaps(f, s0, e0)]
        if not cover:
            probs.append(f"truth_base: no frozen shard covers events {s0}-{e0}")
        for f, m in cover:
            try:
                now = stat_of(f)
            except OSError as x:
                probs.append(f"truth_base {Path(f).name}: {x}")
                continue
            if any(now[k] != m[k] for k in ("realpath", "size", "mtime_ns", "inode")):
                probs.append(f"truth_base {Path(f).name}: differs from the frozen base")
    return probs


def _window_of(name: str):
    import re
    m = re.search(r"\.events(\d+)-(\d+)\.parquet$", name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _overlaps(path: str, s: int, e: int) -> bool:
    w = _window_of(Path(path).name)
    return bool(w) and w[0] <= e and w[1] >= s


# ------------------------------------------------------------------------------------------------ audit
# Restated from the packaging rules (independently of package_native_parquet.note_inputs): which run files a chunk
# must read for each object.
ID_OBJECTS = {"particles", "tracker_hits", "tracker_simhits", "tracks", "truth_tracks"}     # particle-id remap
COV_FILES = {"tracks": "tracksummary_ambi.root", "truth_tracks": "tracksummary_truth.root"}


def expected_chunk_inputs(rec: dict, cr: dict, unit: int, cache: dict, jobs) -> tuple[dict, list[str]]:
    """{input path: ("producer", stage, run) | ("parent", name, run) | ("carried", obj)} that the chunk must read,
    derived from the launch record (flags, objects, contributors), the accepted upstream outputs and the frozen
    parents and carried shards."""
    fl = cr.get("package_flags") or {}
    g4 = fl.get("sim_input_name") or SIM_INPUT
    runs_dir = Path(cr["version_dir"]) / "runs"
    want, probs = {}, []
    s0, e0 = chunk_window(unit, cr["chunk_size"], cr["release_events"])

    def outputs_of(stage, r, top):
        key = (stage, r)
        if key not in cache:
            cache[key] = unit_state(rec, cr, stage, r, jobs)
        u = cache[key]
        if u["state"] != "accepted":
            probs.append(f"{stage} run {r}: {u['state']}")
            return []
        return [e["path"] for e in u["rec"].get("outputs") or [] if e.get("path", "").split("/", 1)[0] == top]

    for c in contributors(unit, cr["chunk_size"], cr["release_events"], cr["run_size"]):
        r = c["run"]
        rd = runs_dir / str(r)

        def prod(stage, top):
            got = outputs_of(stage, r, top)
            if not got:
                probs.append(f"run {r}: no accepted {stage} output {top}")
            for rel in got:
                want[str(rd / rel)] = ("producer", stage, r)

        def parent(name):
            fz = cr["parents"].get(str(r), {}).get(name)
            if not fz:
                probs.append(f"run {r}: parent {name} not frozen")
                return
            for rel in fz["files"]:
                want[str(rd / name) if rel == "." else str(rd / name / rel)] = ("parent", name, r)
        tb = cr.get("truth_base") or {}
        for obj in cr["objects"]:
            if obj == tb.get("object"):
                # D8: Release 1 rows (frozen base shards, below) completed with the rows the run's native-particle
                # map adds: the native particles table, the ACTS particles.root (perigee of added rows; the packager
                # reads it for every contributing run) and the Geant4 MCParticles
                prod("digitization", "particles")
                prod("digitization", "particles.root")
                parent(g4)
                continue
            if obj == "vertices":
                prod("digitization", "performance_vertexing.root")
                if fl.get("vertex_links"):
                    prod("digitization", "vertex_track_links.csv")
                    prod("digitization", "tracker_simhits")
                    prod("digitization", "particles")
                    parent(g4)
            elif obj == "generator_particles":
                parent("generator_particles")
            elif obj in RECO_TABLE_OBJECTS:
                prod("reco_tables", obj)
            else:
                prod("digitization", obj)
            if fl.get("track_covariance") and obj in COV_FILES:
                prod("digitization", COV_FILES[obj])
            if fl.get("remap_particle_ids") and obj in ID_OBJECTS:
                prod("digitization", "particles")
                parent(g4)
    if "vertices" in cr["objects"] and fl.get("vertex_links"):
        for obj, fs in cr["carried"].items():
            if obj == "particles":
                for f in fs:
                    if _overlaps(f, s0, e0):
                        want[f] = ("carried", obj)
    tb = cr.get("truth_base")
    if tb and tb["object"] in cr["objects"]:
        for f, m in tb["shards"].items():
            if _overlaps(f, s0, e0):
                want["truthbase:" + m["realpath"]] = ("truth_base", f)
    return want, probs


def _parent_entry(cr, realpath: str, runs: list[int]):
    for r in runs:
        for name in cr["parent_names"]:
            fz = cr["parents"].get(str(r), {}).get(name) or {}
            for rel, f in (fz.get("files") or {}).items():
                if f["realpath"] == realpath:
                    return f
    return None


def audit_unit(rec: dict, cr: dict, stage: str, unit: int, jobs, cache: dict) -> list[str]:
    """Exact input binding of an ACCEPTED unit. The set of inputs it must have consumed is derived here (not taken
    from its record); every one must be in its record or chunk manifest, nothing else may be, and each must equal
    the accepted upstream output (sha256), the frozen parent (realpath, size, inode, mtime; the Geant4 file's
    sha256 recorded by digitization and by reco_tables must agree) or the frozen carried shard. Both ends of a
    chunk window are checked."""
    st = unit_state(rec, cr, stage, unit, jobs)
    if st["state"] != "accepted":
        return [f"{stage} {unit}: not accepted ({st['state']})"]
    me = st["rec"]
    runs_dir = Path(cr["version_dir"]) / "runs"

    def up(s_, u):
        key = (s_, u)
        if key not in cache:
            cache[key] = unit_state(rec, cr, s_, u, jobs)
        return cache[key]

    def producer_entry(realpath, r, stages):
        for s_ in stages:
            u = up(s_, r)
            if u["state"] == "accepted":
                e = _output_index(u["rec"], runs_dir / str(r)).get(realpath)
                if e:
                    return e, s_
        return None, None

    probs = []
    if stage in RUN_STAGES:
        rd = runs_dir / str(unit)
        got = {e.get("path"): e for e in me.get("inputs") or [] if e.get("path")}
        if any(e.get("missing") for e in me.get("inputs") or []):
            probs.append(f"{stage} run {unit}: a declared input was missing at record time")
        want = {}
        if stage == "digitization":
            want[SIM_INPUT] = "parent"
        elif stage == "pandora_reco":
            want["sim_with_tracks.root"] = "digitization"
        else:
            u = up("pandora_reco", unit)
            parts = [e["path"] for e in (u.get("rec") or {}).get("outputs") or []
                     if e.get("path", "").startswith("reco_parts/") or e.get("path") == "reco_edm4hep.root"]
            if not parts:
                probs.append(f"reco_tables run {unit}: no accepted Pandora outputs")
            want.update({p_: "pandora_reco" for p_ in parts})
            want[SIM_INPUT] = "parent"
        if set(got) != set(want):
            probs.append(f"{stage} run {unit}: inputs {sorted(set(got) ^ set(want))[:4]} differ from the required set")
        for path, kind in want.items():
            e = got.get(path)
            if e is None:
                continue
            if not e.get("sha256"):
                probs.append(f"{stage} run {unit}: input {path} not hashed")
                continue
            if kind == "parent":
                f = _parent_entry(cr, e.get("realpath"), [unit])
                if f is None or e.get("size") != f["size"]:
                    probs.append(f"{stage} run {unit}: {path} is not the frozen parent")
                other = up("reco_tables" if stage == "digitization" else "digitization", unit)
                for oe in (other.get("rec") or {}).get("inputs") or []:
                    if oe.get("path") == SIM_INPUT and oe.get("sha256") and oe["sha256"] != e["sha256"]:
                        probs.append(f"run {unit}: the Geant4 input hashed differently by digitization and reco_tables")
                continue
            pe, who = producer_entry(e.get("realpath"), unit, [kind])
            if pe is not None and pe.get("path") != path:
                pe = None
            if pe is None:
                probs.append(f"{stage} run {unit}: input {path} has no accepted {kind} producer")
            elif pe.get("sha256") != e["sha256"]:
                probs.append(f"{stage} run {unit}: input {path} differs from the accepted {who} output")
        return probs
    man = me.get("container_record") or {}
    inputs = dict(man.get("inputs") or {})
    tb = cr.get("truth_base") or {}
    base_real = {m["realpath"]: (f, m) for f, m in (tb.get("shards") or {}).items()}
    for k in list(inputs):        # the packager may name a base shard by any path: bind it by its terminal
        r_ = (inputs[k] or {}).get("realpath") or os.path.realpath(k)
        if r_ in base_real:
            inputs["truthbase:" + r_] = inputs.pop(k)
    s0, e0 = chunk_window(unit, cr["chunk_size"], cr["release_events"])
    if (man.get("start_event"), man.get("end_event")) != (s0, e0):
        probs.append(f"chunk {unit}: manifest window {man.get('start_event')}-{man.get('end_event')} != {s0}-{e0}")
    want, wprobs = expected_chunk_inputs(rec, cr, unit, cache, jobs)
    probs += [f"chunk {unit}: {x}" for x in wprobs]
    missing, extra = sorted(set(want) - set(inputs)), sorted(set(inputs) - set(want))
    if missing:
        probs.append(f"chunk {unit}: {len(missing)} required input(s) not in the manifest, e.g. {missing[0]}")
    if extra:
        probs.append(f"chunk {unit}: {len(extra)} manifest input(s) outside the required set, e.g. {extra[0]}")
    carried_index = {f: m for o in cr["carried"].values() for f, m in o.items()}
    for path in sorted(set(want) & set(inputs)):
        meta, kind = inputs[path], want[path]
        real = meta.get("realpath")
        if kind[0] == "truth_base":
            m = tb["shards"][kind[1]]
            if real != m["realpath"] or meta.get("size") != m["size"] or meta.get("inode") != m["inode"] or \
                    meta.get("sha256") != m["sha256"]:
                probs.append(f"chunk {unit}: truth base {Path(kind[1]).name} differs from the frozen base (sha256)")
        elif kind[0] == "carried":
            m = carried_index[path]
            if real != m["realpath"] or meta.get("size") != m["size"] or meta.get("inode") != m["inode"]:
                probs.append(f"chunk {unit}: carried {Path(path).name} differs from the frozen inventory")
        elif kind[0] == "parent":
            f = _parent_entry(cr, real, [kind[2]])
            if f is None or any(meta.get(k) != f[k] for k in ("size", "inode", "mtime_ns")) or \
                    ("sha256" in f and meta.get("sha256") != f["sha256"]):
                probs.append(f"chunk {unit}: parent {path} differs from the frozen parent")
        else:
            pe, who = producer_entry(real, kind[2], [kind[1]])
            base_r = runs_dir / str(kind[2])
            rel = str(Path(path).relative_to(base_r)) if Path(path).is_relative_to(base_r) else None
            if pe is not None and pe.get("path") != rel:
                pe = None                       # the metadata is another output of the producer (e.g. truth_tracks for tracks)
            if pe is None:
                probs.append(f"chunk {unit}: input {path} is not that output of the accepted {kind[1]} run {kind[2]}")
            elif not meta.get("sha256") or pe.get("sha256") != meta["sha256"]:
                probs.append(f"chunk {unit}: input {path} differs from the accepted {who} output")
    return probs


# ------------------------------------------------------------------------------------------------ reconcile
def _job_running(jid: str) -> bool | None:
    try:
        out = subprocess.run(["squeue", "-h", "-j", str(jid).split("_")[0], "-o", "%T"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 and "Invalid job id" not in out.stderr:
        return None
    return bool(out.stdout.strip())


def _stage_processes(version_dir: Path, unit: int) -> list[str]:
    """Processes on this node that still work on this unit of this version (a container child that outlived its
    run_stage wrapper): any command line naming the version directory and the unit."""
    try:
        out = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ["ps failed"]
    pat = (f"--run-list {unit}", f"--chunk-index {unit}", f"--output-subdir {unit}", f"/{unit}.a")
    return [l for l in out.splitlines() if str(version_dir) in l and any(p_ in l for p_ in pat) and "v23_control" not in l]


def reconcile(rec: dict, cname: str, apply: bool, purge_failed: bool) -> int:
    """Recovery of interrupted attempts, serialized with every writer (Codex re-review finding 4): with --apply the
    whole pass runs inside the version's control lock (rp.ControlLock), which run_stage holds while it takes a unit
    lock and the finalizer while it freezes, so no unit lock can be taken, and no temporary file created by a new
    attempt, while locks are reclaimed and files removed. A lock is reclaimed only when its owner job is no longer
    in Slurm (asked again right before the reclaim) or, inside a live job on the owner's node, when the owner pid is
    gone AND no process of that unit is left; the lock is re-read and must be byte-identical to the one judged
    stale, then unlinked (no rename-and-restore). Temporary files are removed only for units that hold no lock at
    action time. An interrupted promotion (journal "started", no lock) is rolled back to the previous complete
    generation. --purge-failed deletes failed attempt directories of every stage (<unit>.a<K>), never one that
    keeps a replaced generation while the unit has no newer accepted attempt; other entries are reported."""
    cr = camp(rec, cname)
    v = Path(cr["version_dir"])
    ctl = None
    if apply:
        ctl = rp.ControlLock(v, "v23_control reconcile", timeout_s=float(os.environ.get("COLLIDERML_RECONCILE_WAIT_S", "60")))
        if not ctl.acquire():
            owner = ctl.path.read_text(errors="replace")[:300] if ctl.path.exists() else "gone"
            print(f"undecided {ctl.path} held ({owner}); another reconcile, a finalizer or a writer holds it, or its "
                  "holder died (remove by hand only after checking the owner)")
            return 1
    try:
        return _reconcile(rec, cr, cname, v, apply, purge_failed)
    finally:
        if ctl is not None:
            ctl.release()


def _reconcile(rec, cr, cname, v, apply, purge_failed) -> int:
    jobs = live_jobs()
    actions, undecided, held, info = [], [], set(), []
    for stage in STAGES:
        for u in units_of(cr, stage):
            d, stem = prov_dir(cr, stage, u)
            lock = d / f"{stem}.lock"
            path, last = rp.latest_record(d, stem)
            if lock.exists():
                text = lock.read_text(errors="replace")
                owner = lock_owner(lock)
                alive = lock_alive(owner, jobs)
                if alive is not False:
                    held.add((stage, u))
                    if alive is None:
                        undecided.append(f"{lock}: owner {owner} cannot be checked from here")
                    continue
                actions.append(("unlock", lock, path, last, text, owner, stage, u))
            elif last and last.get("status") in ("started", "promoting"):
                actions.append(("interrupt", None, path, last, None, None, stage, u))
    for j in sorted(glob.glob(str(v / ".attempts" / "*" / "*.a*" / rp.JOURNAL))):
        ad = Path(j).parent
        stage = ad.parent.name
        if not re.fullmatch(r"\d+\.a\d+", ad.name) or stage not in STAGES:
            undecided.append(f"{j}: journal outside a <unit>.a<K> attempt directory; kept")
            continue
        if (rp.read_journal(ad) or {}).get("state") == "started":
            u, k = (int(x) for x in ad.name.split(".a"))
            if (stage, u) in held:
                continue                    # its writer still runs
            actions.append(("rollback", ad, None, None, None, None, stage, u))
    for tmp in glob.glob(str(v / "parquet" / "*" / "*" / ".*.tmp")):
        w = _window_of(Path(tmp).name.rsplit(".", 2)[0].lstrip("."))
        if w is not None and ("package_parquet", w[0] // cr["chunk_size"]) not in held:
            actions.append(("tmp", Path(tmp), None, None, None, None, "package_parquet", w[0] // cr["chunk_size"]))
    for tmp in glob.glob(str(v / "runs" / "*" / "provenance" / ".*.tmp")):
        u = int(Path(tmp).parts[-3])
        if not any((st_, u) in held for st_ in RUN_STAGES):
            actions.append(("tmp", Path(tmp), None, None, None, None, None, u))
    for tmp in glob.glob(str(v / "provenance" / "*" / ".*.tmp")):
        if not any(st_ == "package_parquet" for st_, _ in held):
            actions.append(("tmp", Path(tmp), None, None, None, None, None, None))
    if (v / rp.FREEZE).exists() and not (v / rp.MARKER).exists():
        undecided.append(f"{v / rp.FREEZE} exists without a finalization marker: a finalizer or release step holds the "
                         f"version or was killed ({(v / rp.FREEZE).read_text(errors='replace').strip()[:160]}); remove it "
                         "by hand only after checking that no such process runs")
    if purge_failed:
        for ad in sorted(glob.glob(str(v / ".attempts" / "*" / "*"))):
            stage = Path(ad).parent.name
            name = Path(ad).name
            if stage not in STAGES or not re.fullmatch(r"\d+\.a\d+", name):
                info.append(f"{ad}: not a <unit>.a<K> attempt directory of a stage (e.g. an earlier packaging "
                            "backup events<A>-<B>.<token>); kept, never purged")
                continue
            u, k = (int(x) for x in name.split(".a"))
            if u not in units_of(cr, stage):
                info.append(f"{ad}: unit {u} is not an expected unit of {stage}; kept")
                continue
            d, stem = prov_dir(cr, stage, u)
            recf = d / f"{stem}_attempt{k}.json"
            r_ = rp.latest_record_at(recf) or {}
            if (d / f"{stem}.lock").exists() or rp.is_accepted(r_):
                continue
            if (rp.read_journal(ad) or {}).get("state") == "started":
                continue                        # rolled back first (this pass or the next)
            if (Path(ad) / "replaced").exists():
                later = [p_ for p_ in rp.attempt_records(d, stem) if int(p_.stem.rsplit("attempt", 1)[1]) > k]
                if not any(rp.is_accepted(rp.latest_record_at(p_)) for p_ in later):
                    undecided.append(f"{ad}: keeps the replaced generation of {stage} {u}; not purged")
                    continue
            actions.append(("purge", Path(ad), recf, r_, None, None, stage, u))
    now = datetime.datetime.utcnow().isoformat() + "Z"
    done = 0
    order = {"unlock": 0, "interrupt": 1, "rollback": 2, "tmp": 3, "purge": 4}
    for kind, p_, recp, last, text, owner, stage, u in sorted(actions, key=lambda x: order[x[0]]):
        print(f"{'do' if apply else 'would'} {kind} {p_ or recp}")
        if not apply:
            continue
        if kind == "unlock":
            jid = owner.get("slurm_job_id")
            if jid and _job_running(jid) is not False and not (owner.get("host") == socket.gethostname()):
                undecided.append(f"{p_}: job {jid} is running (re-queried); kept"); continue
            if owner.get("host") == socket.gethostname():
                left = _stage_processes(v, u)
                if left:
                    undecided.append(f"{p_}: processes of unit {u} still run: {left[:2]}"); continue
            try:
                if p_.read_text(errors="replace") != text:
                    undecided.append(f"{p_}: changed owner during reconcile; kept"); continue
            except FileNotFoundError:
                continue
            cur = rp.latest_record_at(recp) if recp else None
            if recp is not None and cur and cur.get("status") in ("started", "promoting"):
                rp.set_status(recp, "interrupted", reconciled_utc=now,
                              reconciled_reason="owner job ended without a final record"
                              + ("; promotion was in progress: rolled back from its journal"
                                 if cur.get("status") == "promoting" else ""))
            p_.unlink()
        elif kind == "interrupt":
            cur = rp.latest_record_at(recp)
            d, stem = prov_dir(cr, stage, u)
            if cur and cur.get("status") in ("started", "promoting") and not (d / f"{stem}.lock").exists():
                rp.set_status(recp, "interrupted", reconciled_utc=now, reconciled_reason="started record without a lock")
        elif kind == "rollback":
            d, stem = prov_dir(cr, stage, u)
            if (d / f"{stem}.lock").exists():
                undecided.append(f"{p_}: its unit is locked now; not rolled back"); continue
            k = int(p_.name.split(".a")[1])
            recf = d / f"{stem}_attempt{k}.json"
            cur = rp.latest_record_at(recf)
            if rp.is_accepted(cur):
                # killed after the accepted record and before the journal was closed: the generation is complete
                attempt_isolation.complete_journal(p_)
                info.append(f"{p_}: record accepted; journal closed, nothing rolled back")
                continue
            did = attempt_isolation.rollback(p_)
            if cur and cur.get("status") in ("started", "promoting", "complete", "incomplete"):
                rp.set_status(recf, "interrupted", reconciled_utc=now,
                              reconciled_reason=f"interrupted promotion rolled back: {did}")
        elif kind == "tmp":
            if stage is not None and u is not None:          # the unit's lock is re-checked at action time
                d_, stem_ = prov_dir(cr, stage, u)
                if (d_ / f"{stem_}.lock").exists():
                    undecided.append(f"{p_}: its unit is locked now; kept"); continue
            elif p_.parent.name == "provenance" and any((p_.parent / f"{st_}.lock").exists() for st_ in RUN_STAGES):
                undecided.append(f"{p_}: a stage of its run is locked now; kept"); continue
            elif p_.parent.parent.name == "provenance" and list(p_.parent.glob("*.lock")):
                undecided.append(f"{p_}: a chunk is locked now; kept"); continue
            p_.unlink(missing_ok=True)
        elif kind == "purge":
            attempt_isolation.purge(p_)
        done += 1
    for x in info:
        print(f"info {x}")
    for x in undecided:
        print(f"undecided {x}")
    print(f"reconcile {cname}: {len(actions)} action(s){'' if apply else ' (dry run)'}, {done} done, {len(undecided)} undecided")
    return 1 if undecided else 0


# ------------------------------------------------------------------------------------------------ verification units
def _verification_paths(cr: dict, i: int) -> tuple[Path, Path, Path]:
    vd = Path(cr.get("verification_dir") or Path(cr["version_dir"]) / "verification")
    return vd / f"chunk{i}.json", vd / f"chunk{i}.summary.json", vd / f"chunk{i}.log"


def verification_report_problems(rec: dict, cr: dict, i: int, doc: dict, produced_sha: dict) -> list[str]:
    """A chunk report is current and binding only if: all checks pass, among them every mandatory check and the
    verifier's own coverage check; it was written by the frozen verifier for this launch record, prefix, chunk and
    window; its objects and carried objects are exactly the frozen ones; its contracts directory is the frozen
    verifier's and the contract hashes equal the frozen ones; the approved resources are the frozen ones (when the
    campaign has Pandora objects); its summary hash is recorded; the shards it read are exactly the chunk's
    produced shards (with the bytes of the accepted chunk manifest) and the frozen carried shards of the window."""
    probs = []
    meta = doc.get("meta") or {}
    cmd = meta.get("cmd") or {}
    w = list(chunk_window(i, cr["chunk_size"], cr["release_events"]))
    if not doc.get("all_pass"):
        probs.append("checks failed")
    res = {r.get("check"): r for r in doc.get("results") or []}
    mand = {"chunk_view", "inventory", "event_identity", "schema", "links", "coverage"}
    objs = set(cr["objects"])
    if objs & {"tracks", "truth_tracks"}:
        mand |= {"shared_hits", "track_numeric_d6"}
    if "vertices" in objs:
        mand |= {"vertices_content", "vertex_links_vs_source", "vertex_link_event_coverage", "vertices_vs_ntuple",
                 "vertex_truth_links"}
    if objs & RECO_TABLE_OBJECTS:
        mand |= {"released_endpoint", "pfo_track_kinematics"}
    if cr.get("truth_base"):
        mand.add("source_manifest")
    for m in sorted(mand):
        if m not in res:
            probs.append(f"mandatory check {m} absent")
        elif not res[m].get("pass"):
            probs.append(f"mandatory check {m} failed")
    if meta.get("verifier_sha256") != rec["verifier"]["digests"]:
        probs.append("another verifier")
    if meta.get("launch_digest") != rec["digest"]:
        probs.append("not bound to this launch record")
    if (cmd.get("chunk"), cmd.get("prefix"), meta.get("window")) != (i, cr["prefix"], w):
        probs.append(f"report is for {cmd.get('prefix')} chunk {cmd.get('chunk')} window {meta.get('window')}")
    if sorted(meta.get("objects") or []) != sorted(list(cr["objects"]) + list(cr["carried_objects"])):
        probs.append(f"objects {meta.get('objects')} are not the frozen set")
    if sorted(meta.get("carried") or []) != sorted(cr["carried_objects"]):
        probs.append(f"carried {meta.get('carried')} are not the frozen carried objects")
    cdir = (Path(rec["verifier"]["dir"]) / "contracts").resolve()
    if meta.get("contracts_dir") != str(cdir):
        probs.append(f"contracts dir {meta.get('contracts_dir')} is not the frozen verifier's {cdir}")
    frozen_c = {k.split("/", 1)[1]: h for k, h in rec["verifier"]["digests"].items() if k.startswith("contracts/")}
    if ((meta.get("contracts_sha256") or {}).get("files") or {}) != frozen_c:
        probs.append("contract hashes differ from the frozen verifier's")
    if objs & RECO_TABLE_OBJECTS:
        if meta.get("approved_resources") != rec.get("approved_resources") or not meta.get("resources_sha256"):
            probs.append("approved Pandora resources absent or not the frozen ones")
    if not meta.get("summary_sha256"):
        probs.append("summary hash not recorded")
    if cr.get("truth_base") and not meta.get("source_manifests"):
        probs.append("source chunk manifest not bound")
    shards = {os.path.realpath(f): h for f, h in (meta.get("shards") or {}).items()}
    s0, e0 = w
    want = {os.path.realpath(f): h for f, h in produced_sha.items()}
    for o in cr["carried_objects"]:
        for f, m in (cr["carried"].get(o) or {}).items():
            if _overlaps(f, s0, e0):
                want.setdefault(m["realpath"], None)
    if set(shards) != set(want):
        probs.append(f"shards read are not exactly the chunk's produced and carried shards "
                     f"({len(set(shards) - set(want))} other, {len(set(want) - set(shards))} not read)")
    for f, h in want.items():
        if h is not None and shards.get(f) not in (None, h):
            probs.append(f"{Path(f).name}: read bytes differ from the accepted chunk manifest")
    return probs


def verification_current(rec: dict, cr: dict, i: int, jobs=None) -> tuple[bool, list[str]]:
    """(True, []) if chunk i has a current passing report bound to this launch record and to the accepted chunk
    manifest's bytes."""
    f, sf, _ = _verification_paths(cr, i)
    if not f.exists():
        return False, ["no report"]
    try:
        doc = json.loads(f.read_text())
    except (OSError, ValueError) as e:
        return False, [f"unreadable report: {e}"]
    st = unit_state(rec, cr, "package_parquet", i, jobs)
    if st["state"] != "accepted":
        return False, [f"chunk {i} packaging {st['state']}"]
    produced = {k: m["sha256"] for k, m in ((st["rec"].get("container_record") or {}).get("outputs") or {}).items()}
    probs = verification_report_problems(rec, cr, i, doc, produced)
    if sf.exists() and (doc.get("meta") or {}).get("summary_sha256") != sha256_file(sf):
        probs.append("summary changed since the report")
    return not probs, probs


def unit_dispatch_problems(rec: dict, cr: dict, stage: str, u: int, jobs, cache: dict) -> list[str]:
    """Why a unit may not be dispatched now; the same rule for plan, dispatch and the worker's per-node check
    (Codex re-review finding 6): a unit to run is missing, failed or interrupted, has fewer than two failed attempts,
    every upstream input is accepted, in place and bound by the exact audit."""
    probs = []
    if stage == VERIFY:
        st = unit_state(rec, cr, "package_parquet", u, jobs)
        if st["state"] != "accepted":
            return [f"verify {u}: chunk packaging is {st['state']}"]
        probs += audit_unit(rec, cr, "package_parquet", u, jobs, cache)
        ok, _ = verification_current(rec, cr, u, jobs)
        if ok:
            probs.append(f"verify {u}: a current passing report exists")
        return probs
    s = unit_state(rec, cr, stage, u, jobs)
    if s["state"] not in ("missing", "failed", "interrupted"):
        probs.append(f"{stage} {u}: {s['state']} {s.get('why', '')}: not a unit to dispatch")
    if s.get("n_failed", 0) >= 2:
        probs.append(f"{stage} {u}: {s['n_failed']} failed attempts: the stage stops for a decision")
    probs += upstream_problems(rec, cr, stage, u, jobs, cache)
    upstage = {"pandora_reco": "digitization", "reco_tables": "pandora_reco"}.get(stage)
    ups = [(upstage, u)] if upstage else []
    if stage == "package_parquet":
        need_rt = bool(RECO_TABLE_OBJECTS & set(cr["objects"]))
        ups = [(s_, c["run"]) for c in contributors(u, cr["chunk_size"], cr["release_events"], cr["run_size"])
               for s_ in (["digitization"] + (["reco_tables"] if need_rt else []))]
    for s_, r in ups:
        probs += audit_unit(rec, cr, s_, r, jobs, cache)
    return probs


def parse_time_limit(t: str) -> float:
    """HH:MM:SS -> hours; refuses a zero or malformed limit (Slurm reads 0 as no limit)."""
    m = re.fullmatch(r"(\d+):([0-5]\d):([0-5]\d)", t or "")
    if not m:
        raise ValueError(f"time limit {t!r} is not HH:MM:SS")
    h = int(m.group(1)) + int(m.group(2)) / 60 + int(m.group(3)) / 3600
    if h <= 0:
        raise ValueError("a zero time limit is refused (Slurm treats it as unlimited)")
    return h


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


def _ledger_entries(rec) -> list[dict]:
    ledger = Path(rec["budget"]["ledger"])
    return [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()] if ledger.exists() else []


def budget(rec: dict, add_node_hours: float = 0.0, verbose: bool = True, entries=None) -> tuple[bool, dict]:
    """Consumed + forecast + the block about to be dispatched <= cap. Consumed: every ledger reservation at its full
    node x time limit until its job has finished (then elapsed x nodes from sacct); a reservation whose submission
    never recorded a job id counts at its full limit. Forecast: every expected unit not accepted, at the frozen
    per-unit rate x (1 + retry allowance), plus verification of every chunk not yet verified. Double counting of
    units of pending reservations is accepted (conservative)."""
    b = rec["budget"]
    entries = _ledger_entries(rec) if entries is None else entries
    jobs_of = {e["reservation"]: e["job_id"] for e in entries if e.get("kind") == "submitted"}
    acct = _sacct(sorted(set(jobs_of.values())))
    consumed = 0.0
    for e in entries:
        if e.get("kind") != "reservation":
            continue
        a_ = acct.get(str(jobs_of.get(e["reservation"], "")).split("_")[0])
        if a_ and a_["state"].split()[0] in TERMINAL_STATES:
            # a finished job: what it used, never less than nothing; any other state (pending, running, configuring,
            # completing, requeued, suspended, ...) keeps the reservation at its full limit
            consumed += max(0, a_["elapsed_s"]) * a_["nodes"] / 3600
        else:
            consumed += e["nodes"] * e["time_limit_h"]
    jobs = live_jobs()
    forecast, detail = 0.0, {}
    for cname, cr in rec["campaigns"].items():
        rates = b["rates_node_hours_per_unit"].get(cname, {})
        for stage in STAGES + ["verify"]:
            if stage == VERIFY:
                # only a CURRENT passing report bound to this launch record retires a chunk's verification
                n_left = sum(1 for u in units_of(cr, "package_parquet")
                             if verification_current(rec, cr, u, jobs)[0] is not True)
            else:
                n_left = sum(1 for u in units_of(cr, stage) if unit_state(rec, cr, stage, u, jobs)["state"] != "accepted")
            nh = n_left * float(rates[stage]) * (1 + b["retry_allowance"])
            detail[f"{cname}/{stage}"] = {"units_left": n_left, "node_hours": round(nh, 2)}
            forecast += nh
    total = consumed + forecast + add_node_hours
    ok = total <= b["cap_node_hours"]
    rep = {"consumed": round(consumed, 2), "forecast": round(forecast, 2), "this_block": add_node_hours,
           "total": round(total, 2), "cap": b["cap_node_hours"], "ok": ok, "detail": detail}
    if verbose:
        print(json.dumps(rep, indent=1))
    return ok, rep


class _LedgerLock:
    """Exclusive lock on the ledger (flock), held from the budget check to the reservation and the submission."""
    def __init__(self, rec):
        self.path = Path(rec["budget"]["ledger"]).with_suffix(".lock")

    def __enter__(self):
        import fcntl
        self.f = open(self.path, "a")
        fcntl.flock(self.f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        import fcntl
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def _ledger_append(rec, entry: dict) -> None:
    with open(rec["budget"]["ledger"], "a") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()
        os.fsync(f.fileno())


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
    if tgt.parent.exists() and list(tgt.parent.glob(f".{tgt.name}.publishing.*")):
        probs.append(f"an interrupted publication of {tgt.name} exists under {tgt.parent} (rollback first)")
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


def _manifest_digest(man: dict) -> str:
    return hashlib.sha256(canonical({k: v for k, v in man.items() if k != "digest"})).hexdigest()


def _tree_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() or p.is_symlink())


def _meta_files(v: Path, finalized: bool) -> tuple[dict, list[str]]:
    """The public files next to parquet/: the dataset card README.md, configs/, and (finalized) the marker."""
    out, probs = {}, []
    for name in PUBLIC_META:
        q = v / name
        if name == rp.MARKER and not finalized:
            continue
        if not q.exists():
            probs.append(f"public file {name} missing in {v} (dataset card, configs and marker are published)")
            continue
        for f in ([q] if q.is_file() else sorted(x for x in q.rglob("*") if x.is_file())):
            out[str(f.relative_to(v))] = sha256_file(f)
    return out, probs


def release_problems(rec, cname, verification_dir, decision_files, jobs_n, finalized: bool = True) -> tuple[list, dict]:
    """Every release condition (inventory, and the finalizer before it writes its marker). Returns (problems, data
    for the manifest)."""
    cr = camp(rec, cname)
    v = Path(cr["version_dir"])
    probs = []
    if finalized and (not (v / rp.MARKER).exists() or not (v / rp.FREEZE).exists()):
        probs.append("version not finalized and frozen")
    if not finalized and not (v / rp.FREEZE).exists():
        probs.append("version not frozen (the finalizer checks under .freeze)")
    frozen_vd = Path(cr.get("verification_dir") or v / "verification").resolve()
    if verification_dir and Path(verification_dir).resolve() != frozen_vd:
        probs.append(f"verification dir {verification_dir} is not the frozen {frozen_vd}")
    jobs = live_jobs()
    cache = {}
    # the whole producer chain under freeze (Codex re-review finding 2): every run-stage unit accepted, bound to its
    # inputs and its outputs still in place, then every chunk
    for stage in RUN_STAGES:
        for u in units_of(cr, stage):
            st = unit_state(rec, cr, stage, u, jobs)
            if st["state"] != "accepted":
                probs.append(f"{stage} {u}: {st['state']}")
                continue
            probs += [f"audit: {p_}" for p_ in audit_unit(rec, cr, stage, u, jobs, cache)]
            for e in st["rec"].get("outputs") or []:
                why = _current(e, v / "runs" / str(u))
                if why:
                    probs.append(f"{stage} {u}: {why}")
    chunks = units_of(cr, "package_parquet")
    produced = {}
    for i in chunks:
        st = unit_state(rec, cr, "package_parquet", i, jobs)
        if st["state"] != "accepted":
            probs.append(f"chunk {i}: {st['state']}")
            continue
        for f, m in ((st["rec"].get("container_record") or {}).get("outputs") or {}).items():
            produced[f] = {"chunk": i, "size": m["size"], "sha256_manifest": m["sha256"]}
        probs += [f"audit: {p_}" for p_ in audit_unit(rec, cr, "package_parquet", i, jobs, cache)]
    carried = {f: {"object": o, **m} for o, fs in cr["carried"].items() for f, m in fs.items()}
    present = [str(p_) for p_ in _tree_files(v / "parquet")]
    unknown = [p_ for p_ in present if p_ not in produced and p_ not in carried]
    if unknown:
        probs.append(f"{len(unknown)} file(s) under parquet/ neither produced nor carried, e.g. {unknown[0]}")
    absent = [f for f in list(produced) + list(carried) if f not in set(present)]
    if absent:
        probs.append(f"{len(absent)} produced or carried file(s) absent, e.g. {absent[0]}")
    files = sorted(set(produced) | set(carried))

    def h(f):
        try:
            return f, sha256_file(f)
        except OSError as e:
            return f, f"error: {e}"
    with ThreadPoolExecutor(jobs_n) as ex:
        sha = dict(ex.map(h, files))
    for f, m in produced.items():
        if sha.get(f) != m["sha256_manifest"]:
            probs.append(f"{Path(f).name}: sha256 {sha.get(f)} != chunk manifest")
    for f, m in carried.items():
        try:
            now = stat_of(f)
        except OSError as e:
            probs.append(f"carried {Path(f).name}: {e}"); continue
        if any(now[k] != m[k] for k in ("realpath", "size", "mtime_ns", "inode")):
            probs.append(f"carried {Path(f).name}: changed since freezing")
    R = cr["release_events"]
    for obj in cr["objects"] + cr["carried_objects"]:
        d = obj_dir(cr, obj)
        wins = sorted(_window_of(Path(p_).name) or (-1, -1) for p_ in (sorted(map(str, d.iterdir())) if d.is_dir() else []))
        pos = 0
        for a_, b_ in wins:
            if a_ != pos:
                probs.append(f"{obj}: windows do not tile the release at event {pos} (next {a_}-{b_})"); break
            pos = b_ + 1
        if pos != R and not (obj in cr["carried_objects"] and pos > R and wins and wins[-1][0] < R):
            probs.append(f"{obj}: windows end at {pos - 1}, release is 0-{R - 1}")
    ver, vd = {}, frozen_vd
    summaries = {}
    for i in chunks:
        f = vd / f"chunk{i}.json"
        if not f.exists():
            probs.append(f"verification of chunk {i} missing"); continue
        try:
            doc = json.loads(f.read_text())
        except ValueError as e:
            probs.append(f"verification of chunk {i} unreadable: {e}"); continue
        mine = {f_: m["sha256_manifest"] for f_, m in produced.items() if m["chunk"] == i}
        probs += [f"verification of chunk {i}: {x}" for x in verification_report_problems(rec, cr, i, doc, mine)]
        for path, hsh in ((doc.get("meta") or {}).get("shards") or {}).items():
            if path in sha and sha[path] != hsh:
                probs.append(f"verification of chunk {i}: {Path(path).name} has other bytes than released")
        ver[str(i)] = sha256_file(f)
        sf = vd / f"chunk{i}.summary.json"
        summaries[str(sf)] = sha256_file(sf) if sf.exists() else None
        if summaries[str(sf)] is None or (doc.get("meta") or {}).get("summary_sha256") != summaries[str(sf)]:
            probs.append(f"verification of chunk {i}: summary {sf.name} missing or not the one the report wrote")
    red = vd / "release_reductions.json"
    if not red.exists():
        probs.append("release-wide reductions missing")
    else:
        d = json.loads(red.read_text())
        if not d.get("all_pass"):
            probs.append("release-wide reductions failed")
        if (d.get("meta") or {}).get("summaries_sha256") != summaries:
            probs.append("release-wide reductions were computed from other summaries than these chunks'")
        if (d.get("meta") or {}).get("verifier_sha256") != rec["verifier"]["digests"]:
            probs.append("release-wide reductions ran another verifier")
        ver["release_reductions"] = sha256_file(red)
    if rec.get("require_tracking_bound", True):
        tb = vd / "tracking_bound.json"
        tprobs = tracking_bound_problems(rec, cname, tb)
        probs += [f"P5: {x}" for x in tprobs]
        if not tprobs:
            ver["tracking_bound"] = sha256_file(tb)
    dec, dprobs = decision_problems(rec, decision_files)
    probs += dprobs
    meta_files, mprobs = _meta_files(v, finalized)
    probs += mprobs
    return probs, {"files": files, "sha": sha, "produced": produced, "ver": ver, "dec": dec, "meta_files": meta_files}


def tracking_bound_problems(rec: dict, cname: str, tb: Path) -> list[str]:
    """P5 binds only as the verifier's production_tracking_bound result for THIS campaign, the frozen run list and
    this launch record, passing, with at least one conclusive bin (Codex re-review finding 1/3)."""
    if not tb.exists():
        return ["tracking_bound.json missing"]
    try:
        doc = json.loads(tb.read_text())
    except ValueError as e:
        return [f"unreadable: {e}"]
    probs = []
    meta = doc.get("meta") or {}
    res = [r for r in doc.get("results") or [] if r.get("check") == "production_tracking_bound"]
    if len(res) != 1 or len(doc.get("results") or []) != 1:
        probs.append("not a production_tracking_bound result")
    if not doc.get("all_pass") or not (res and res[0].get("pass")):
        probs.append("failed")
    if meta.get("campaign") != cname:
        probs.append(f"for campaign {meta.get('campaign')}")
    want = (rec.get("frozen_tracking_sample") or {}).get(cname)
    if not want or meta.get("runs") != want:
        probs.append(f"run list {str(meta.get('runs'))[:60]} is not the frozen sample {str(want)[:60]}")
    if meta.get("launch_digest") != rec["digest"]:
        probs.append("not bound to this launch record")
    if meta.get("verifier_sha256") != rec["verifier"]["digests"]:
        probs.append("another verifier")
    n = ((res[0].get("info") or {}).get("n_conclusive_p5_bins") if res else None)
    if not isinstance(n, int) or n <= 0:
        probs.append(f"{n} conclusive P5 bins")
    return probs


def decision_problems(rec: dict, decision_files) -> tuple[dict, list[str]]:
    """Each decision record given must be one the launch record names (approved_decision_records, from the lead's
    authorization) with the same sha256; at least one."""
    approved = rec.get("approved_decision_records") or {}
    probs, dec = [], {}
    if not decision_files:
        probs.append("no decision record bound (--decision)")
    if not approved:
        probs.append("the launch record names no approved decision records (freeze --authorization)")
    norm = {os.path.realpath(k): h for k, h in approved.items()}
    for f in decision_files:
        h = sha256_file(f) if Path(f).exists() else None
        dec[str(f)] = h
        if norm.get(os.path.realpath(f)) is None:
            probs.append(f"decision {f} is not an approved decision record of the launch record")
        elif norm[os.path.realpath(f)] != h:
            probs.append(f"decision {f} changed since its approval (sha256)")
    return dec, probs


def release_inventory(rec, cname, out, verification_dir, decision_files, jobs_n) -> int:
    """RELEASE_MANIFEST of one campaign, written only if every condition of release_problems holds: finalized and
    frozen; every run-stage unit and chunk accepted and bound by the exact audit, outputs in place; every file under
    parquet/ produced (sha256 == accepted chunk manifest) or frozen carried, nothing else, windows tiling the
    release; each chunk's verification current and binding (verification_report_problems); the reductions bound to
    these summaries; P5 bound; approved decision records; the public card, configs and marker hashed."""
    cr = camp(rec, cname)
    v = Path(cr["version_dir"])
    probs, D = release_problems(rec, cname, verification_dir, decision_files, jobs_n, finalized=True)
    if probs:
        print("\n".join(probs[:200]))
        print(f"inventory: {len(probs)} problem(s); no manifest written")
        return 1
    files, sha, produced = D["files"], D["sha"], D["produced"]
    man = {"schema": "colliderml.v23.release/1", "campaign": cname, "launch_record_digest": rec["digest"],
           "approved_sha": rec["approved_sha"], "written_utc": datetime.datetime.utcnow().isoformat() + "Z",
           "release_events": cr["release_events"], "objects": cr["objects"] + cr["carried_objects"],
           "files": {str(Path(f).relative_to(v / "parquet")): {"sha256": sha[f], "size": os.path.getsize(f),
                                                               "kind": "produced" if f in produced else "carried",
                                                               "realpath": os.path.realpath(f)} for f in files},
           "meta_files": D["meta_files"],
           "verification_sha256": D["ver"], "decisions_sha256": D["dec"],
           "run_products_retained_privately": str(v / "runs")}
    man["digest"] = _manifest_digest(man)
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps(man, indent=1, sort_keys=True) + "\n")
    print(f"inventory: {len(files)} files, manifest {out} ({man['digest']})")
    return 0


def _check_tree_against(man: dict, root: Path, jobs_n: int) -> list[str]:
    probs = []
    present = {str(p_.relative_to(root)) for p_ in _tree_files(root)}
    want = set(man["files"])
    if present != want:
        probs.append(f"file set differs from the manifest: {len(present - want)} extra, {len(want - present)} missing")

    def h(r):
        try:
            return r, sha256_file(root / r)
        except OSError as e:
            return r, f"error: {e}"
    with ThreadPoolExecutor(jobs_n) as ex:
        got = dict(ex.map(h, sorted(want & present)))
    probs += [f"{r}: sha256 {s_[:16]} differs from the manifest" for r, s_ in got.items() if s_ != man["files"][r]["sha256"]]
    return probs


def _check_meta_against(man: dict, root: Path) -> list[str]:
    probs = []
    for rel, h in (man.get("meta_files") or {}).items():
        q = root / rel
        if not q.is_file() or sha256_file(q) != h:
            probs.append(f"public file {rel} missing or not the manifest's bytes")
    return probs


def _baseline_problems(rec: dict, cr: dict) -> list[str]:
    """Parents, carried shards and the truth base still as frozen (stat), after a rollback."""
    probs = []
    for r in cr.get("parents") or {}:
        for name in cr.get("parent_names") or []:
            probs += _parent_current(cr, int(r), name)
    for o, fs in (cr.get("carried") or {}).items():
        for f, m in fs.items():
            try:
                now = stat_of(f)
            except OSError as e:
                probs.append(f"carried {Path(f).name}: {e}"); continue
            if any(now[k] != m[k] for k in ("realpath", "size", "mtime_ns", "inode")):
                probs.append(f"carried {Path(f).name}: changed since freezing")
    for f, m in ((cr.get("truth_base") or {}).get("shards") or {}).items():
        try:
            now = stat_of(f)
        except OSError as e:
            probs.append(f"truth base {Path(f).name}: {e}"); continue
        if any(now[k] != m[k] for k in ("realpath", "size", "mtime_ns", "inode")):
            probs.append(f"truth base {Path(f).name}: changed since freezing")
    return probs


def release_publish(rec, cname, manifest, published_root, min_free_inodes, jobs_n) -> int:
    """Publication as one transaction: the manifest bytes are read ONCE, checked (digest, campaign, launch record)
    and those bytes are published; the staging tree and the public files are rehashed against it; produced shards
    become read-only; an intent file next to the target is written before the hidden assembly directory exists, so
    every interruption point is visible to rollback; the card, configs and marker are copied in, parquet/ is moved
    in, and one rename of the assembly directory to the target is the publication point. A failure before it
    moves everything back."""
    cr = camp(rec, cname)
    v = Path(cr["version_dir"])
    raw = Path(manifest).read_bytes()
    man = json.loads(raw)
    probs = release_preflight(rec, cname, published_root, min_free_inodes)
    if man.get("digest") != _manifest_digest(man):
        probs.append("manifest digest does not match its content")
    if man.get("launch_record_digest") != rec["digest"] or man.get("campaign") != cname:
        probs.append(f"manifest is for {man.get('campaign')} / launch {str(man.get('launch_record_digest'))[:12]}")
    if not (v / rp.FREEZE).exists():
        probs.append(f"{v / rp.FREEZE} missing: the version must stay frozen after finalization")
    if not probs:
        probs += _check_tree_against(man, v / "parquet", jobs_n)
        probs += _check_meta_against(man, v)
    if probs:
        print("\n".join(probs[:100]))
        return 1
    import shutil
    tgt = _published_dir(cr, published_root)
    txid = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S") + f".{os.getpid()}"
    tmp = tgt.parent / f".{tgt.name}.publishing.{txid}"
    intent = tgt.parent / f".{tgt.name}.publishing.{txid}.intent"
    for rel, m in man["files"].items():
        p_ = v / "parquet" / rel
        if m["kind"] == "produced" and not p_.is_symlink():
            os.chmod(p_, 0o444)
    tgt.parent.mkdir(parents=True, exist_ok=True)
    txrec = {"from": str(v), "manifest_digest": man["digest"], "txid": txid, "host": socket.gethostname(),
             "utc": datetime.datetime.utcnow().isoformat() + "Z"}
    intent.write_text(json.dumps(txrec))
    os.mkdir(tmp)
    moved = False
    try:
        (tmp / "RELEASE_MANIFEST.json").write_bytes(raw)
        (tmp / ".published").write_text(json.dumps(txrec))
        for rel in man.get("meta_files") or {}:
            dst = tmp / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(v / rel, dst)
        os.rename(v / "parquet", tmp / "parquet"); moved = True
        os.rename(tmp, tgt)                                 # publication point
    except BaseException:
        if moved and (tmp / "parquet").exists():
            os.rename(tmp / "parquet", v / "parquet")
        if tmp.exists():
            shutil.rmtree(tmp)                              # only the copies and the transaction files are left
        intent.unlink(missing_ok=True)
        raise
    intent.unlink(missing_ok=True)
    print(f"published {tgt} (transaction {txid})")
    return 0


def release_verify_bytes(rec, cname, published_root, remote_listing, jobs_n, no_portal=False) -> int:
    cr = camp(rec, cname)
    tgt = _published_dir(cr, published_root)
    man = json.loads((tgt / "RELEASE_MANIFEST.json").read_text())
    probs = []
    if man.get("digest") != _manifest_digest(man) or man.get("launch_record_digest") != rec["digest"]:
        probs.append("published manifest is not the one of this launch record")
    want_top = {".published", "RELEASE_MANIFEST.json", "parquet"} | {r.split("/", 1)[0] for r in man.get("meta_files") or {}}
    top = sorted(p_.name for p_ in tgt.iterdir())
    if set(top) != want_top:
        probs.append(f"published version dir holds {top}, expected {sorted(want_top)}")
    probs += _check_tree_against(man, tgt / "parquet", jobs_n)
    probs += _check_meta_against(man, tgt)
    if remote_listing:
        remote = json.loads(Path(remote_listing).read_text())
        allowed = {**{r: m["sha256"] for r, m in man["files"].items()}, **(man.get("meta_files") or {})}
        probs += [f"portal {r}: {remote.get(r)} != {m['sha256']}" for r, m in man["files"].items() if remote.get(r) != m["sha256"]]
        probs += [f"portal {r}: {remote.get(r)} != {h}" for r, h in (man.get("meta_files") or {}).items()
                  if r in remote and remote[r] != h]
        extra = sorted(set(remote) - set(allowed))
        if extra:
            probs.append(f"portal lists {len(extra)} entr(y/ies) outside the manifest, e.g. {extra[0]}")
    elif not no_portal:
        probs.append("no portal listing given (--remote-listing); pass --no-portal only for a local rehearsal")
    print("\n".join(probs[:100]) or f"verify-bytes: {len(man['files'])} files equal to the manifest")
    return 1 if probs else 0


def release_rollback(rec, cname, published_root, jobs_n: int = 16) -> int:
    """Undo a publication (visible target) or an interrupted one (hidden assembly directory, with or without its
    .published record, found through the intent file written before it): parquet/ goes back to the staging version
    dir, the published copies and skeleton are removed; then the restored tree is rehashed against the manifest and
    the parents, carried shards and truth base are checked against the frozen baseline."""
    import shutil
    cr = camp(rec, cname)
    v = Path(cr["version_dir"])
    tgt = _published_dir(cr, published_root)
    if not tgt.parent.exists():
        print(f"nothing published from {v} under {tgt.parent}")
        return 1
    intents = {}
    for f in sorted(tgt.parent.glob(f".{tgt.name}.publishing.*.intent")):
        try:
            intents[str(f)[: -len(".intent")]] = (f, json.loads(f.read_text()))
        except ValueError:
            intents[str(f)[: -len(".intent")]] = (f, {})
    cands = [tgt] + sorted(p_ for p_ in tgt.parent.glob(f".{tgt.name}.publishing.*") if p_.is_dir())
    done, mans, probs = 0, [], []
    for d in cands:
        if not d.exists():
            continue
        pub = d / ".published"
        src = None
        if pub.exists():
            try:
                src = json.loads(pub.read_text()).get("from")
            except ValueError:
                src = None
        if src is None and str(d) in intents:
            src = intents[str(d)][1].get("from")
        if src != str(v):
            continue
        if (d / "RELEASE_MANIFEST.json").exists():
            try:
                mans.append(json.loads((d / "RELEASE_MANIFEST.json").read_text()))
            except ValueError:
                pass
        if (d / "parquet").exists():
            if (v / "parquet").exists():
                print(f"{v / 'parquet'} exists; refusing to roll back {d}")
                return 1
            os.rename(d / "parquet", v / "parquet")
        shutil.rmtree(d)                     # the manifest copy, .published and the public-file copies
        if str(d) in intents:
            intents[str(d)][0].unlink(missing_ok=True)
        print(f"rolled back {d} -> {v / 'parquet'}")
        done += 1
    for key, (f, doc) in intents.items():   # an intent whose directory was never created
        if not Path(key).exists() and doc.get("from") == str(v):
            f.unlink(missing_ok=True)
            print(f"removed the intent {f.name} of a publication that never created its directory")
            done += 1
    if not done:
        print(f"nothing published from {v} under {tgt.parent}")
        return 1
    if not (v / "parquet").is_dir():
        probs.append(f"{v / 'parquet'} missing after the rollback")
    elif mans:
        probs += _check_tree_against(mans[0], v / "parquet", jobs_n)
    probs += _baseline_problems(rec, cr)
    if probs:
        print("\n".join(probs[:100]))
        print("rollback done; the restored staging or the frozen baseline differs (see above)")
        return 1
    return 0


# ------------------------------------------------------------------------------------------------ worker side
def unit_check(rec: dict, config: str, units: str) -> int:
    """Run by run_runs_interactive.sh on each execution node before it writes anything: the live identity on this
    node (code, env, configs, images), the stage config and archive as frozen, and for each unit of the wave the
    same rule dispatch applied (state, failure count, upstream accepted, in place and bound)."""
    raw = yaml.safe_load(Path(config).read_text())
    cname, stage = raw.get("campaign"), raw.get("stage")
    probs = check(rec, images=True)
    cr = rec["campaigns"].get(cname)
    if cr is None or stage not in cr["stages"]:
        print(f"unit-check: {cname}/{stage} is not in the launch record")
        return 1
    fz = cr["stages"][stage]
    if sha256_file(config) != fz["config_sha256"]:
        probs.append(f"{config}: not the frozen {cname}/{stage} config")
    if driver_tarball(raw, stage) != fz.get("tarball"):
        probs.append(f"archive {driver_tarball(raw, stage)} is not the frozen {fz.get('tarball')}")
    try:
        us = [int(x) for x in units.split(",") if x]
    except ValueError:
        us, probs = [], probs + [f"invalid unit list {units!r}"]
    expected = set(units_of(cr, stage))
    jobs, cache = live_jobs(), {}
    for u in us:
        if u not in expected:
            probs.append(f"{stage} {u}: not an expected unit")
            continue
        probs += unit_dispatch_problems(rec, cr, stage, u, jobs, cache)
    print("\n".join(probs) or f"unit-check {cname}/{stage} {units}: ok on {socket.gethostname()}")
    return 1 if probs else 0


def verify_chunk_cmd(rec: dict, record_path: str, cname: str, i: int, tmpdir: str | None) -> int:
    """One chunk verification with the frozen verifier: arguments derived from the launch record (objects, carried,
    contracts, truth base, the accepted chunk record as source manifest). An earlier report of the chunk is moved to
    <verification>/attempts/ first (never overwritten)."""
    cr = camp(rec, cname)
    probs = check(rec, images=False)
    jobs = live_jobs()
    probs += unit_dispatch_problems(rec, cr, VERIFY, i, jobs, {})
    if probs:
        print("\n".join(probs[:50]))
        print(f"verify-chunk {cname} {i}: refused")
        return 1
    st = unit_state(rec, cr, "package_parquet", i, jobs)
    out, summary, log = _verification_paths(cr, i)
    out.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S") + f".{os.getpid()}"
    for f in (out, summary, log):
        if f.exists():
            (out.parent / "attempts").mkdir(exist_ok=True)
            os.rename(f, out.parent / "attempts" / f"{f.name}.{stamp}")
    v = Path(cr["version_dir"])
    vdir = Path(rec["verifier"]["dir"])
    cmd = [sys.executable, str(vdir / "chunk_verify.py"), "chunk", "--release-root", str(v), "--prefix", cr["prefix"],
           "--runs-root", str(v / "runs"), "--objects", ",".join(list(cr["objects"]) + list(cr["carried_objects"])),
           "--out", str(out), "--chunk", str(i), "--chunk-size", str(cr["chunk_size"]),
           "--release-events", str(cr["release_events"]), "--run-size", str(cr["run_size"]),
           "--particles-contract", cr["verify"]["particles_contract"], "--contracts", str(vdir / "contracts"),
           "--summary", str(summary), "--carried", ",".join(cr["carried_objects"]), "--launch-digest", rec["digest"],
           "--launch-record", str(Path(record_path).resolve()), "--campaign", cname, "--manifest", st["record"]]
    if cr.get("truth_base"):
        cmd += ["--truth-base", cr["truth_base"]["dir"]]
    if tmpdir:
        cmd += ["--tmpdir", tmpdir]
    with open(log, "w") as fh:
        fh.write(" ".join(cmd) + "\n")
        fh.flush()
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    ok, why = verification_current(rec, cr, i, jobs)
    print(f"verify-chunk {cname} {i}: verifier rc {rc}, report {'current and passing' if ok else 'NOT accepted: ' + '; '.join(why[:5])}")
    return 0 if (rc == 0 and ok) else 1


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("freeze"); p.add_argument("spec"); p.add_argument("--out", required=True)
    p.add_argument("--authorization", help="the lead's authorization JSON (outside the repo)")
    p = sub.add_parser("check"); p.add_argument("record"); p.add_argument("--no-images", action="store_true")
    p = sub.add_parser("expect"); p.add_argument("record"); p.add_argument("--campaign")
    for name in ("status", "plan", "audit"):
        p = sub.add_parser(name); p.add_argument("record"); p.add_argument("--campaign", required=True)
        p.add_argument("--stage", required=True, choices=STAGES + [VERIFY])
        p.add_argument("--first", type=int); p.add_argument("--last", type=int)
        p.add_argument("--units", help="comma-separated units (default: all expected)")
        if name == "plan":
            p.add_argument("--out", required=True)
    p = sub.add_parser("reconcile"); p.add_argument("record"); p.add_argument("--campaign", required=True)
    p.add_argument("--apply", action="store_true"); p.add_argument("--purge-failed", action="store_true")
    p = sub.add_parser("budget"); p.add_argument("record"); p.add_argument("--add-node-hours", type=float, default=0.0)
    p = sub.add_parser("dispatch"); p.add_argument("record"); p.add_argument("--campaign", required=True)
    p.add_argument("--stage", required=True, choices=STAGES + [VERIFY]); p.add_argument("--units", required=True)
    p.add_argument("--nodes", type=int, required=True); p.add_argument("--time", required=True)
    p.add_argument("--per-wave", type=int, required=True); p.add_argument("--qos", default="regular")
    p.add_argument("--account", default="m4958"); p.add_argument("--submit", action="store_true")
    p = sub.add_parser("unit-check", help="worker side: the node's identity and the per-unit rule before any write")
    p.add_argument("record"); p.add_argument("--config", required=True); p.add_argument("--units", required=True)
    p = sub.add_parser("verify-chunk", help="verification of one chunk with the frozen verifier and arguments")
    p.add_argument("record"); p.add_argument("--campaign", required=True); p.add_argument("--chunk", type=int, required=True)
    p.add_argument("--tmpdir")
    p = sub.add_parser("release"); p.add_argument("step", choices=["preflight", "inventory", "publish", "verify-bytes", "rollback"])
    p.add_argument("record"); p.add_argument("--campaign", required=True); p.add_argument("--published-root")
    p.add_argument("--manifest"); p.add_argument("--verification-dir"); p.add_argument("--decision", action="append", default=[])
    p.add_argument("--remote-listing"); p.add_argument("--no-portal", action="store_true"); p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--min-free-inodes", type=int, default=50000)
    a = ap.parse_args(argv)

    if a.cmd == "freeze":
        return freeze(a.spec, a.out, a.authorization)
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
        if a.stage == VERIFY and a.cmd != "plan":
            print("status/audit of verification: use plan --stage verify or budget")
            return 2
        us = units_of(cr, "package_parquet" if a.stage == VERIFY else a.stage)
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
            if a.stage == VERIFY:
                if verification_current(rec, cr, u, jobs)[0]:
                    continue
            else:
                s = unit_state(rec, cr, a.stage, u, jobs)
                if s["state"] in ("accepted", "running"):
                    continue
            # the plan lists only units dispatch accepts (same rule; Codex re-review finding 6)
            probs = unit_dispatch_problems(rec, cr, a.stage, u, jobs, cache)
            if probs:
                blocked.append(f"{a.stage} {u}: {probs[:3]}")
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
        raw = Path(a.units).read_text()
        units = [int(x) for x in raw.split()]
        probs = check(rec)
        expected = set(units_of(cr, "package_parquet" if a.stage == VERIFY else a.stage))
        if not units:
            probs.append("empty unit list")
        if len(set(units)) != len(units):
            probs.append("duplicate units in the list")
        out_of = sorted(set(units) - expected)
        if out_of:
            probs.append(f"units {out_of[:10]} are not expected units of {a.stage}")
        try:
            block_nh = a.nodes * parse_time_limit(a.time)
        except ValueError as e:
            probs.append(str(e)); block_nh = 0.0
        if a.nodes <= 0 or a.per_wave <= 0:
            probs.append("nodes and per-wave must be positive")
        jobs = live_jobs()
        cache = {}
        for u in sorted(set(units) & expected):
            probs += unit_dispatch_problems(rec, cr, a.stage, u, jobs, cache)
        # the block is sealed: a read-only, content-addressed copy of the unit list; the driver refuses another one
        digest = hashlib.sha256(raw.encode()).hexdigest()
        sealed = Path(rec["budget"]["ledger"]).parent / "units" / f"{a.campaign}.{a.stage}.{digest[:16]}.txt"
        if probs:
            print("\n".join(probs[:100]))
            print(f"dispatch refused: {len(probs)} problem(s)")
            return 1
        recp = Path(a.record).resolve()
        with _LedgerLock(rec):
            ok, rep_ = budget(rec, block_nh, verbose=False)
            if not ok:
                print(f"dispatch refused: budget: consumed {rep_['consumed']} + forecast {rep_['forecast']} + block "
                      f"{block_nh:.1f} > cap {rep_['cap']}")
                return 1
            sealed.parent.mkdir(parents=True, exist_ok=True)
            if not sealed.exists():
                sealed.write_text(raw)
                os.chmod(sealed, 0o440)
            # every execution node repeats check (images on that node) and the per-unit rule before any write:
            # run_runs_interactive.sh calls `unit-check` when COLLIDERML_LAUNCH_RECORD is set
            if a.stage == VERIFY:
                body = f"bash {REPO}/scripts/production/verify_blocks.sh {recp} {a.campaign} {sealed} {a.per_wave}"
            else:
                body = f"bash {REPO}/scripts/cli/run_blocks_multinode.sh {cr['stages'][a.stage]['config']} --units {sealed} {a.per_wave}"
            cmd = ["sbatch", "-A", a.account, "-C", "cpu", "-q", a.qos, "-N", str(a.nodes), "-t", a.time,
                   "-J", f"v23-{a.campaign}-{a.stage}", "--wrap",
                   f"export COLLIDERML_APPROVED_SHA={rec['approved_sha']} COLLIDERML_UNITS_SHA256={digest} "
                   f"COLLIDERML_LAUNCH_RECORD={recp} COLLIDERML_CAMPAIGN={a.campaign}; "
                   f"cd {REPO} && [ \"$(git rev-parse HEAD)\" = {rec['approved_sha']} ] && "
                   f"{sys.executable} {REPO}/scripts/production/v23_control.py check {recp} && {body}"]
            print(" ".join(cmd))
            if not a.submit:
                return 0
            resv = f"{a.campaign}.{a.stage}.{digest[:16]}.{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')}"
            _ledger_append(rec, {"kind": "reservation", "reservation": resv, "campaign": a.campaign, "stage": a.stage,
                                 "units_file": str(sealed), "units_sha256": digest, "n_units": len(units),
                                 "nodes": a.nodes, "time_limit_h": block_nh / a.nodes,
                                 "utc": datetime.datetime.utcnow().isoformat() + "Z"})
            out = subprocess.run(cmd, capture_output=True, text=True)
            if out.returncode != 0:
                print(out.stderr)
                return 1        # the reservation stays charged at its full limit until reconciled by hand
            jid = out.stdout.split()[-1]
            _ledger_append(rec, {"kind": "submitted", "reservation": resv, "job_id": jid,
                                 "utc": datetime.datetime.utcnow().isoformat() + "Z"})
        print(f"submitted {jid}")
        return 0
    if a.cmd == "unit-check":
        return unit_check(rec, a.config, a.units)
    if a.cmd == "verify-chunk":
        return verify_chunk_cmd(rec, a.record, a.campaign, a.chunk, a.tmpdir)
    if a.cmd == "release":
        if a.step == "preflight":
            probs = release_preflight(rec, a.campaign, a.published_root, a.min_free_inodes)
            print("\n".join(probs) or "preflight ok")
            return 1 if probs else 0
        if a.step == "inventory":
            return release_inventory(rec, a.campaign, a.manifest, a.verification_dir, a.decision, a.jobs)
        if a.step == "publish":
            return release_publish(rec, a.campaign, a.manifest, a.published_root, a.min_free_inodes, a.jobs)
        if a.step == "verify-bytes":
            return release_verify_bytes(rec, a.campaign, a.published_root, a.remote_listing, a.jobs, a.no_portal)
        return release_rollback(rec, a.campaign, a.published_root, a.jobs)
    return 2


if __name__ == "__main__":
    sys.exit(main())
