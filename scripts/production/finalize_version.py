#!/usr/bin/env python3
"""Write <version>/.git_commit_success once, after the whole packaging stage has been verified.

For every release chunk 0..N-1 (N = ceil(release_events / chunk_size)):
  * the accepted attempt is the highest-numbered attempt (numeric order); it must have status "complete",
    return_code 0 and git sha == --approved-sha. A later unfinished or failed attempt blocks finalization
    until the operator resolves it;
  * its embedded chunk manifest (container_record, written by the packager) must have status "ok", the
    chunk's exact event window, every configured object, and outputs whose current size and sha256 equal
    the manifest (a deleted, truncated or replaced shard fails);
  * every configured object has exactly one canonically named shard per window under parquet/ and no other
    files there.
Only then is the marker created (exclusive create, never overwritten) and made read-only (0444). Exit 1 with
the problems otherwise; nothing is written.

Writer exclusion (v23, Codex round 5 finding 6): the check runs under <version>/.freeze, created exclusively
before anything is read. run_stage takes its unit lock first and then refuses to run while .freeze or the marker
exists; the finalizer creates .freeze first and then requires that no unit lock exists (runs/*/provenance/*.lock,
provenance/*/*.lock). Either order of the two leaves at most one of them proceeding. .freeze is removed after a
dry run or a refusal and stays after a successful finalization (with the marker, the version refuses writes).

  python finalize_version.py <package_parquet_config.yaml> --approved-sha <sha> [--dry-run]
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import yaml

GROUP = {"particles": "truth", "tracker_simhits": "truth", "generator_particles": "truth",
         "particles_full": "truth", "tracker_hits": "reco", "tracks": "reco", "truth_tracks": "reco",
         "calo_cells": "reco", "calo_clusters": "reco", "pfos": "reco", "vertices": "reco"}
PUBLISHED = {"particles_full": "particles"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def attempt_no(p):
    return int(re.search(r"_attempt(\d+)\.json$", p).group(1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--approved-sha", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    c = yaml.safe_load(Path(a.config).read_text())
    base = (c.get("common") or {}).get("output_base_dir") or c["output_base_dir"]
    vdir = Path(base) / c["campaign"] / c["dataset"] / c["version"]
    freeze = vdir / ".freeze"
    try:
        fd = os.open(freeze, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    except FileExistsError:
        print(f"{freeze} exists (another finalization, verification or publication holds the version); refusing")
        return 1
    with os.fdopen(fd, "w") as f:
        f.write(f"finalize_version.py {'dry run' if a.dry_run else 'finalization'} pid {os.getpid()} "
                f"host {os.uname().nodename} at {datetime.datetime.now().isoformat()}\n")
    try:
        locks = sorted(glob.glob(str(vdir / "runs" / "*" / "provenance" / "*.lock")) +
                       glob.glob(str(vdir / "provenance" / "*" / "*.lock")))
        if locks:
            print(f"{len(locks)} unit lock(s) held (an attempt is running or was interrupted), e.g. {locks[0]}; "
                  "run v23_control.py reconcile first")
            os.unlink(freeze)
            return 1
        rc = _check_and_write(a, c, vdir)
    except BaseException:
        os.unlink(freeze)
        raise
    if rc != 0 or a.dry_run:
        os.unlink(freeze)
    return rc


def _check_and_write(a, c, vdir) -> int:
    release, chunk = int(c["release_events"]), int(c["chunk_size"])
    n = (release + chunk - 1) // chunk
    windows = [(i * chunk, min(release, (i + 1) * chunk) - 1) for i in range(n)]
    objects = list(c["objects"])
    prefix = f"{c['campaign']}.{c['dataset']}.{c['version']}"
    pdir = vdir / "provenance" / "package_parquet"
    problems = [] if objects and release > 0 else ["empty release or object list"]
    expected_files, per_chunk = set(), {}
    for s, e in windows:
        per_chunk[(s, e)] = set()
        for obj in objects:
            g, pub = GROUP[obj], PUBLISHED.get(obj, obj)
            f = str(vdir / "parquet" / g / pub / f"{prefix}.{g}.{pub}.events{s}-{e}.parquet")
            per_chunk[(s, e)].add(f)
            expected_files.add(f)
    for i, (s, e) in enumerate(windows):
        recs = sorted(glob.glob(str(pdir / f"chunk{i}_attempt*.json")), key=attempt_no)
        if not recs:
            problems.append(f"chunk {i}: no attempt")
            continue
        last = json.loads(Path(recs[-1]).read_text())
        if last.get("status") != "complete" or last.get("return_code") != 0:
            problems.append(f"chunk {i}: latest attempt {attempt_no(recs[-1])} is {last.get('status')}/"
                            f"{last.get('return_code')}; resolve it before finalizing")
            continue
        if (last.get("git") or {}).get("sha") != a.approved_sha:
            problems.append(f"chunk {i}: accepted attempt ran {(last.get('git') or {}).get('sha')}, not the approved SHA")
        man = last.get("container_record") or {}
        if man.get("status") != "ok" or man.get("start_event") != s or man.get("end_event") != e:
            problems.append(f"chunk {i}: embedded manifest status {man.get('status')} window "
                            f"{man.get('start_event')}-{man.get('end_event')} != ok {s}-{e}")
            continue
        if sorted(man.get("objects", [])) != sorted(objects):
            problems.append(f"chunk {i}: manifest objects {man.get('objects')} != configured {objects}")
        outs = man.get("outputs") or {}
        if set(outs) != per_chunk[(s, e)]:
            problems.append(f"chunk {i}: manifest outputs are not exactly this chunk's shards "
                            f"({len(set(outs) - per_chunk[(s, e)])} foreign, {len(per_chunk[(s, e)] - set(outs))} missing)")
        for f, meta in outs.items():
            if f not in per_chunk[(s, e)]:
                continue
            elif not Path(f).exists() or os.path.getsize(f) != meta.get("size") or sha256(f) != meta.get("sha256"):
                problems.append(f"chunk {i}: {Path(f).name} missing or changed since the accepted attempt")
    present = set()
    for obj in objects:
        g, pub = GROUP[obj], PUBLISHED.get(obj, obj)
        d = vdir / "parquet" / g / pub
        for p in sorted(d.iterdir()) if d.is_dir() else []:
            present.add(str(p))
            if str(p) not in expected_files:
                problems.append(f"unexpected file {p}")
    missing = sorted(expected_files - present)
    if missing:
        problems.append(f"{len(missing)} expected shards missing, e.g. {missing[0]}")
    marker = vdir / ".git_commit_success"
    if marker.exists():
        problems.append(f"{marker} already exists; it is never overwritten")
    if problems:
        print("\n".join(problems))
        print(f"{len(problems)} problem(s); marker not written")
        return 1
    print(f"{n} chunks x {len(objects)} objects verified, code {a.approved_sha}")
    if a.dry_run:
        return 0
    with open(marker, "x") as f:
        f.write(f"Finalized at {datetime.datetime.now()}\nGit Hash: {a.approved_sha}\nConfig: configs/{Path(a.config).name}\n"
                f"Chunks: {n}\nObjects: {', '.join(objects)}\nStage: package_parquet (finalize_version.py)\n")
    os.chmod(marker, 0o444)
    print(f"wrote {marker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
