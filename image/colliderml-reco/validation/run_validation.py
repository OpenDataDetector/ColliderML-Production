#!/usr/bin/env python3
"""Reconstruct a list of event slices in the image and compare each with a reference file.

Host side:
    run_validation.py --image IMAGE --tasks tasks.json --out results.json \\
        --work-dir /tmp/reco_validation --mount /data:/data [--workers N]

It starts one container, which runs this same file with --inner: one
"colliderml-reco run" per task, each pinned to its own core, followed by compare_reco.py.

Exit code: 0 if every slice ran and equals its reference, 1 if all slices ran and some
differ, 2 if a run or a comparison did not complete.

tasks.json is a list of objects:
    name       label of the slice
    category   free-form group label, used in the summary
    input      EDM4hep input file (a path visible inside the container, see --mount)
    first      first input entry
    count      number of events
    ref        reference output file
    ref_first  optional: first entry of this slice in a merged reference file
"""
import argparse
import concurrent.futures
import json
import os
import queue
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_task(task, work_dir, cores, env_extra, k4run_args, keep_outputs):
    """Reconstruct one slice on a free core, compare it with its reference, return a record."""
    task_dir = work_dir / task["name"]
    task_dir.mkdir(parents=True, exist_ok=True)
    output = task_dir / "reco.root"
    log_path = task_dir / "k4run.log"
    core = cores.get()
    try:
        command = [
            "taskset", "-c", str(core),
            "colliderml-reco", "run",
            "--input", task["input"], "--output", str(output),
            "--first-event", str(task["first"]), "--events", str(task["count"]),
            "--work-dir", str(task_dir),
        ]
        if k4run_args:
            command += ["--", *k4run_args]
        start = time.monotonic()
        with open(log_path, "w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=dict(os.environ, **env_extra))
            _, status, usage = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
        wall = time.monotonic() - start
    finally:
        cores.put(core)

    record = {
        "name": task["name"], "category": task["category"], "count": task["count"],
        "rc": process.returncode,
        "wall_s": round(wall, 2),
        "max_rss_gb": round(usage.ru_maxrss / 1024**2, 3),   # ru_maxrss is in KiB; this is GiB
    }
    log_text = log_path.read_text(errors="replace")
    pandora_times = re.findall(r"PANDORA ALGORITHM SELF-TIME \(this event\) : total=([0-9.]+) s", log_text)
    if pandora_times:
        record["pandora_s"] = round(sum(map(float, pandora_times)), 2)
    if record["rc"] != 0:
        record["error"] = "k4run failed: " + log_text[-1500:]
        return record

    compare = [sys.executable, str(HERE / "compare_reco.py"), task["ref"], str(output),
               "--expect-events", str(task["count"])]
    if "ref_first" in task:
        compare += ["--ref-first", str(task["ref_first"])]
    done = subprocess.run(compare, capture_output=True, text=True)
    try:
        result = json.loads(done.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        result = None
    if result is None or done.returncode != (0 if result["identical"] else 1):
        record["error"] = "comparison failed: " + (done.stdout + done.stderr)[-1500:]
        return record
    record.update(result)
    if not keep_outputs:
        output.unlink(missing_ok=True)
    return record


def inner(config_path):
    config = json.loads(Path(config_path).read_text())
    work_dir = Path(config["work_dir"])
    cores = queue.Queue()
    first = config["first_core"]
    selected = sorted(os.sched_getaffinity(0))[first: first + config["workers"]]
    if len(selected) < config["workers"]:
        raise SystemExit(f"{config['workers']} workers from core index {first} requested, "
                         f"{len(selected)} cores available there")
    for core in selected:
        cores.put(core)
    env_extra = dict(item.split("=", 1) for item in config["env"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        futures = [pool.submit(run_task, task, work_dir, cores, env_extra, config["k4run_args"],
                               config["keep_outputs"])
                   for task in config["tasks"]]
        records = [future.result() for future in futures]
    (work_dir / "records.json").write_text(json.dumps(records, indent=1))


def summarise(records):
    """Per-category totals: events, events equal to the reference, time per event."""
    summary = {}
    for record in records:
        entry = summary.setdefault(record["category"], {
            "slices": 0, "events": 0, "slices_with_error": 0, "events_equal": 0,
            "calo_hits_compared": 0, "pandora_objects_compared": 0,
            "wall_s": 0.0, "pandora_s": 0.0, "max_rss_gb": 0.0,
        })
        entry["slices"] += 1
        entry["events"] += record["count"]
        entry["wall_s"] += record["wall_s"]
        entry["pandora_s"] += record.get("pandora_s", 0.0)
        entry["max_rss_gb"] = max(entry["max_rss_gb"], record["max_rss_gb"])
        if "error" in record:
            entry["slices_with_error"] += 1
            continue
        differing = set(record["pandora_events_differing"]) | set(record["calo_events_differing"])
        entry["events_equal"] += record["count"] - len(differing)
        entry["calo_hits_compared"] += record["calo_hits_compared"]
        entry["pandora_objects_compared"] += record["pandora_objects_compared"]
    for entry in summary.values():
        entry["wall_s_per_event"] = round(entry["wall_s"] / entry["events"], 3)
        entry["pandora_s_per_event"] = round(entry["pandora_s"] / entry["events"], 3)
        entry["wall_s"] = round(entry["wall_s"], 1)
        entry["pandora_s"] = round(entry["pandora_s"], 1)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inner", metavar="CONFIG", help="run inside the container (set by the host side)")
    parser.add_argument("--image")
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--work-dir", type=Path, help="scratch directory, about 0.2 GB per PU200 event")
    parser.add_argument("--engine", default="podman")
    parser.add_argument("--workers", type=int, default=8, help="parallel k4run processes (about 5.5 GB each at PU200)")
    parser.add_argument("--first-core", type=int, default=0,
                        help="index of the first core to pin to, for two runs on one machine")
    parser.add_argument("--mount", action="append", default=[], metavar="SRC:DST",
                        help="read-only bind mount for inputs and references (repeatable)")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                        help="environment for k4run, e.g. PANDORA_ALG_TIMING=1 (repeatable)")
    parser.add_argument("--k4run-arg", action="append", default=[], metavar="ARG",
                        help="extra k4run argument for every task (repeatable)")
    parser.add_argument("--run-arg", action="append", default=[], metavar="ARG",
                        help="extra argument for '<engine> run', e.g. --run-arg=--userns=keep-id so that "
                             "the files in --work-dir belong to you (repeatable)")
    parser.add_argument("--keep-outputs", action="store_true")
    args = parser.parse_args()

    if args.inner:
        inner(args.inner)
        return
    for name in ("image", "tasks", "out", "work_dir"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    args.out.resolve().parent.mkdir(parents=True, exist_ok=True)
    tasks = json.loads(args.tasks.read_text())
    if not tasks:
        parser.error(f"{args.tasks} has no tasks")
    (work_dir / "config.json").write_text(json.dumps({
        "tasks": tasks, "work_dir": "/validation-work", "workers": args.workers,
        "first_core": args.first_core,
        "env": args.env, "k4run_args": args.k4run_arg, "keep_outputs": args.keep_outputs,
    }))
    image_id = subprocess.check_output(
        [args.engine, "image", "inspect", "--format", "{{.Id}}", args.image]).decode().strip()

    command = [args.engine, "run", "--rm", "--network=none", *args.run_arg,
               "-v", f"{HERE}:/validation:ro", "-v", f"{work_dir}:/validation-work"]
    for mount in args.mount:
        command += ["-v", f"{mount}:ro"]
    command += [args.image, "python3", "/validation/run_validation.py",
                "--inner", "/validation-work/config.json"]
    records_path = work_dir / "records.json"
    if records_path.exists():
        records_path.unlink()
    start = time.monotonic()
    container = subprocess.run(command)
    total_wall = time.monotonic() - start
    if container.returncode != 0 or not records_path.exists():
        print(f"container run failed with exit code {container.returncode}; no report written", file=sys.stderr)
        sys.exit(2)

    records = json.loads(records_path.read_text())
    errors = sum("error" in record for record in records)
    report = {
        "image": args.image,
        "image_id": image_id,
        "workers": args.workers,
        "env": args.env,
        "k4run_args": args.k4run_arg,
        "total_wall_s": round(total_wall, 1),
        "slices_with_error": errors,
        "all_identical": errors == 0 and all(record["identical"] for record in records),
        "summary": summarise(records),
        "records": records,
    }
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=1))
    sys.exit(2 if errors else 0 if report["all_identical"] else 1)


if __name__ == "__main__":
    main()
