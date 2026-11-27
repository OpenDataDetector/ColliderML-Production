#!/usr/bin/env python3
"""Run a command and append its wall time, exit code and peak memory to a CSV file.

    python3 stage_timer.py --record /output/pipeline_timing.csv --stage simulation -- <command...>

Columns: stage, wall_seconds, exit_code, max_process_rss_mb, cgroup_peak_mb.
``max_process_rss_mb`` is the largest resident set of any single process the command
started. ``cgroup_peak_mb`` is the container's total peak (cgroup v2 ``memory.peak``),
empty where the kernel or the container runtime does not expose it. Standard library only,
so it runs under any python3 of either image.
"""

from __future__ import annotations

import argparse
import csv
import resource
import subprocess
import sys
import time
from pathlib import Path

COLUMNS = ["stage", "wall_seconds", "exit_code", "max_process_rss_mb", "cgroup_peak_mb"]
CGROUP_PEAK = Path("/sys/fs/cgroup/memory.peak")


def cgroup_peak_mb(path: Path = CGROUP_PEAK) -> str:
    try:
        return f"{int(path.read_text().strip()) / 1024**2:.0f}"
    except (OSError, ValueError):
        return ""


def append_record(record_path: Path, row: dict) -> None:
    new_file = not record_path.exists()
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with open(record_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", type=Path, required=True, help="CSV file to append to")
    parser.add_argument("--stage", required=True, help="Stage name written to the first column")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run, after --")
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command given")

    start = time.monotonic()
    exit_code = subprocess.call(command)
    wall = time.monotonic() - start
    # ru_maxrss is in kilobytes on Linux
    rss_mb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    append_record(args.record, {
        "stage": args.stage,
        "wall_seconds": f"{wall:.1f}",
        "exit_code": exit_code,
        "max_process_rss_mb": f"{rss_mb:.0f}",
        "cgroup_peak_mb": cgroup_peak_mb(),
    })
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
