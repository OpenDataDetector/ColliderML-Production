#!/usr/bin/env python3
"""Symlink unchanged objects of an existing dataset version into a new version.

A new version (e.g. v20, a reconstruction pass on the Release 1 sim) reuses the parent
version's inputs instead of copying them:

  * per-run files, e.g. runs/<N>/edm4hep.root  -> <parent>/runs/<N>/edm4hep.root
  * published parquet files of whole objects, e.g. parquet/truth/particles, for the
    files whose event range overlaps the new version's events. The link keeps the
    parent's file name (it carries the parent version), so it is clear what it is.

Never overwrites: an existing path that is not already the identical link is an error.
Default is a dry run; pass --apply to create the links.

  python link_version_objects.py --dataset-dir .../simulation/hard_scatter/ttbar \
      --parent v1 --version v20 --runs 0 7 --run-files edm4hep.root \
      --parquet truth/particles --events 0 10239 --apply
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

RANGE = re.compile(r"\.events(\d+)-(\d+)\.parquet$")


def plan(args) -> list[tuple[Path, Path]]:
    base = Path(args.dataset_dir)
    parent, new = base / args.parent, base / args.version
    links = []
    if args.runs:
        for n in range(args.runs[0], args.runs[1] + 1):
            for name in args.run_files:
                src = parent / "runs" / str(n) / name
                if not src.exists():
                    raise FileNotFoundError(src)
                links.append((new / "runs" / str(n) / name, src))
    for obj in args.parquet:
        lo, hi = args.events
        found = 0
        for src in sorted((parent / "parquet" / obj).glob("*.parquet")):
            m = RANGE.search(src.name)
            if m and int(m.group(1)) <= hi and int(m.group(2)) >= lo:
                links.append((new / "parquet" / obj / src.name, src))
                found += 1
        if not found:
            raise FileNotFoundError(f"no {obj} files in {parent / 'parquet' / obj} overlap events {lo}-{hi}")
    return links


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", required=True, help="<simulation>/<campaign>/<dataset>")
    p.add_argument("--parent", required=True, help="existing version, e.g. v1")
    p.add_argument("--version", required=True, help="new version, e.g. v20")
    p.add_argument("--runs", type=int, nargs=2, metavar=("FIRST", "LAST"), help="inclusive run range")
    p.add_argument("--run-files", nargs="*", default=["edm4hep.root"])
    p.add_argument("--parquet", nargs="*", default=[], help="objects under parquet/, e.g. truth/particles")
    p.add_argument("--events", type=int, nargs=2, metavar=("FIRST", "LAST"), help="global event range for --parquet")
    p.add_argument("--apply", action="store_true", help="create the links (default: dry run)")
    args = p.parse_args()
    if args.parquet and not args.events:
        p.error("--parquet needs --events")

    links = plan(args)
    for dst, src in links:
        if dst.is_symlink() and Path(os.readlink(dst)) == src:
            print(f"exists  {dst}")
            continue
        if dst.exists() or dst.is_symlink():
            print(f"REFUSE  {dst} exists and is not a link to {src}", file=sys.stderr)
            return 1
        print(f"{'link' if args.apply else 'would link'}  {dst} -> {src}")
        if args.apply:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src)
    print(f"{len(links)} links {'in place' if args.apply else 'planned (dry run)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
