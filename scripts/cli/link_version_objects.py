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

  # v23 in staging, parent in the published tree, links to terminal files:
  python link_version_objects.py --dataset-dir .../staging/simulation/full_pileup/ttbar \
      --parent-dataset-dir .../simulation/full_pileup/ttbar --parent v1 --version v23 \
      --runs 0 1562 --run-files edm4hep.root --terminal --apply
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

RANGE = re.compile(r"\.events(\d+)-(\d+)\.parquet$")


def _terminal(src: Path) -> Path:
    """Resolve a chain of links to the real file, so the new version points at the terminal
    file (v23 D4: links only to terminal files listed in the parent baseline). Parquet shards
    must resolve to regular files; run objects may be directories (e.g. generator_particles)."""
    real = Path(os.path.realpath(src))
    if not real.exists():
        raise FileNotFoundError(f"{src} resolves to missing {real}")
    if "/staging/" in str(real):
        raise ValueError(f"{src} resolves into staging ({real}); terminals must be published files")
    return real


def plan(args) -> list[tuple[Path, Path]]:
    base = Path(args.dataset_dir)
    parent_base = Path(args.parent_dataset_dir) if args.parent_dataset_dir else base
    parent, new = parent_base / args.parent, base / args.version
    links = []
    if args.runs:
        for n in range(args.runs[0], args.runs[1] + 1):
            for name in args.run_files:
                src = parent / "runs" / str(n) / name
                if not src.exists():
                    raise FileNotFoundError(src)
                links.append((new / "runs" / str(n) / name, _terminal(src) if args.terminal else src))
    for obj in args.parquet:
        lo, hi = args.events
        found = 0
        wins = []
        for src in sorted((parent / "parquet" / obj).glob("*.parquet")):
            m = RANGE.search(src.name)
            if m and int(m.group(1)) <= hi and int(m.group(2)) >= lo:
                name = src.name
                if args.rename_version:
                    # <campaign>.<dataset>.<parent version>.<group>... -> the new version's name
                    parts = name.split(".")
                    if len(parts) < 4 or parts[2] != args.parent_name_version:
                        raise ValueError(f"{name}: cannot rename, third field is not {args.parent_name_version!r}")
                    parts[2] = args.version
                    name = ".".join(parts)
                links.append((new / "parquet" / obj / name, _terminal(src) if args.terminal else src))
                wins.append((int(m.group(1)), int(m.group(2))))
                found += 1
        wins.sort()
        gaps = [(a, b) for (_, a), (b, _) in zip(wins, wins[1:]) if b != a + 1]
        if args.expect_files is not None and (gaps or not wins or wins[0][0] > lo or wins[-1][1] < hi):
            raise ValueError(f"{obj}: windows do not tile events {lo}-{hi} (gaps {gaps[:3]}, span {wins[:1]}..{wins[-1:]})")
        if not found:
            raise FileNotFoundError(f"no {obj} files in {parent / 'parquet' / obj} overlap events {lo}-{hi}")
        if args.expect_files is not None and found != args.expect_files:
            raise ValueError(f"{obj}: {found} parent files overlap events {lo}-{hi}, expected {args.expect_files}")
        if args.terminal:
            for dst, src in links[-found:]:
                if not src.is_file():
                    raise ValueError(f"{dst.name}: terminal {src} is not a regular file")
    return links


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", required=True, help="<simulation>/<campaign>/<dataset>")
    p.add_argument("--parent", required=True, help="existing version, e.g. v1")
    p.add_argument("--version", required=True, help="new version, e.g. v20")
    p.add_argument("--parent-dataset-dir", default=None,
                   help="<base>/<campaign>/<dataset> holding the parent when it is not under --dataset-dir "
                        "(e.g. the new version is built in staging or a pilot area)")
    p.add_argument("--terminal", action="store_true",
                   help="link to the resolved terminal file instead of the parent's path (follows parent links)")
    p.add_argument("--runs", type=int, nargs=2, metavar=("FIRST", "LAST"), help="inclusive run range")
    p.add_argument("--run-files", nargs="*", default=["edm4hep.root"])
    p.add_argument("--parquet", nargs="*", default=[], help="objects under parquet/, e.g. truth/particles")
    p.add_argument("--events", type=int, nargs=2, metavar=("FIRST", "LAST"), help="global event range for --parquet")
    p.add_argument("--expect-files", type=int, default=None,
                   help="with --parquet: exact number of parent files per object (e.g. 1000 for PU200 v23)")
    p.add_argument("--rename-version", action="store_true",
                   help="name parquet links with the new version (third name field), target unchanged")
    p.add_argument("--parent-name-version", default=None,
                   help="version written in the parent shard names (default: --parent)")
    p.add_argument("--protect", action="store_true",
                   help="make each terminal target read-only (chmod a-w; permission bits only)")
    p.add_argument("--apply", action="store_true", help="create the links (default: dry run)")
    args = p.parse_args()
    if args.parquet and not args.events:
        p.error("--parquet needs --events")

    args.parent_name_version = args.parent_name_version or args.parent
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
    if args.apply and args.protect:
        import stat
        for _, src in links:
            targets = [src] if src.is_file() else [x for x in src.rglob("*") if x.is_file()]
            for t in targets:
                mode = t.stat().st_mode
                if mode & 0o222:
                    os.chmod(t, mode & ~0o222)
                    print(f"protected {t}")
    print(f"{len(links)} links {'in place' if args.apply else 'planned (dry run)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
