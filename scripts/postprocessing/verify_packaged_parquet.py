#!/usr/bin/env python3
"""Verify a packaged parquet dataset produced by package_native_parquet.py.

The stage validator checks file presence and size. Those are the cheap failures.
The expensive ones are silent: a chunk that is short a few events because a run
was partially digitized, or tables whose event sets drifted apart so per-event
joins quietly return nothing. Neither shows up as a small file.

This checks, for every chunk:
  * the file exists for every object
  * event_id covers exactly the chunk window (the final chunk may be short)
  * row count and distinct-event count both equal the window size
  * all objects in a chunk carry the IDENTICAL event set
and across the dataset:
  * chunks tile the event range with no gaps and no overlaps

Usage:
    python verify_packaged_parquet.py --config <package_parquet_config.yaml>
    python verify_packaged_parquet.py --config <cfg> --chunks 13-18
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from package_native_parquet import OBJECT_LAYOUT  # noqa: E402


def parse_chunks(spec: str | None, n_chunks: int) -> list[int]:
    if not spec:
        return list(range(n_chunks))
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--chunks", default=None, help="e.g. 13-18 or 0,5,7 (default: all)")
    ap.add_argument("--base", default=None, help="Override output_base_dir")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    campaign, dataset, version = cfg["campaign"], cfg["dataset"], cfg["version"]
    run_size, chunk_size = int(cfg["run_size"]), int(cfg["chunk_size"])
    objects = list(cfg.get("objects", list(OBJECT_LAYOUT)))
    dot = f"{campaign}.{dataset}.{version}"
    base = Path(args.base or cfg["output_base_dir"]) / campaign / dataset / version / "parquet"

    runs_dir = Path(cfg["input_base_dir"]) / campaign / dataset / version / "runs"
    n_runs = len([p for p in runs_dir.iterdir() if p.is_dir() and p.name.isdigit()]) if runs_dir.exists() else 0
    n_chunks = (n_runs * run_size + chunk_size - 1) // chunk_size if n_runs else 0

    chunks = parse_chunks(args.chunks, n_chunks)
    print(f"verifying {dot} under {base}")
    print(f"  {len(chunks)} chunk(s), chunk_size={chunk_size}, objects={objects}\n")

    failures: list[str] = []
    covered: list[tuple[int, int]] = []

    total_events = n_runs * run_size

    for chunk in chunks:
        lo = chunk * chunk_size
        # The final chunk is legitimately short whenever the event count is not a
        # multiple of chunk_size: 2016 runs x 100k = 201.6M over 1M chunks leaves
        # 600k in the last one. Expecting a full chunk there would report a
        # correct dataset as corrupt.
        hi = min(total_events, lo + chunk_size) - 1 if total_events else lo + chunk_size - 1
        expected = hi - lo + 1
        sets: dict[str, set] = {}
        for obj in objects:
            group, name = OBJECT_LAYOUT[obj]
            f = base / group / name / f"{dot}.{group}.{name}.events{lo}-{hi}.parquet"
            if not f.exists():
                failures.append(f"chunk {chunk} {obj}: file missing ({f.name})")
                continue
            ev = pq.read_table(f, columns=["event_id"]).column("event_id").to_pylist()
            if not ev:
                failures.append(f"chunk {chunk} {obj}: empty file")
                continue
            s = set(ev)
            sets[obj] = s
            if min(ev) != lo or max(ev) != hi:
                failures.append(f"chunk {chunk} {obj}: range {min(ev)}-{max(ev)}, expected {lo}-{hi}")
            if len(ev) != expected:
                failures.append(f"chunk {chunk} {obj}: {len(ev)} rows, expected {expected}")
            if len(s) != expected:
                failures.append(f"chunk {chunk} {obj}: {len(s)} distinct events, expected {expected}")

        if len(sets) == len(objects) and sets:
            ref_obj = objects[0]
            ref = sets[ref_obj]
            for obj, s in sets.items():
                if s != ref:
                    failures.append(
                        f"chunk {chunk} {obj}: event set differs from {ref_obj} "
                        f"(+{len(s - ref)} / -{len(ref - s)}) - per-event joins would break"
                    )
        if sets:
            covered.append((lo, hi))
        status = "OK" if not any(f.startswith(f"chunk {chunk} ") for f in failures) else "FAIL"
        print(f"  chunk {chunk:4d}  events {lo}-{hi}  {status}")

    # Tiling: consecutive requested chunks must abut exactly.
    covered.sort()
    for (a_lo, a_hi), (b_lo, b_hi) in zip(covered, covered[1:]):
        if b_lo != a_hi + 1:
            gap = "gap" if b_lo > a_hi + 1 else "overlap"
            failures.append(f"{gap} between events {a_hi} and {b_lo}")

    print()
    if covered:
        print(f"event coverage: {covered[0][0]}-{covered[-1][1]} across {len(covered)} chunk(s)")
    print("RESULT:", "PASS" if not failures else f"FAIL ({len(failures)} problem(s))")
    for f in failures[:20]:
        print("  -", f)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
