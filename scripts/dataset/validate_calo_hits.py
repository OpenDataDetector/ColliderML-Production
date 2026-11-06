"""Validate calo_hits parquet files before upload.

Checks per file:
- every `detector` code is a calorimeter code 9-14 (Release 1 shards had blocks of 255, a lost region label);
- `contrib_particle_ids`, `contrib_energies` (and `contrib_times` if present) have equal lengths per hit;
- optionally, the number of events equals an expected count.

Usage: validate_calo_hits.py [--expected-events N] <file> [<file> ...]
Prints one line per file, OK or BAD; exit status 1 if any file is BAD. See ColliderML-Production#53.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

CALO_DETECTOR_CODES = np.arange(9, 15)
CONTRIB_COLUMNS = ("contrib_particle_ids", "contrib_energies", "contrib_times")


def check_calo_hits_file(path: Path, expected_events: Optional[int] = None) -> Tuple[bool, str]:
    """Return (ok, summary) for one calo_hits parquet file."""
    names = pq.read_schema(path).names
    contrib = [c for c in CONTRIB_COLUMNS if c in names]
    t = pq.read_table(path, columns=["event_id", "detector"] + contrib)
    det = t.column("detector").combine_chunks().flatten().to_numpy(zero_copy_only=False)
    n_bad_det = int(np.sum(~np.isin(det, CALO_DETECTOR_CODES)))
    lens = [pc.list_value_length(t.column(c).combine_chunks().flatten()).to_numpy(zero_copy_only=False)
            for c in contrib]
    n_bad_len = int(np.sum(np.any([lens[0] != other for other in lens[1:]], axis=0))) if len(lens) > 1 else 0
    n_events = len(np.unique(t.column("event_id").to_numpy()))
    ok = n_bad_det == 0 and n_bad_len == 0 and (expected_events is None or n_events == expected_events)
    expected = "" if expected_events is None else f"/{expected_events}"
    return ok, f"events {n_events}{expected} bad_detector_hits {n_bad_det} bad_contrib_hits {n_bad_len}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--expected-events", type=int, default=None,
                        help="Require this many distinct event_ids per file")
    args = parser.parse_args(argv)
    bad = 0
    for f in args.files:
        ok, msg = check_calo_hits_file(f, args.expected_events)
        bad += not ok
        print(f"{'OK ' if ok else 'BAD'} {f.name} {msg}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
