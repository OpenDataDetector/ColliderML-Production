#!/usr/bin/env python3
"""Mechanically verify a stage config against declared intent — the gate that
makes "the file doesn't say what the plan said" a hard failure instead of a
production incident.

Born from a real one (2026-07-14): a chained string-replace stacked a zero and
a gun config said events=1000000 while the presented plan said 100000; the
human spot-check verified other fields but not that one. 12.8M wrong-count
events went into a production tree. Machines diff intent; humans shouldn't.

Usage (ALWAYS run before submitting any generated/edited config):
    python3 scripts/cli/verify_config_intent.py <config.yaml> \
        events=100000 job_config.n_runs=128 job_config.runs_per_node=32 \
        job_config.qos=debug job_config.execution_mode=multi_node_slurm \
        version=v1 threads=8

Every argument is a dotted-path=expected pair; values compare as YAML scalars
(so 100000 == "100000" is False for a string field but int comparison is used
when the file holds an int). Exits 0 with a full table on success; exits 1
listing every mismatch. There is no partial pass.
"""
import sys

import yaml


def get_path(cfg, dotted):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return "<MISSING>"
        cur = cur[part]
    return cur


def parse_expected(raw):
    # interpret the expected value with YAML scalar rules (ints, floats, bools)
    return yaml.safe_load(raw)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    path, pairs = sys.argv[1], sys.argv[2:]
    cfg = yaml.safe_load(open(path))
    failures = []
    print(f"intent check: {path}")
    for pair in pairs:
        dotted, _, raw = pair.partition("=")
        expected = parse_expected(raw)
        actual = get_path(cfg, dotted)
        ok = actual == expected
        print(f"  {'OK  ' if ok else 'FAIL'} {dotted:35} expected={expected!r:>12}  actual={actual!r}")
        if not ok:
            failures.append(dotted)
    if failures:
        print(f"INTENT MISMATCH on {len(failures)} field(s): {', '.join(failures)} — DO NOT SUBMIT")
        sys.exit(1)
    print("ALL FIELDS MATCH INTENT")


if __name__ == "__main__":
    main()
