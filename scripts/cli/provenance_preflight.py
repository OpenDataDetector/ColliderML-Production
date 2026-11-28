#!/usr/bin/env python3
"""Preflight of the provenance gate, run by the shell drivers BEFORE they create log directories or load
images: exit 0 if the config has no provenance gate or the gate passes, else print the reason and exit 1.
  python provenance_preflight.py <config.yaml>"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_provenance  # noqa: E402


def main() -> int:
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
    prov = (cfg or {}).get("provenance") or {}
    req = run_provenance.provenance_required_error(cfg or {})
    if req:                     # .provenance_required: before the driver creates a log dir or loads an image
        print(f"provenance preflight: {req}")
        return 1
    if not prov.get("require_pushed_commit"):
        return 0
    import cli_utils  # the same base-path guard run_stage applies, before the driver writes anything
    try:
        cli_utils.load_and_process_config(sys.argv[1], Path(__file__).resolve().parent / "env_setup.yaml")
    except ValueError as e:
        print(f"provenance preflight: {e}")
        return 1
    repo = Path(__file__).resolve().parents[2]
    if not (repo / ".git").exists():
        print(f"provenance preflight: {repo} is not a git checkout")
        return 1
    state, err = run_provenance.gate_error(cfg, repo)
    if err:
        print(f"provenance preflight: {err}")
        return 1
    print(f"provenance preflight: {state['sha']} ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
