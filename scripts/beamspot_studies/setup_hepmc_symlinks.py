#!/usr/bin/env python3
"""
Create symlinks to existing HepMC files for beam spot study datasets.

This avoids re-running MadGraph generation by reusing the hard_scatter/ttbar/v1
HepMC event files. DDSim will read from these symlinks and apply the new
vertexOffset and vertexSigma from the beam spot study configs.

Usage:
    python setup_hepmc_symlinks.py \
        --source-dir /global/cfs/cdirs/m4958/data/ColliderML/simulation/hard_scatter/ttbar/v1/runs \
        --target-dir /global/cfs/cdirs/m4958/data/ColliderML/simulation/beamspot_studies/ttbar_shifted_300um/v1/runs \
        --n-runs 50 \
        --input-filename events.hepmc
"""

import argparse
import sys
from pathlib import Path


def setup_symlinks(source_dir, target_dir, n_runs, input_filename):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    if not source_dir.exists():
        print(f"ERROR: Source directory does not exist: {source_dir}")
        sys.exit(1)

    # Validate all source files exist before creating anything
    missing = []
    for i in range(n_runs):
        src = source_dir / str(i) / input_filename
        if not src.exists():
            missing.append(src)

    if missing:
        print(f"ERROR: {len(missing)} source files missing:")
        for m in missing[:10]:
            print(f"  {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
        sys.exit(1)

    print(f"All {n_runs} source files verified.")

    # Create symlinks
    created = 0
    skipped = 0
    for i in range(n_runs):
        src = source_dir / str(i) / input_filename
        tgt_dir = target_dir / str(i)
        tgt = tgt_dir / input_filename

        tgt_dir.mkdir(parents=True, exist_ok=True)

        if tgt.exists() or tgt.is_symlink():
            if tgt.is_symlink() and tgt.resolve() == src.resolve():
                skipped += 1
                continue
            else:
                print(f"WARNING: {tgt} already exists and points elsewhere. Skipping.")
                skipped += 1
                continue

        tgt.symlink_to(src)
        created += 1

    print(f"Created {created} symlinks, skipped {skipped} (already exist).")
    print(f"Target directory: {target_dir}")


def main():
    parser = argparse.ArgumentParser(description="Set up HepMC symlinks for beam spot studies")
    parser.add_argument("--source-dir", required=True, help="Source runs directory (e.g., .../hard_scatter/ttbar/v1/runs)")
    parser.add_argument("--target-dir", required=True, help="Target runs directory (e.g., .../beamspot_studies/ttbar_shifted_300um/v1/runs)")
    parser.add_argument("--n-runs", type=int, required=True, help="Number of runs to symlink")
    parser.add_argument("--input-filename", default="events.hepmc", help="HepMC filename (default: events.hepmc)")
    args = parser.parse_args()

    setup_symlinks(args.source_dir, args.target_dir, args.n_runs, args.input_filename)


if __name__ == "__main__":
    main()
