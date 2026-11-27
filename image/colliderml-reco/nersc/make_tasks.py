#!/usr/bin/env python3
"""Write the task lists of the release validation (inputs and references on NERSC CFS).

All references were produced with the unoptimised image reco:20261106:
  suite345.json    15 PU200 events, 30 PU0 events and 300 single-particle events
  pu200_192.json   every event of the PU200 ttbar pilot runs 0, 16 and 1562, one event per slice
  pu0_run0.json    PU0 ttbar pilot run 0, 1280 events in the 16 slices of 80 used for the reference
  timing.json      5 PU200 events and one PU0 slice, for single-process timing
  differing.json   the six slices with an event that differed in an earlier optimised build
  reference_variant.json   suite345 plus differing, the set run with the reference variant
  gap_timing.json  two PU200 events, timed with and without the ECAL gap correction

Usage: make_tasks.py OUTPUT_DIR
"""
import json
import sys
from pathlib import Path

TESTBED = Path("/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software/colliderml-release-2/testbed")
PILOT = TESTBED / "v23_plan/pilot/data_main"
GOLDEN = TESTBED / "pandora_speedup_2026-11-22/carbon/validation/golden"
SINGLES = TESTBED / "pflow_cld_parity/E_singles_prodstack/sims"

PU200_RUNS = (0, 16, 1562)
SUITE_PU200 = {0: (0, 4, 7, 38, 46), 16: (0, 7, 13, 36, 49), 1562: (7, 28, 33, 42, 46)}
SUITE_SINGLES = ("gamma_E10_eta0.5", "gamma_E100_eta0.5", "pi-_E10_eta0.5", "pi-_E100_eta0.5",
                 "kaon0L_E10_eta0.5", "kaon0L_E100_eta0.5")


def pu200_input(run):
    return str(PILOT / f"full_pileup/ttbar/v23/runs/{run}/sim_with_tracks.root")


def suite345():
    tasks = []
    for run, entries in SUITE_PU200.items():
        for entry in entries:
            name = f"pu200_r{run}_e{entry:02d}"
            tasks.append({"name": name, "category": "pu200", "input": pu200_input(run),
                          "first": entry, "count": 1, "ref": str(GOLDEN / f"{name}.root")})
    tasks.append({"name": "pu0_r0_e00_29", "category": "pu0",
                  "input": str(PILOT / "hard_scatter/ttbar/v23/runs/0/sim_with_tracks.root"),
                  "first": 0, "count": 30, "ref": str(GOLDEN / "pu0_r0_e00_29.root")})
    for sample in SUITE_SINGLES:
        tasks.append({"name": f"sp_{sample}", "category": "single",
                      "input": str(SINGLES / sample / "sim_with_tracks.root"),
                      "first": 0, "count": 50, "ref": str(GOLDEN / f"sp_{sample}.root")})
    return tasks


def pu200_192():
    return [{"name": f"pu200_r{run}_e{entry:02d}", "category": "pu200", "input": pu200_input(run),
             "first": entry, "count": 1,
             "ref": str(PILOT / f"full_pileup/ttbar/v23/runs/{run}/reco_parts/reco_part_{entry:03d}.root")}
            for run in PU200_RUNS for entry in range(64)]


def pu0_run0():
    run_dir = PILOT / "hard_scatter/ttbar/v23/runs/0"
    return [{"name": f"pu0_r0_slice{i:02d}", "category": "pu0", "input": str(run_dir / "sim_with_tracks.root"),
             "first": 80 * i, "count": 80, "ref": str(run_dir / "reco_edm4hep.root"), "ref_first": 80 * i}
            for i in range(16)]


def timing():
    tasks = [task for task in suite345() if task["name"].startswith("pu200_r0_")]
    tasks.append(pu0_run0()[0])
    return tasks


DIFFERING = ("pu200_r0_e49", "pu200_r0_e57", "pu200_r16_e31", "pu200_r16_e56", "pu200_r1562_e38",
             "pu0_r0_slice02")


def differing():
    return [task for task in pu200_192() + pu0_run0() if task["name"] in DIFFERING]


def main():
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, tasks in (("suite345", suite345()), ("pu200_192", pu200_192()),
                        ("pu0_run0", pu0_run0()), ("timing", timing()), ("differing", differing()),
                        ("reference_variant", suite345() + differing()),
                        ("gap_timing", [t for t in timing() if t["name"] in ("pu200_r0_e00", "pu200_r0_e38")])):
        (out_dir / f"{name}.json").write_text(json.dumps(tasks, indent=1))
        print(f"{name}: {len(tasks)} slices, {sum(task['count'] for task in tasks)} events")


if __name__ == "__main__":
    main()
