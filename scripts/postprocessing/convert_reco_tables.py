#!/usr/bin/env python3
"""Per-run stage: write the Release-2 calorimeter / particle-flow parquet tables.

Reads the k4ODD reconstruction output of ONE run directory and drops one
parquet file per table next to the ACTS-native tracker tables, so that
``package_native_parquet.py`` can publish them with the same chunking and
global event numbering as everything else:

    <runs>/<N>/calo_cells/calo_cells_000000-<n>.parquet        (CALO_CELLS_PARQUET_TYPES)
    <runs>/<N>/calo_clusters/calo_clusters_000000-<n>.parquet  (CALO_CLUSTERS_PARQUET_TYPES)
    <runs>/<N>/pfos/pfos_000000-<n>.parquet                    (PFOS_PARQUET_TYPES)

event_id restarts at 0 in every run, exactly like the ACTS-native writer; the
packager adds ``run * run_size``.

Inputs (all in the run directory):
  * ``reco_edm4hep.root`` from ``pandora_reco`` carries the digitised cells
    (ODDreconstruction.py runs DDCaloDigi inline) AND the PFOs/clusters, so it
    serves every table.
  * ``edm4hep_digitized.root`` from ``calo_digitization`` carries cells only;
    it is the fallback for ``calo_cells`` when Pandora has not run.

Stage config keys (all optional):
  objects          subset of [calo_cells, calo_clusters, pfos]; default all three
  reco_input_name  default reco_edm4hep.root
  calo_input_name  default: reco_edm4hep.root if present, else edm4hep_digitized.root
  events           max events to convert (-1 = all)

Alignment guard: if the run already has an ACTS-native ``particles`` table, the
number of converted events must equal its row count. The reco file and the
tracker tables both descend from the same edm4hep.root in file order, so a
mismatch means a truncated k4run, and it must not pass silently.

This is a "simulation-shaped" stage in cli_utils terms (it receives --output
<runs dir> --output-subdir <run>), and it runs in the reco image via
``common.stage_containers.reco_tables``.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
import traceback
from pathlib import Path

import pyarrow.parquet as pq
import uproot
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_calo_digi import convert_file as convert_calo_cells  # noqa: E402
from convert_pfos import convert_file as convert_pfos_and_clusters  # noqa: E402

ALL_OBJECTS = ("calo_cells", "calo_clusters", "pfos")
DEFAULT_RECO_INPUT = "reco_edm4hep.root"
DEFAULT_CALO_DIGI_INPUT = "edm4hep_digitized.root"
# A branch that only exists if the corresponding producer ran.
REQUIRED_KEY = {
    "calo_cells": "digiECalBarrelCollection",
    "calo_clusters": "GaudiPandoraClusters",
    "pfos": "GaudiPandoraPFOs",
}

logger = logging.getLogger("reco_tables")


def _load_timing_recorder():
    """TimingRecorder lives in scripts/simulation/utils; import it by path so the
    two ``utils`` packages (simulation vs postprocessing) never collide."""
    path = Path(__file__).resolve().parent.parent / "simulation" / "utils" / "app_logging.py"
    spec = importlib.util.spec_from_file_location("sim_app_logging", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TimingRecorder


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None, help="YAML stage configuration")
    p.add_argument("--output", "-o", type=Path, required=True, help="runs directory")
    p.add_argument("--output-subdir", type=str, default=None, help="run number (subdirectory of --output)")
    p.add_argument("--events", "-n", type=int, default=None, help="max events (-1 = all)")
    p.add_argument("--objects", nargs="+", choices=ALL_OBJECTS, default=None)
    p.add_argument("--reco-input-name", default=None)
    p.add_argument("--calo-input-name", default=None)
    # accepted for run_stage compatibility, unused
    p.add_argument("--seed", default=None)
    p.add_argument("--performance-metrics", action="store_true", default=None)
    return p


def load_config(args: argparse.Namespace) -> argparse.Namespace:
    if args.config is not None:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        for key, value in cfg.items():
            if getattr(args, key, None) is None:
                setattr(args, key, value)
    if args.events is None:
        args.events = -1
    if args.objects is None:
        args.objects = list(ALL_OBJECTS)
    if args.reco_input_name is None:
        args.reco_input_name = DEFAULT_RECO_INPUT
    return args


def event_count_of_native_table(run_dir: Path, table: str = "particles") -> int | None:
    """Row count (= event count) of an ACTS-native per-run table, if present."""
    files = sorted((run_dir / table).glob("*.parquet"))
    if not files:
        return None
    return sum(pq.ParquetFile(f).metadata.num_rows for f in files)


def resolve_inputs(run_dir: Path, args: argparse.Namespace) -> dict[str, Path]:
    """Map each requested object to the file it is read from, failing loudly if
    the producer stage has not run."""
    reco = run_dir / args.reco_input_name
    if args.calo_input_name:
        calo = run_dir / args.calo_input_name
    else:
        calo = reco if reco.exists() else run_dir / DEFAULT_CALO_DIGI_INPUT

    inputs: dict[str, Path] = {}
    for obj in args.objects:
        src = calo if obj == "calo_cells" else reco
        if not src.exists():
            raise FileNotFoundError(
                f"{obj}: input {src} not found - run "
                f"{'calo_digitization or pandora_reco' if obj == 'calo_cells' else 'pandora_reco'} first")
        inputs[obj] = src

    for src in set(inputs.values()):
        keys = set(k.split(";")[0] for k in uproot.open(src)["events"].keys())
        for obj, path in inputs.items():
            if path == src and REQUIRED_KEY[obj] not in keys:
                raise RuntimeError(
                    f"{obj}: {src.name} has no '{REQUIRED_KEY[obj]}' branch - "
                    "the producing algorithm did not run (silent reco failure)")
    return inputs


def _atomic_target(run_dir: Path, obj: str) -> tuple[Path, Path]:
    out_dir = run_dir / obj
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f".{obj}.partial.parquet"
    if tmp.exists():
        tmp.unlink()
    return out_dir, tmp


def _finalise(out_dir: Path, tmp: Path, obj: str, n_events: int) -> Path:
    final = out_dir / f"{obj}_000000-{n_events:06d}.parquet"
    os.replace(tmp, final)
    return final


def convert_run(run_dir: Path, args: argparse.Namespace) -> dict[str, tuple[Path, int]]:
    inputs = resolve_inputs(run_dir, args)
    max_events = int(args.events) if args.events is not None else -1
    written: dict[str, tuple[Path, int]] = {}

    if "calo_cells" in inputs:
        out_dir, tmp = _atomic_target(run_dir, "calo_cells")
        n = convert_calo_cells(inputs["calo_cells"], tmp, max_events, 0)
        written["calo_cells"] = (_finalise(out_dir, tmp, "calo_cells", n), n)

    want_pfos = "pfos" in inputs
    want_clu = "calo_clusters" in inputs
    if want_pfos or want_clu:
        # convert_pfos always writes both; drop the one that was not requested.
        src = inputs.get("pfos") or inputs["calo_clusters"]
        pfo_dir, pfo_tmp = _atomic_target(run_dir, "pfos")
        clu_dir, clu_tmp = _atomic_target(run_dir, "calo_clusters")
        n = convert_pfos_and_clusters(src, pfo_tmp, clu_tmp, max_events, 0)
        if want_pfos:
            written["pfos"] = (_finalise(pfo_dir, pfo_tmp, "pfos", n), n)
        else:
            pfo_tmp.unlink()
        if want_clu:
            written["calo_clusters"] = (_finalise(clu_dir, clu_tmp, "calo_clusters", n), n)
        else:
            clu_tmp.unlink()

    counts = {n for _, n in written.values()}
    if len(counts) != 1:
        raise RuntimeError(f"tables disagree on event count: { {k: v[1] for k, v in written.items()} }")
    n_events = counts.pop()
    if n_events == 0:
        raise RuntimeError("zero events converted")

    native = event_count_of_native_table(run_dir)
    if native is not None and max_events < 0 and native != n_events:
        raise RuntimeError(
            f"event-count mismatch: reco file has {n_events} events but the ACTS-native "
            f"particles table has {native}; the tables would not align after packaging")
    return written


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)-12s %(message)s")
    args = load_config(build_parser().parse_args())
    run_dir = Path(args.output)
    if args.output_subdir:
        run_dir = run_dir / str(args.output_subdir)
    if not run_dir.is_dir():
        logger.error("run directory does not exist: %s", run_dir)
        return 1

    TimingRecorder = _load_timing_recorder()
    timer = TimingRecorder(run_dir)
    logger.info("=" * 80)
    logger.info("Release-2 reco tables for %s: %s", run_dir, args.objects)
    logger.info("=" * 80)
    try:
        with timer.record("Reco Tables"):
            written = convert_run(run_dir, args)
        timer.write_report()
    except Exception as exc:  # noqa: BLE001 - stage boundary, report everything
        timer.write_report()
        logger.error("Fatal error in reco_tables: %s", exc)
        logger.error(traceback.format_exc())
        return 1

    for obj, (path, n) in written.items():
        logger.info("  %-14s %s (%d events, %.1f MB)", obj, path.name, n, path.stat().st_size / 1024**2)
    logger.info("✓ reco tables complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
