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
  sim_input_name   default edm4hep.root (the ddsim file, for the event-number map)
  sim_input_file   absolute path override for the ddsim file
  calo_input_name  default: reco_edm4hep.root if present, else edm4hep_digitized.root
  events           max events to convert (-1 = all)

Event numbering (the pilot of 2026-09-22 caught this): multithreaded stages write
events in completion order. ddsim already stores its events shuffled relative to
EventHeader.eventNumber, and the ACTS PodioWriter that produces sim_with_tracks.root
shuffles them again, while the ACTS-native parquet tables use the ddsim FILE POSITION
as event_id. So every reco entry is mapped back to its ddsim position through
EventHeader.eventNumber (unique within a run) and that position is written as
event_id. The sim file is ``sim_input_name`` (default edm4hep.root in the run dir)
or ``sim_input_file`` (absolute override, used when the run dir holds no copy).

Pandora slices: if <run>/reco_parts/reco_part_*.root exist (pandora_reco with
merge_slices: false), every object is read from those files instead, each part mapped
to ddsim positions on its own; output rows are sorted by event_id.

Alignment guard: if the run already has an ACTS-native ``particles`` table, the
number of converted events must equal its row count (a truncated k4run must not
pass silently).

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

import numpy as np
import pyarrow.parquet as pq
import uproot
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_calo_digi import convert_file as convert_calo_cells  # noqa: E402
from convert_pfos import convert_file as convert_pfos_and_clusters  # noqa: E402

ALL_OBJECTS = ("calo_cells", "calo_clusters", "pfos")
DEFAULT_RECO_INPUT = "reco_edm4hep.root"
DEFAULT_CALO_DIGI_INPUT = "edm4hep_digitized.root"
DEFAULT_SIM_INPUT = "edm4hep.root"
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
    p.add_argument("--sim-input-name", default=None)
    p.add_argument("--sim-input-file", default=None, help="absolute path to the ddsim file")
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
    if args.sim_input_name is None:
        args.sim_input_name = DEFAULT_SIM_INPUT
    return args


def _event_numbers(path: Path, max_events: int = -1):
    tree = uproot.open(path)["events"]
    stop = None if max_events is None or max_events < 0 else max_events
    arr = tree["EventHeader/EventHeader.eventNumber"].array(entry_stop=stop, library="np")
    return np.asarray([int(x[0]) for x in arr])


def ddsim_positions(reco_file: Path, sim_file: Path, max_events: int = -1) -> np.ndarray:
    """event_id for each reco entry = position of the same EventHeader.eventNumber
    in the ddsim file (what the ACTS-native tables use)."""
    sim_en = _event_numbers(sim_file)
    if len(set(sim_en.tolist())) != len(sim_en):
        raise RuntimeError(f"{sim_file}: EventHeader.eventNumber is not unique; cannot map events")
    pos_of = {int(e): i for i, e in enumerate(sim_en)}
    reco_en = _event_numbers(reco_file, max_events)
    missing = [int(e) for e in reco_en if int(e) not in pos_of]
    if missing:
        raise RuntimeError(f"{reco_file}: {len(missing)} event numbers absent from {sim_file} (first {missing[:5]})")
    ids = np.asarray([pos_of[int(e)] for e in reco_en], dtype=np.int64)
    if len(set(ids.tolist())) != len(ids):
        raise RuntimeError(f"{reco_file}: duplicate event numbers")
    return ids


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


def reco_parts(run_dir: Path) -> list[Path]:
    """Slice files written by pandora_reco with merge_slices: false, in slice order."""
    return sorted((run_dir / "reco_parts").glob("reco_part_*.root"))


def _convert_sources(srcs: list[Path], sim_file: Path, max_events: int, run_dir: Path,
                     want: set[str]) -> dict[str, tuple[Path, int]]:
    """Convert one or more reco files (each mapped to ddsim positions through
    EventHeader.eventNumber) and write one parquet file per object, rows sorted by event_id."""
    import pyarrow as pa

    pieces: dict[str, list[Path]] = {o: [] for o in want}
    total = 0
    for k, src in enumerate(srcs):
        ids = ddsim_positions(src, sim_file, max_events if len(srcs) == 1 else -1)
        logger.info("%s: %d entries mapped to ddsim positions (first %s)", src.name, len(ids), ids[:6].tolist())
        if "calo_cells" in want:
            _, tmp = _atomic_target(run_dir, "calo_cells")
            part = tmp.with_name(f".calo_cells.part{k:03d}.parquet")
            convert_calo_cells(src, part, max_events if len(srcs) == 1 else -1, 0, event_ids=ids)
            pieces["calo_cells"].append(part)
        if want & {"pfos", "calo_clusters"}:
            _, ptmp = _atomic_target(run_dir, "pfos")
            _, ctmp = _atomic_target(run_dir, "calo_clusters")
            pp, cp = ptmp.with_name(f".pfos.part{k:03d}.parquet"), ctmp.with_name(f".calo_clusters.part{k:03d}.parquet")
            convert_pfos_and_clusters(src, pp, cp, max_events if len(srcs) == 1 else -1, 0, event_ids=ids)
            for obj, f in (("pfos", pp), ("calo_clusters", cp)):
                if obj in want:
                    pieces[obj].append(f)
                else:
                    f.unlink()
        total += len(ids)

    written: dict[str, tuple[Path, int]] = {}
    for obj, files in pieces.items():
        out_dir, tmp = _atomic_target(run_dir, obj)
        table = pa.concat_tables([pq.read_table(f) for f in files])
        table = table.sort_by("event_id")
        for f in files:
            f.unlink()
        ev = table.column("event_id").to_numpy()
        if len(np.unique(ev)) != len(ev):
            # Two slices holding the same event would otherwise pass the row-count guard
            # while another event is missing (Codex audit 2026-10-01, finding 1).
            dup = np.unique(ev[np.r_[False, np.diff(ev) == 0]])[:5].tolist()
            raise RuntimeError(f"{obj}: duplicate event_id across reco sources (first {dup})")
        pq.write_table(table, tmp)
        written[obj] = (_finalise(out_dir, tmp, obj, table.num_rows), table.num_rows)
    return written


def convert_run(run_dir: Path, args: argparse.Namespace) -> dict[str, tuple[Path, int]]:
    max_events = int(args.events) if args.events is not None else -1
    sim_file = Path(args.sim_input_file) if args.sim_input_file else run_dir / args.sim_input_name
    if not sim_file.exists():
        raise FileNotFoundError(f"ddsim file {sim_file} not found; needed to map reco entries to event_id")

    parts = reco_parts(run_dir)
    if parts:
        # pandora_reco merge_slices: false. Every object comes from the slices.
        if max_events >= 0:
            raise RuntimeError("an event cap (events >= 0) is not supported with Pandora slice files; "
                               "slices already hold exactly the reconstructed events")
        logger.info("reading %d Pandora slice files from %s", len(parts), run_dir / "reco_parts")
        keys = set(k.split(";")[0] for k in uproot.open(parts[0])["events"].keys())
        for obj in args.objects:
            if REQUIRED_KEY[obj] not in keys:
                raise RuntimeError(f"{obj}: {parts[0].name} has no '{REQUIRED_KEY[obj]}' branch")
        written = _convert_sources(parts, sim_file, max_events, run_dir, set(args.objects))
    else:
        inputs = resolve_inputs(run_dir, args)
        written = {}
        for src in sorted(set(inputs.values())):
            want = {o for o, p in inputs.items() if p == src}
            written.update(_convert_sources([src], sim_file, max_events, run_dir, want))

    counts = {n for _, n in written.values()}
    if len(counts) != 1:
        raise RuntimeError(f"tables disagree on event count: { {k: v[1] for k, v in written.items()} }")
    n_events = counts.pop()
    if n_events == 0:
        raise RuntimeError("zero events converted")

    native = event_count_of_native_table(run_dir)
    if native is not None and max_events < 0:
        if native != n_events:
            raise RuntimeError(
                f"event-count mismatch: reco file has {n_events} events but the ACTS-native "
                f"particles table has {native}; the tables would not align after packaging")
        for obj, (path, _) in written.items():
            ev = np.sort(pq.read_table(path, columns=["event_id"]).column("event_id").to_numpy())
            if not np.array_equal(ev, np.arange(native)):
                raise RuntimeError(f"{obj}: event_ids are not exactly 0..{native - 1}")
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
