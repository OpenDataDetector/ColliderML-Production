#!/usr/bin/env python3
"""Package per-run ACTS-native parquet into the published dataset layout.

The native Arrow writer in digi_and_reco drops one parquet file per table per
run, under ``<runs>/<N>/<table>/<table>_000000-100000.parquet``, with event_id
restarting at 0 in every run. That is a production intermediate, not the
published form. This step converts it to the same layout convert_all.py has
always produced:

    <out>/<campaign>/<dataset>/<version>/parquet/truth/particles/
                                                /truth/tracker_simhits/
                                                /reco/tracker_hits/
                                                /reco/tracks/
                                                /reco/truth_tracks/
        <campaign>.<dataset>.<version>.<group>.<object>.events<START>-<END>.parquet

Two conventions are inherited from convert_all.py deliberately, so the native
path and the legacy path produce interchangeable datasets:

  * **Global event numbering.** ``event_id += abs_run * run_size``
    (convert_all.py:265, "Update event_ids to global numbering"). There is no
    run_id column - the run is recoverable as ``event_id // run_size``.
  * **Chunking by events, not by runs.** ``chunk_size`` is the target number of
    events per output file and a chunk freely spans runs, exactly as
    utils/driver.py:iterate_and_process_chunks does for the legacy path. Runs
    are an input detail; they are not visible in the output.

Unlike convert_all.py this reads parquet rather than edm4hep/ROOT, so it stays
in Arrow the whole way: pandas would flatten the nested list columns
(``hit_ids`` is list<list<uint32>>) and cost a large copy. Tables are streamed
into the output file run by run rather than concatenated, so peak memory is one
run's table rather than a whole chunk's.

Usage (matches convert_all.py, so run_stage drives it identically):
    python package_native_parquet.py --config <cfg.yaml> [--chunk-index N]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.driver import iterate_and_process_chunks, local_events_for_run  # noqa: E402

logger = logging.getLogger(__name__)

# Where each native table lands in the published tree. truth/ is what the
# simulation knows; reco/ is what reconstruction produced. truth_tracks sits
# under reco/ because it IS a reconstructed object - truth only supplied the
# hit-to-track assignment, the fit is real.
OBJECT_LAYOUT: dict[str, tuple[str, str]] = {
    "particles": ("truth", "particles"),
    "tracker_simhits": ("truth", "tracker_simhits"),
    "tracker_hits": ("reco", "tracker_hits"),
    "tracks": ("reco", "tracks"),
    "truth_tracks": ("reco", "truth_tracks"),
}

# The ACTS-native writer emits time-like quantities in ACTS native units
# (lengths, c=1): a "time" is a path length in mm. The legacy convert_all
# datasets are in nanoseconds, and mm-as-ns is exactly the field bug a
# downstream user caught (pixel hit "271 ns" that was really 271 mm = 0.91 ns).
# Convert at packaging time so the published tables are in ns; per-run native
# intermediates stay untouched. Value is the unit power: 1 for times (/c),
# 2 for variances (/c^2).
_MM_PER_NS = 299.792458
TIME_COLUMNS: dict[str, dict[str, int]] = {
    "tracker_hits": {"time": 1, "var_time": 2},
    "tracker_simhits": {"true_time": 1},
    "particles": {"time": 1},
    "tracks": {"t": 1},
    "truth_tracks": {"t": 1},
}


def _convert_times(table: pa.Table, obj: str) -> pa.Table:
    """Divide time-like columns by c (mm/ns), power-aware, list-layout-aware."""
    for col, power in TIME_COLUMNS.get(obj, {}).items():
        if col not in table.column_names:
            logger.warning("%s: expected time column %r absent, skipping", obj, col)
            continue
        idx = table.schema.get_field_index(col)
        factor = _MM_PER_NS ** power
        chunks = []
        for chunk in table.column(col).chunks:
            if pa.types.is_list(chunk.type):
                vals = pc.divide(chunk.values, pa.scalar(factor, type=chunk.type.value_type))
                chunks.append(pa.ListArray.from_arrays(chunk.offsets, vals))
            else:
                chunks.append(pc.divide(chunk, pa.scalar(factor, type=chunk.type)))
        table = table.set_column(idx, col, pa.chunked_array(chunks))
    return table


def _read_run_table(run_dir: Path, obj: str) -> pa.Table | None:
    """Read a run's single native parquet file for one object."""
    files = sorted((run_dir / obj).glob("*.parquet"))
    if not files:
        return None
    if len(files) > 1:
        # The native writer emits one shard per run when eventsPerShard >= the
        # run size. More than one means a different sharding was used; still
        # correct to concatenate, but worth flagging.
        logger.warning("%s/%s: %d shards (expected 1), concatenating", run_dir.name, obj, len(files))
    return pa.concat_tables([pq.read_table(f) for f in files])


def _slice_and_offset(table: pa.Table, local_start: int, local_stop: int, offset: int) -> pa.Table:
    """Keep local events [start, stop) and shift event_id into global numbering."""
    ev = table.column("event_id")
    mask = pc.and_(pc.greater_equal(ev, local_start), pc.less(ev, local_stop))
    table = table.filter(mask)
    if table.num_rows == 0:
        return table
    shifted = pc.add(table.column("event_id"), pa.scalar(offset, type=ev.type))
    return table.set_column(table.schema.get_field_index("event_id"), "event_id", shifted)


def package_chunk(
    *,
    start_event: int,
    end_event: int,
    start_run: int,
    start_local: int,
    end_run: int,
    end_local: int,
    run_dirs: list[Path],
    run_size: int,
    objects: list[str],
    out_base: Path,
    dataset_name_dot: str,
    row_group_size: int | None,
    compression: str,
) -> None:
    """Write one output file per object for the event window [start_event, end_event]."""
    t0 = time.time()
    expected = end_event - start_event + 1
    logger.info(
        "chunk events %d-%d (runs %d..%d), %d objects",
        start_event, end_event, start_run, end_run, len(objects),
    )

    for obj in objects:
        if obj not in OBJECT_LAYOUT:
            logger.error("unknown object %r - not in OBJECT_LAYOUT, skipping", obj)
            continue
        group, published_name = OBJECT_LAYOUT[obj]
        out_dir = out_base / group / published_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / (
            f"{dataset_name_dot}.{group}.{published_name}.events{start_event}-{end_event}.parquet"
        )

        writer: pq.ParquetWriter | None = None
        rows = 0
        events_seen = 0
        try:
            for abs_run in range(start_run, end_run + 1):
                run_dir = run_dirs[abs_run]
                local_start, local_stop = local_events_for_run(
                    start_run=start_run, start_local=start_local,
                    end_run=end_run, end_local=end_local,
                    abs_run=abs_run, run_size=run_size,
                )
                table = _read_run_table(run_dir, obj)
                if table is None:
                    logger.warning("run %d: no %s parquet, skipping", abs_run, obj)
                    continue
                table = _slice_and_offset(table, local_start, local_stop, abs_run * run_size)
                if table.num_rows == 0:
                    continue
                table = _convert_times(table, obj)
                if writer is None:
                    writer = pq.ParquetWriter(out_file, table.schema, compression=compression)
                # Stream run by run: peak memory is one run, not one chunk.
                writer.write_table(table, row_group_size=row_group_size)
                rows += table.num_rows
                events_seen += local_stop - local_start
        finally:
            if writer is not None:
                writer.close()

        if writer is None:
            logger.warning("%s: no data in this chunk, no file written", obj)
            continue
        # The event count is the load-bearing check: a short file here means a
        # run was missing or partially digitized, which must not pass silently.
        if events_seen != expected:
            logger.error(
                "%s: expected %d events in chunk, wrote %d - INCOMPLETE (%s)",
                obj, expected, events_seen, out_file.name,
            )
        logger.info("wrote %s (rows=%d, events=%d)", out_file.name, rows, events_seen)

    logger.info("chunk %d-%d done in %.1fs", start_event, end_event, time.time() - t0)


def discover_run_dirs(runs_dir: Path, max_run: int | None) -> list[Path]:
    """Return run dirs indexed by run number, asserting the numbering is dense.

    Global event numbering multiplies by the run's position in this list, so a
    gap here would silently offset every downstream event id.
    """
    numbered = sorted(
        (int(p.name), p) for p in runs_dir.iterdir() if p.is_dir() and p.name.isdigit()
    )
    if max_run is not None:
        numbered = [(n, p) for n, p in numbered if n <= max_run]
    if not numbered:
        raise SystemExit(f"no numbered run directories under {runs_dir}")
    for expected_idx, (n, _) in enumerate(numbered):
        if n != expected_idx:
            raise SystemExit(
                f"run directories are not dense from 0: position {expected_idx} holds run {n}. "
                "Global event numbering assumes run N sits at index N; refusing to guess."
            )
    return [p for _, p in numbered]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    parser.add_argument("--chunk-index", type=int, default=None,
                        help="Process exactly this chunk (for distributed runs)")
    parser.add_argument("--output", "-o", default=None, help="Override output_base_dir")
    parser.add_argument("--output-subdir", default=None, help="Ignored; accepted for run_stage compatibility")
    parser.add_argument("--seed", default=None, help="Ignored; accepted for run_stage compatibility")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    campaign = config["campaign"]
    dataset = config["dataset"]
    version = config["version"]
    run_size = int(config["run_size"])
    chunk_size = int(config["chunk_size"])
    objects = list(config.get("objects", list(OBJECT_LAYOUT)))
    row_group_size = config.get("row_group_size")
    compression = config.get("compression", "snappy")
    max_run = config.get("max_run")

    runs_dir = Path(config["input_base_dir"]) / campaign / dataset / version / "runs"
    dataset_base = f"{campaign}/{dataset}/{version}"
    dataset_name_dot = dataset_base.replace("/", ".")
    out_base = Path(args.output or config["output_base_dir"]) / dataset_base / "parquet"

    run_dirs = discover_run_dirs(runs_dir, max_run)
    logger.info(
        "packaging %s: %d runs x %d events, chunk_size=%d, objects=%s",
        dataset_name_dot, len(run_dirs), run_size, chunk_size, objects,
    )
    logger.info("input : %s", runs_dir)
    logger.info("output: %s", out_base)

    def process(start_event, end_event, start_run, start_local, end_run, end_local):
        package_chunk(
            start_event=start_event, end_event=end_event,
            start_run=start_run, start_local=start_local,
            end_run=end_run, end_local=end_local,
            run_dirs=run_dirs, run_size=run_size, objects=objects,
            out_base=out_base, dataset_name_dot=dataset_name_dot,
            row_group_size=row_group_size, compression=compression,
        )

    iterate_and_process_chunks(
        run_dirs=run_dirs,
        run_size=run_size,
        chunk_size=chunk_size,
        config=config,
        chunk_index=args.chunk_index,
        process_chunk_fn=process,
    )


if __name__ == "__main__":
    main()
