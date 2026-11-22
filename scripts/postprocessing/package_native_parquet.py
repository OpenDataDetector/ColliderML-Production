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
                                                /reco/calo_cells/      (Release 2, from reco_tables)
                                                /reco/calo_clusters/   (Release 2, from reco_tables)
                                                /reco/pfos/            (Release 2, from reco_tables)
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
import hashlib
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.driver import iterate_and_process_chunks, local_events_for_run  # noqa: E402
from utils.particle_id_map import ID_COLUMNS, RunParticleMap  # noqa: E402

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
    # Release-2 reco tables written per run by reco_tables (convert_reco_tables.py)
    # from the k4ODD output; same per-run layout, event_id restarting at 0.
    "calo_cells": ("reco", "calo_cells"),
    "calo_clusters": ("reco", "calo_clusters"),
    "pfos": ("reco", "pfos"),
    # Complete truth-particle table (every MCParticle + raw status words), written per run by
    # particles_full (convert_particles_full.py); published in place of the ACTS particle table.
    # Times are already ns (edm4hep), ids already the MCParticle index: no conversion, no remap.
    "particles_full": ("truth", "particles"),
    # Generator history only (beams, hard process, showers; particles_full with generator_particles),
    # same particle_id as truth/particles; ns and MCParticle ids already, no conversion, no remap.
    "generator_particles": ("truth", "generator_particles"),
    # AMVF primary vertices, from the per-run ROOT ntuple (see vertices_from_root).
    "vertices": ("reco", "vertices"),
}

# The ACTS-native writer emits time-like quantities in ACTS native units
# (lengths, c=1): a "time" is a path length in mm. The legacy convert_all
# datasets are in nanoseconds, and mm-as-ns is exactly the field bug a
# downstream user caught (pixel hit "271 ns" that was really 271 mm = 0.91 ns).
# Convert at packaging time so the published tables are in ns; per-run native
# intermediates stay untouched. Value is the unit power: 1 for times (/c),
# 2 for variances (/c^2). The k4ODD-derived tables (calo_cells etc.) are
# already in ns (edm4hep convention) and are deliberately absent here.
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


# --- Track covariance -------------------------------------------------------
# The ACTS Arrow track writer emits only the five perigee parameters; the fitted
# covariance lives in the ROOT track summary, which the digitization stage writes
# when `performance_metrics` (CKF) / `truth_tracking_root_summary` (truth) are on.
# We join it in here on (event, track_id): the ROOT writer stores
# `track_nr = track.index()` and the Arrow writer stores the same value as
# `track_id`, so this is a real key, not an ordering assumption (verified on the
# 2026-09-12 pilot: d0/phi/t agree to 0.0 between the two).
#
# Only the 5x5 spatial+momentum block is published. The time row and column are
# deliberately dropped: geometric digitisation produces no time measurement, so
# they carry the seeding prior and nothing else (measured: err_t is a constant
# 10 ns for every CKF track, 33 ps for every truth track).
#
# Units are ACTS native, matching the parameter columns already published:
# mm^2, mm*rad, rad^2, rad/GeV, 1/GeV^2. No time terms, so no ns conversion.
_COV_PARAMS = ("d0", "z0", "phi", "theta", "qop")
_COV_ACTS = {"d0": "eLOC0", "z0": "eLOC1", "phi": "ePHI", "theta": "eTHETA", "qop": "eQOP"}
COV_COLUMNS = [
    (f"cov_{a}_{b}", f"cov_{_COV_ACTS[a]}_{_COV_ACTS[b]}")
    for i, a in enumerate(_COV_PARAMS) for b in _COV_PARAMS[i:]
]
COV_SOURCE = {"tracks": "tracksummary_ambi.root", "truth_tracks": "tracksummary_truth.root"}

# Fit quality, joined from the same ROOT summary. Without these a consumer has NO
# way to reject a bad fit: measured on single_muon_100GeV, 4.0% of truth tracks
# have a parameter pull beyond 10 sigma and 0.26% have the wrong charge sign,
# with fully truth-pure hits and (for truth tracks) no outlier flags. chi2/ndf
# separates them cleanly - good tracks median 0.75, catastrophic median 33.0, and
# a chi2/ndf < 3 cut removes 92% of them for 4% of the good ones.
QUALITY_COLUMNS = [
    ("chi2", "chi2Sum", "f4"),
    ("ndf", "NDF", "i4"),
    ("n_measurements", "nMeasurements", "i4"),
    ("n_outliers", "nOutliers", "i4"),
    ("n_holes", "nHoles", "i4"),
    ("n_shared_hits", "nSharedHits", "i4"),
]


def _attach_covariance(table: pa.Table, run_dir: Path, obj: str, strict: bool = False) -> pa.Table:
    """Append the 15 upper-triangle covariance columns, joined on (event, track_id).
    strict (production, with chunk manifests): every track must have exactly one ROOT entry."""
    src = COV_SOURCE.get(obj)
    if src is None or "track_id" not in table.schema.names:
        return table
    path = run_dir / src
    if not path.exists():
        logger.warning("%s: %s missing, packaging %s without covariance", run_dir.name, src, obj)
        return table
    import uproot  # available in the stage container via setup_container_env.sh

    branches = (["event_nr", "track_nr"] + [b for _, b in COV_COLUMNS]
                + [b for _, b, _ in QUALITY_COLUMNS])
    tree = uproot.open(path)["tracksummary"]
    have = set(tree.keys())
    missing = [b for b in branches if b not in have]
    if missing:
        logger.warning("%s: %s lacks %s, packaging without covariance", run_dir.name, src, missing[:3])
        return table
    arr = tree.arrays(branches, library="np")

    def cat(key):
        v = arr[key]
        return np.concatenate([np.asarray(x) for x in v]) if v.dtype == object else np.asarray(v)

    n_per_event = np.array([len(x) for x in arr["track_nr"]])
    r_event = np.repeat(np.asarray(arr["event_nr"], dtype=np.int64), n_per_event)
    r_track = cat("track_nr").astype(np.int64)
    r_key = r_event * _TRACK_KEY_STRIDE + r_track

    tid = table.column("track_id").combine_chunks()
    offsets = tid.offsets.to_numpy()
    p_track = tid.values.to_numpy(zero_copy_only=False).astype(np.int64)
    p_event = np.repeat(table.column("event_id").to_numpy().astype(np.int64), np.diff(offsets))
    p_key = p_event * _TRACK_KEY_STRIDE + p_track

    if strict:
        uk, cnt = np.unique(r_key, return_counts=True)
        if (cnt > 1).any():
            raise RuntimeError(f"{run_dir.name}/{obj}: {int((cnt > 1).sum())} duplicate (event, track) keys in {src}")
    order = np.argsort(r_key)
    pos = np.searchsorted(r_key[order], p_key)
    pos = np.clip(pos, 0, len(order) - 1)
    idx = order[pos]
    ok = r_key[idx] == p_key
    if strict and not ok.all():
        raise RuntimeError(f"{run_dir.name}/{obj}: {int((~ok).sum())} of {len(p_key)} tracks without a {src} entry")
    if not ok.all():
        logger.warning(
            "%s/%s: %d of %d tracks have no covariance entry, filled with NaN",
            run_dir.name, obj, int((~ok).sum()), len(p_key),
        )

    off = pa.array(offsets, type=pa.int32())
    for name, branch in COV_COLUMNS:
        vals = cat(branch).astype(np.float32)[idx]
        vals[~ok] = np.nan
        table = table.append_column(
            name, pa.ListArray.from_arrays(off, pa.array(vals, type=pa.float32()))
        )
    for name, branch, dt in QUALITY_COLUMNS:
        raw = cat(branch)[idx]
        if dt == "f4":
            vals = raw.astype(np.float32); vals[~ok] = np.nan
            typ = pa.float32()
        else:
            vals = raw.astype(np.int32); vals[~ok] = -1
            typ = pa.int32()
        table = table.append_column(
            name, pa.ListArray.from_arrays(off, pa.array(vals, type=typ))
        )
    return table


_TRACK_KEY_STRIDE = 1 << 20  # track_id is uint16, so 2^20 is a safe stride


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def release_chunk_window(chunk_index: int, chunk_size: int, release_events: int) -> tuple[int, int]:
    """[start, end] of a release chunk; an index outside the release is an error, not a warning."""
    n_chunks = (release_events + chunk_size - 1) // chunk_size
    if not 0 <= chunk_index < n_chunks:
        raise SystemExit(f"chunk index {chunk_index} outside the release (0..{n_chunks - 1}, "
                         f"{release_events} events in chunks of {chunk_size})")
    start = chunk_index * chunk_size
    return start, min(release_events, start + chunk_size) - 1


# --- Shared hits --------------------------------------------------------------
# The ROOT summary's nSharedHits is 0 for every track in the pr-8 chain: nothing
# computes it (TrackFindingAlgorithm has no computeSharedHits step and the greedy
# ambiguity solver does not write the flag). Recompute it here from hit_ids, per
# event and per collection. Released definition (not claimed equal to ACTS
# computeSharedHits, which decorates states of its input tracks, including tracks
# later removed): the number of distinct non-outlier hit ids of a track that are
# also non-outlier hits of another track of the same final collection and event.
def compute_shared_hits(table: pa.Table) -> pa.Table:
    """Return ``table`` with ``n_shared_hits`` (list<int32>, one per track) recomputed."""
    if "hit_ids" not in table.column_names:
        raise RuntimeError("shared_hits_from_hit_ids: table has no hit_ids column")
    if "hit_outlier" not in table.column_names:
        raise RuntimeError("shared_hits_from_hit_ids: table has no hit_outlier column")
    hits = table.column("hit_ids").combine_chunks()        # list<list<uint32>> per event
    trk_off = hits.offsets.to_numpy()                       # event -> tracks
    per_trk = hits.flatten()                                # list<uint32> per track
    hit_off = per_trk.offsets.to_numpy()                    # track -> hits
    hit_val = per_trk.flatten().to_numpy(zero_copy_only=False).astype(np.int64)
    n_tracks = len(hit_off) - 1
    n_events = len(trk_off) - 1
    flags = table.column("hit_outlier").combine_chunks()
    flags_trk = flags.flatten()
    # both nesting levels must agree (a short list on one track and a long one on the next
    # would otherwise move outlier flags between tracks)
    if not (np.array_equal(flags.offsets.to_numpy(), trk_off)
            and np.array_equal(flags_trk.offsets.to_numpy(), hit_off)):
        raise RuntimeError("hit_outlier and hit_ids nesting differs (event or track level)")
    if (hits.null_count or per_trk.null_count or per_trk.flatten().null_count
            or flags.null_count or flags_trk.null_count or flags_trk.flatten().null_count):
        raise RuntimeError("null entries in hit_ids or hit_outlier")
    outl = flags_trk.flatten().to_numpy(zero_copy_only=False).astype(bool)

    trk_of_hit = np.repeat(np.arange(n_tracks, dtype=np.int64), np.diff(hit_off))
    ev_of_trk = np.repeat(np.arange(n_events, dtype=np.int64), np.diff(trk_off))
    ev_of_hit = ev_of_trk[trk_of_hit]
    meas = ~outl
    # unique (event, hit, track) among measurements: a hit counted once per track
    key = np.stack([ev_of_hit[meas], hit_val[meas], trk_of_hit[meas]], axis=1)
    key = np.unique(key, axis=0) if len(key) else key.reshape(0, 3)
    n_shared = np.zeros(n_tracks, dtype=np.int32)
    if len(key):
        eh = key[:, 0] * (1 << 32) + key[:, 1]               # hit ids are uint32
        _, inv, counts = np.unique(eh, return_inverse=True, return_counts=True)
        shared = counts[inv] >= 2
        np.add.at(n_shared, key[shared, 2], 1)
    col = pa.ListArray.from_arrays(pa.array(trk_off.astype(np.int32), type=pa.int32()),
                                   pa.array(n_shared, type=pa.int32()))
    if "n_shared_hits" in table.column_names:
        return table.set_column(table.schema.get_field_index("n_shared_hits"), "n_shared_hits", col)
    return table.append_column("n_shared_hits", col)


def sort_by_event_id(table: pa.Table, obj: str) -> pa.Table:
    """Sort outer rows (one per event) by event_id; refuse duplicated event ids."""
    ev = table.column("event_id").to_numpy().astype(np.int64)
    if len(np.unique(ev)) != len(ev):
        raise RuntimeError(f"{obj}: duplicated event_id rows in one run table")
    if len(ev) > 1 and np.all(ev[1:] > ev[:-1]):
        return table
    return table.take(pa.array(np.argsort(ev, kind="stable")))


# --- Vertices -------------------------------------------------------------------
# AMVF primary vertices (digi_and_reco `vertexing: true`). The pr-8 ACTS build has no
# Arrow vertex writer, so the per-run source is the ROOT ntuple of
# RootVertexNTupleWriter (performance_vertexing.root, tree "vertexing", one entry per
# event, per-vertex vectors). Published per event, one list entry per reco vertex:
#   vertex_id                 0..N-1 in AMVF output order
#   x, y, z                   fitted position [mm]
#   cov_xx ... cov_zz         spatial covariance [mm^2] (time row dropped: no time measurement)
#   n_tracks                  tracks attached to the vertex
#   sum_pt2                   sum of track pT^2 [GeV^2] (hard-scatter selection: max per event)
#   track_weight_sum          sum of AMVF track weights
#   truth_vertex_primary      truth vertex the reco vertex is matched to (VertexTruthMatcher,
#   truth_vertex_secondary    majority track weight); same numbering as truth/particles
#                             vertex_primary / vertex_secondary (hard scatter = 1)
#   truth_match_ratio         weight fraction from the matched truth vertex
#   contamination             1 - purity, as ACTS defines it
#   classification            0 unknown (no match), 1 clean, 2 merged, 3 split (ACTS enum)
# The ntuple's truthX/Y/Z are not published: for split vertices they can be the position of
# another SimVertex with the same (primary, secondary) barcode (measured on the v23 smoke test:
# a reco vertex at z = 196.07 mm matched to the hard scatter carried truthZ = 1622 mm). The
# interaction position is in truth/particles (vx, vy, vz of the primaries with that vertex_primary).
# Track links are NOT in the ntuple (per-vertex track parameters only, evaluated at the
# vertex); they need an ACTS writer change. Times (recoT, covTT) are not published.
VERTEX_SOURCE = "performance_vertexing.root"
_VERTEX_F32 = [
    ("x", "recoX"), ("y", "recoY"), ("z", "recoZ"),
    ("cov_xx", "covXX"), ("cov_xy", "covXY"), ("cov_xz", "covXZ"),
    ("cov_yy", "covYY"), ("cov_yz", "covYZ"), ("cov_zz", "covZZ"),
    ("sum_pt2", "sumPt2"), ("track_weight_sum", "recoVertexTrackWeights"),
    ("truth_match_ratio", "truthVertexMatchRatio"), ("contamination", "recoVertexContamination"),
]
_VERTEX_I32 = [
    ("n_tracks", "nTracksRecoVtx"),
    ("truth_vertex_primary", "vertex_primary"), ("truth_vertex_secondary", "vertex_secondary"),
]


def vertices_from_root(path: Path) -> pa.Table:
    import uproot  # available in the stage container via setup_container_env.sh

    branches = ["event_nr", "nRecoVtx", "recoVertexClassification"] + [b for _, b in _VERTEX_F32 + _VERTEX_I32]
    tree = uproot.open(path)["vertexing"]
    missing = [b for b in branches if b not in set(tree.keys())]
    if missing:
        raise RuntimeError(f"{path}: vertexing tree lacks {missing}")
    arr = tree.arrays(branches, library="np")
    n = np.array([len(x) for x in arr["recoX"]], dtype=np.int64)
    if not np.array_equal(n, np.asarray(arr["nRecoVtx"], dtype=np.int64)):
        raise RuntimeError(f"{path}: per-vertex vector length differs from nRecoVtx")
    off = pa.array(np.concatenate([[0], np.cumsum(n)]).astype(np.int32), type=pa.int32())

    def flat(b, dt):
        v = arr[b]
        lens = np.array([len(x) for x in v])
        if not np.array_equal(lens, n):
            raise RuntimeError(f"{path}: branch {b} length differs from nRecoVtx")
        return np.concatenate([np.asarray(x, dtype=dt) for x in v]) if len(v) else np.zeros(0, dt)

    cols = {"event_id": pa.array(np.asarray(arr["event_nr"], dtype=np.uint32), type=pa.uint32())}
    cols["vertex_id"] = pa.ListArray.from_arrays(
        off, pa.array(np.concatenate([np.arange(k) for k in n]).astype(np.uint16) if len(n) else [], type=pa.uint16()))
    for name, b in _VERTEX_F32:
        cols[name] = pa.ListArray.from_arrays(off, pa.array(flat(b, np.float32), type=pa.float32()))
    for name, b in _VERTEX_I32:
        cols[name] = pa.ListArray.from_arrays(off, pa.array(flat(b, np.int32), type=pa.int32()))
    cols["classification"] = pa.ListArray.from_arrays(
        off, pa.array(flat("recoVertexClassification", np.int8), type=pa.int8()))
    return pa.table(cols)


def _read_run_table(run_dir: Path, obj: str) -> pa.Table | None:
    """Read a run's single native parquet file for one object."""
    if obj == "vertices":
        src = run_dir / VERTEX_SOURCE
        return vertices_from_root(src) if src.exists() else None
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
    track_covariance: bool = False,
    remap_particle_ids: bool = False,
    sim_input_name: str = "edm4hep.root",
    sort_rows: bool = False,
    shared_hits_from_hit_ids: bool = False,
    manifest_path: Path | None = None,
) -> None:
    """Write one output file per object for the event window [start_event, end_event].

    With ``remap_particle_ids`` the ACTS-native particle ids (0..N-1 per event) are
    rewritten to the edm4hep MCParticle index, the id every other ColliderML table
    uses (see utils/particle_id_map.py). The map is built per run from the run's
    native particles table and ``<run>/<sim_input_name>``."""
    t0 = time.time()
    expected = end_event - start_event + 1
    logger.info(
        "chunk events %d-%d (runs %d..%d), %d objects",
        start_event, end_event, start_run, end_run, len(objects),
    )

    incomplete: list[str] = []
    particle_maps: dict[Path, RunParticleMap] = {}
    inputs: dict[str, dict] = {}     # every file this chunk read (v23 chunk manifest)
    outputs: list[Path] = []

    def note_inputs(run_dir: Path, obj: str) -> None:
        if manifest_path is None:   # no manifest requested: no hashing
            return
        files = [run_dir / VERTEX_SOURCE] if obj == "vertices" else sorted((run_dir / obj).glob("*.parquet"))
        if track_covariance and obj in COV_SOURCE:
            files.append(run_dir / COV_SOURCE[obj])
        if remap_particle_ids and obj in ID_COLUMNS:
            files += sorted((run_dir / "particles").glob("*.parquet")) + [run_dir / sim_input_name]
        for f in files:
            if str(f) in inputs or not f.exists():
                continue
            real = Path(os.path.realpath(f))
            st = real.stat()
            rec = {"realpath": str(real), "size": st.st_size, "mtime_ns": st.st_mtime_ns, "inode": st.st_ino}
            # the Geant4 input is frozen in the parent baseline (12 GB per PU200 run): stat only
            if f.name != sim_input_name:
                rec["sha256"] = _sha256(real)
            inputs[str(f)] = rec

    def particle_map(run_dir: Path) -> RunParticleMap:
        if run_dir not in particle_maps:
            # Objects are the outer loop, so keep every run of this chunk: rebuilding per
            # object re-read ~13M PU200 MCParticles 7x per run. A map is ~8 bytes/particle.
            if len(particle_maps) >= 8:
                particle_maps.clear()
            native = _read_run_table(run_dir, "particles")
            if native is None:
                raise RuntimeError(f"{run_dir}: no native particles table; cannot remap particle ids")
            particle_maps[run_dir] = RunParticleMap(native, run_dir / sim_input_name)
        return particle_maps[run_dir]

    for obj in objects:
        if obj not in OBJECT_LAYOUT:
            if manifest_path is not None:
                raise RuntimeError(f"unknown object {obj!r}")
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
                note_inputs(run_dir, obj)
                table = _read_run_table(run_dir, obj)
                if table is None:
                    logger.warning("run %d: no %s parquet, skipping", abs_run, obj)
                    continue
                if track_covariance:
                    # Join BEFORE the event_id shift, while event_id is still the
                    # run-local number the ROOT summary uses.
                    n_cols = table.num_columns
                    table = _attach_covariance(table, run_dir, obj, strict=manifest_path is not None)
                    if manifest_path is not None and obj in COV_SOURCE and table.num_columns == n_cols:
                        raise RuntimeError(f"run {abs_run}: {obj} covariance join failed (see warnings)")
                if remap_particle_ids and obj in ID_COLUMNS:
                    # Also before the shift: the map is keyed by run-local event.
                    table = particle_map(run_dir).remap_table(table, obj)
                if shared_hits_from_hit_ids and obj in ("tracks", "truth_tracks"):
                    # After the covariance join, which brings the all-zero ROOT nSharedHits.
                    table = compute_shared_hits(table)
                table = _slice_and_offset(table, local_start, local_stop, abs_run * run_size)
                if table.num_rows == 0:
                    continue
                if sort_rows:
                    # Runs are written in ascending order and event_id = run*run_size + local,
                    # so sorting each run's slice makes the whole file ascending.
                    table = sort_by_event_id(table, obj)
                table = _convert_times(table, obj)
                if writer is None:
                    writer = pq.ParquetWriter(out_file, table.schema, compression=compression)
                # Stream run by run: peak memory is one run, not one chunk.
                writer.write_table(table, row_group_size=row_group_size)
                rows += table.num_rows
                # Count the events actually present, not the requested window (Codex audit
                # 2026-10-01, finding 4: a truncated run used to pass as complete).
                events_seen += len(np.unique(table.column("event_id").to_numpy()))
        finally:
            if writer is not None:
                writer.close()

        if writer is None:
            logger.error("%s: no data in this chunk, no file written", obj)
            incomplete.append(obj)
            continue
        outputs.append(out_file)
        # The event count is the load-bearing check: a short file here means a
        # run was missing or partially digitized, which must not pass silently.
        if events_seen != expected:
            logger.error(
                "%s: expected %d events in chunk, wrote %d - INCOMPLETE (%s)",
                obj, expected, events_seen, out_file.name,
            )
            incomplete.append(obj)
        logger.info("wrote %s (rows=%d, events=%d)", out_file.name, rows, events_seen)

    if manifest_path is not None:
        import json
        man = {"start_event": start_event, "end_event": end_event, "runs": [start_run, end_run],
               "objects": objects, "status": "incomplete" if incomplete else "ok", "incomplete": incomplete,
               "inputs": inputs,
               "outputs": {str(f): {"size": f.stat().st_size, "sha256": _sha256(f)} for f in outputs}}
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
        os.replace(tmp, manifest_path)
    if incomplete:
        raise RuntimeError(f"chunk {start_event}-{end_event}: incomplete objects {incomplete}")
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

    track_covariance = bool(config.get("track_covariance", False))
    remap_particle_ids = bool(config.get("remap_particle_ids", False))
    sim_input_name = config.get("sim_input_name", "edm4hep.root")
    sort_rows = bool(config.get("sort_by_event_id", False))
    shared_hits_from_hit_ids = bool(config.get("shared_hits_from_hit_ids", False))
    if shared_hits_from_hit_ids and not track_covariance:
        # Without the ROOT join there is no n_shared_hits column to replace; the
        # computed one is appended, which changes the schema. Allowed, but say so.
        logger.warning("shared_hits_from_hit_ids without track_covariance: n_shared_hits is appended")
    logger.info("sort_by_event_id=%s shared_hits_from_hit_ids=%s", sort_rows, shared_hits_from_hit_ids)

    release_events = config.get("release_events")
    version_dir = out_base.parent

    def manifest_for(start_event):
        if not config.get("chunk_manifest", bool(release_events)):
            return None
        return version_dir / "provenance" / "package_parquet" / f"chunk{start_event // chunk_size}_inputs.json"

    def process(start_event, end_event, start_run, start_local, end_run, end_local):
        package_chunk(
            start_event=start_event, end_event=end_event,
            start_run=start_run, start_local=start_local,
            end_run=end_run, end_local=end_local,
            run_dirs=run_dirs, run_size=run_size, objects=objects,
            out_base=out_base, dataset_name_dot=dataset_name_dot,
            row_group_size=row_group_size, compression=compression,
            track_covariance=track_covariance,
            remap_particle_ids=remap_particle_ids,
            sim_input_name=sim_input_name,
            sort_rows=sort_rows,
            shared_hits_from_hit_ids=shared_hits_from_hit_ids,
            manifest_path=manifest_for(start_event),
        )

    if release_events:
        # The release, not the discovered run dirs, defines the chunks (v23: PU200 0-99,999 in
        # 1000 chunks; run 1562 contributes entries 0-31 only). Every chunk needs all its input runs.
        release_events = int(release_events)
        if len(run_dirs) * run_size < release_events:
            raise SystemExit(f"{len(run_dirs)} runs x {run_size} < release_events {release_events}")
        indices = [args.chunk_index] if args.chunk_index is not None else \
            range((release_events + chunk_size - 1) // chunk_size)
        for ci in indices:
            start_event, end_event = release_chunk_window(ci, chunk_size, release_events)
            process(start_event, end_event, start_event // run_size, start_event % run_size,
                    end_event // run_size, end_event % run_size)
        return

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
