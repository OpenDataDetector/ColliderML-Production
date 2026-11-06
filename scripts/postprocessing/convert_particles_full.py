#!/usr/bin/env python3
"""Per-run stage: the complete truth-particle table (every edm4hep MCParticle).

Release 1 `truth/particles` dropped the generator history (generatorStatus 4 and 21-99: beams,
hard process, showers, hadronisation inputs) and kept only the `primary` flag. This table keeps
every MCParticle of the event, so any hit or track can be traced through `parent_ids` to the
hard process, and it carries the raw status words:

    <runs>/<N>/particles_full/particles_full_000000-<n>.parquet

One row per event (list columns per particle). `particle_id` is the MCParticles index, the same id
every other ColliderML table uses, so this is a superset of v1 `truth/particles`: the v1 rows are
the particles with generator_status in (0, 1, 2) (verified when the stage was written; see
tests/regression/test_particles_full.py).

Columns kept from v1, same meaning and type: particle_id, pdg_id, mass, energy, charge, vx, vy, vz,
time, px, py, pz, perigee_d0, perigee_z0, vertex_primary, parent_id (first parent, -1 if none),
primary (= not created in simulation, simulatorStatus bit 30).
New: generator_status, simulator_status (raw edm4hep words), parent_ids and daughter_ids (all links),
endpoint_x/y/z, endpoint_px/py/pz, color_a, color_b.

perigee_d0/perigee_z0/vertex_primary come from the run's ACTS outputs (vertex_primary from the native
particle table, perigee from particles.root; the Arrow writer leaves perigee null), matched to MCParticles
with utils/particle_id_map. Optional `parent_particles_dir` (+ run_size) copies them from a Release 1
truth/particles table instead. Generator-history rows (not in ACTS) get NaN / 0. event_id restarts at 0 per run (= position in edm4hep.root);
package_native_parquet adds run * run_size.

Stage config keys: sim_input_name (default edm4hep.root), native_particles (default particles),
parent_particles_dir + run_size (copy perigee/vertex_primary from v1), events (-1 = all),
generator_particles (off | all | hard_scatter, default off), particles_full_table (default true; false
writes only generator_particles, e.g. next to a Release 1 or v21 particles table that stays published).

generator_particles writes a second table with only the generator history (generator_status not in
0, 1, 2: beams, hard process, showers, hadronisation inputs), for publishing it next to a particles table
that keeps the Release 1 rows:

    <runs>/<N>/generator_particles/generator_particles_000000-<n>.parquet

Same particle_id (MCParticles index), so parent_ids / daughter_ids cross between the two tables. The
`interaction` column numbers the pp interactions of the event (connected parts of the parent/daughter
graph, ordered by their lowest index); the hard scatter is the interaction of MCParticle 0, which comes
first in the merged generator record. `hard_scatter` keeps only that interaction.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pyarrow as pa
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import pyarrow.parquet as pq
import uproot
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.particle_id_map import RunParticleMap  # noqa: E402

logger = logging.getLogger("particles_full")
OBJ = "particles_full"
M = "MCParticles"
FLOAT_BRANCHES = {
    "mass": "mass", "charge": "charge", "time": "time",
    "vx": "vertex.x", "vy": "vertex.y", "vz": "vertex.z",
    "px": "momentum.x", "py": "momentum.y", "pz": "momentum.z",
    "endpoint_x": "endpoint.x", "endpoint_y": "endpoint.y", "endpoint_z": "endpoint.z",
    "endpoint_px": "momentumAtEndpoint.x", "endpoint_py": "momentumAtEndpoint.y", "endpoint_pz": "momentumAtEndpoint.z",
}
INT_BRANCHES = {"generator_status": "generatorStatus", "simulator_status": "simulatorStatus",
                "color_a": "colorFlow.a", "color_b": "colorFlow.b"}
CREATED_IN_SIMULATION_BIT = 30
RELEASE1_STATUS = (0, 1, 2)  # Geant4 particles, final state, decayed in the generator
GEN = "generator_particles"
GEN_COLUMNS = ["event_id", "particle_id", "interaction", "pdg_id", "generator_status", "mass", "energy", "charge",
               "vx", "vy", "vz", "time", "px", "py", "pz", "endpoint_x", "endpoint_y", "endpoint_z",
               "color_a", "color_b", "parent_ids", "daughter_ids"]
GEN_TYPES = {"event_id": pa.uint32(), "particle_id": pa.list_(pa.uint64()), "interaction": pa.list_(pa.uint16()),
             "pdg_id": pa.list_(pa.int64()), "generator_status": pa.list_(pa.int32()),
             "color_a": pa.list_(pa.int32()), "color_b": pa.list_(pa.int32()),
             "parent_ids": pa.list_(pa.list_(pa.int64())), "daughter_ids": pa.list_(pa.list_(pa.int64()))}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--output", "-o", type=Path, required=True, help="runs directory")
    p.add_argument("--output-subdir", type=str, default=None, help="run number")
    p.add_argument("--events", "-n", type=int, default=None)
    p.add_argument("--sim-input-name", default=None)
    p.add_argument("--native-particles", default=None)
    p.add_argument("--parent-particles-dir", default=None)
    p.add_argument("--run-size", type=int, default=None)
    p.add_argument("--generator-particles", choices=["off", "all", "hard_scatter"], default=None)
    p.add_argument("--seed", default=None)  # run_stage compatibility, unused
    p.add_argument("--performance-metrics", action="store_true", default=None)
    return p


def load_config(args):
    if args.config is not None:
        cfg = yaml.safe_load(open(args.config)) or {}
        for k, v in cfg.items():
            if getattr(args, k, None) is None:
                setattr(args, k, v)
    args.events = -1 if args.events is None else int(args.events)
    args.sim_input_name = args.sim_input_name or "edm4hep.root"
    args.native_particles = args.native_particles or "particles"
    args.generator_particles = args.generator_particles or "off"
    if args.generator_particles not in ("off", "all", "hard_scatter"):
        raise ValueError(f"generator_particles must be off, all or hard_scatter, not {args.generator_particles!r}")
    args.particles_full_table = True if getattr(args, "particles_full_table", None) is None else bool(args.particles_full_table)
    if not args.particles_full_table and args.generator_particles == "off":
        raise ValueError("particles_full_table: false needs generator_particles (nothing to write)")
    return args


def _links(begin, end, index):
    """Per-particle lists of linked MCParticle indices (podio OneToMany relation)."""
    return [index[b:e].astype(np.int64).tolist() for b, e in zip(begin, end)]


def interactions(parents: list[list[int]], n: int) -> np.ndarray:
    """Interaction number per particle: connected parts of the parent/daughter graph, numbered in the
    order of their lowest particle index (so the interaction of particle 0 is 0)."""
    src = np.fromiter((i for i, ps in enumerate(parents) for _ in ps), dtype=np.int64)
    dst = np.fromiter((p for ps in parents for p in ps), dtype=np.int64)
    graph = coo_matrix((np.ones(len(src), np.int8), (src, dst)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    first = np.full(labels.max() + 1 if n else 0, n, dtype=np.int64)
    np.minimum.at(first, labels, np.arange(n))
    order = np.empty_like(first)
    order[np.argsort(first)] = np.arange(len(first))
    return order[labels].astype(np.uint16)


def generator_rows(gen_status: np.ndarray, inter: np.ndarray, scope: str) -> np.ndarray:
    """Indices of the generator-history particles to keep (scope all or hard_scatter)."""
    keep = ~np.isin(gen_status, RELEASE1_STATUS)
    if scope == "hard_scatter":
        keep &= inter == 0
    return np.flatnonzero(keep)


def acts_columns(run_dir: Path, native_dir: str, sim_file: Path):
    """{event: (mc_index array, perigee_d0, perigee_z0, vertex_primary)} from the run's ACTS outputs.

    vertex_primary comes from the ACTS-native particle table. perigee_d0/z0 come from the ACTS ROOT
    particle file (particles.root): the Arrow particle converter of the current ACTS build writes them
    as nulls (see scripts/simulation/_arrow_writers.py). particles.root rows are in the same order as the
    native table; that is checked per event (PDG codes), and the values were checked against Release 1
    (33,165 of 33,166 v1 particles identical in run 0 events 0-7; the one is a particle ACTS does not hold).
    """
    files = sorted((run_dir / native_dir).glob("*.parquet"))
    if not files:
        logger.warning("no ACTS particle table in %s: perigee/vertex_primary left empty", run_dir / native_dir)
        return {}
    native = pa.concat_tables([pq.read_table(f) for f in files])
    pmap = RunParticleMap(native, sim_file)
    d = native.select(["event_id", "pdg_id", "perigee_d0", "perigee_z0", "vertex_primary"]).to_pydict()
    root_file = run_dir / "particles.root"
    root = None
    if root_file.exists():
        t = uproot.open(root_file)["particles"]
        r = t.arrays(["event_id", "particle_type", "perigee_d0", "perigee_z0"], library="np")
        root = {int(e): i for i, e in enumerate(r["event_id"])}
    else:
        logger.warning("%s missing: perigee_d0/z0 left NaN", root_file)
    out = {}
    for e, pdg, d0, z0, vp in zip(d["event_id"], d["pdg_id"], d["perigee_d0"], d["perigee_z0"], d["vertex_primary"]):
        d0 = np.asarray(d0, np.float32); z0 = np.asarray(z0, np.float32)
        if root is not None:
            i = root.get(int(e))
            if i is None or not np.array_equal(np.asarray(r["particle_type"][i]), np.asarray(pdg)):
                raise RuntimeError(f"event {e}: particles.root rows do not line up with the native particle table")
            d0 = np.asarray(r["perigee_d0"][i], np.float32); z0 = np.asarray(r["perigee_z0"][i], np.float32)
        out[int(e)] = (pmap.maps[int(e)].astype(np.int64), d0, z0, np.asarray(vp, np.uint16))
    return out


def parent_columns(parent_dir: str, run: int, run_size: int, n_events: int):
    """{local event: (particle_id, perigee_d0, perigee_z0, vertex_primary)} from a v1 particles table."""
    lo, hi = run * run_size, run * run_size + n_events
    files = sorted(Path(parent_dir).glob("*.parquet"))
    keep = []
    for f in files:  # file names carry their event range: ...events<A>-<B>.parquet
        a, b = f.name.rsplit(".events", 1)[1].split(".parquet")[0].split("-")
        if int(a) < hi and int(b) >= lo:
            keep.append(f)
    if not keep:
        raise FileNotFoundError(f"no parent particle files cover events {lo}-{hi - 1} in {parent_dir}")
    t = pq.read_table(keep, columns=["event_id", "particle_id", "perigee_d0", "perigee_z0", "vertex_primary"],
                      filters=[("event_id", ">=", lo), ("event_id", "<", hi)]).to_pydict()
    return {int(e) - lo: (np.asarray(i, np.int64), np.asarray(d, np.float32), np.asarray(z, np.float32), np.asarray(v, np.uint16))
            for e, i, d, z, v in zip(t["event_id"], t["particle_id"], t["perigee_d0"], t["perigee_z0"], t["vertex_primary"])}


def convert_run(run_dir: Path, args) -> tuple[Path, int]:
    sim_file = run_dir / args.sim_input_name
    if not sim_file.exists():
        raise FileNotFoundError(sim_file)
    tree = uproot.open(sim_file)["events"]
    stop = None if args.events < 0 else args.events
    br = ([f"{M}/{M}.PDG"] + [f"{M}/{M}.{b}" for b in FLOAT_BRANCHES.values()] + [f"{M}/{M}.{b}" for b in INT_BRANCHES.values()]
          + [f"{M}/{M}.parents_begin", f"{M}/{M}.parents_end", f"_{M}_parents/_{M}_parents.index",
             f"{M}/{M}.daughters_begin", f"{M}/{M}.daughters_end", f"_{M}_daughters/_{M}_daughters.index"])
    a = tree.arrays(br, entry_stop=stop, library="np")
    n_events = len(a[f"{M}/{M}.PDG"])
    if not args.particles_full_table:
        acts = {}                                   # generator_particles only: no ACTS columns needed
    elif args.parent_particles_dir:
        if not args.run_size:
            raise ValueError("parent_particles_dir needs run_size")
        acts = parent_columns(args.parent_particles_dir, int(run_dir.name), int(args.run_size), n_events)
        logger.info("perigee/vertex_primary copied from %s", args.parent_particles_dir)
    else:
        acts = acts_columns(run_dir, args.native_particles, sim_file)

    cols: dict[str, list] = {k: [] for k in ["event_id", "particle_id", "pdg_id", "mass", "energy", "charge", "vx", "vy", "vz",
                                              "time", "px", "py", "pz", "perigee_d0", "perigee_z0", "vertex_primary",
                                              "parent_id", "primary", *INT_BRANCHES, "parent_ids", "daughter_ids",
                                              "endpoint_x", "endpoint_y", "endpoint_z", "endpoint_px", "endpoint_py", "endpoint_pz"]}
    gen = {k: [] for k in GEN_COLUMNS} if args.generator_particles != "off" else None
    n_acts_missing = 0
    for e in range(n_events):
        g = lambda name: a[f"{M}/{M}.{name}"][e]
        n = len(g("PDG"))
        f = {k: np.asarray(g(b), dtype=np.float32) for k, b in FLOAT_BRANCHES.items()}
        ints = {k: np.asarray(g(b), dtype=np.int32) for k, b in INT_BRANCHES.items()}
        parents = _links(g("parents_begin"), g("parents_end"), a[f"_{M}_parents/_{M}_parents.index"][e])
        daughters = _links(g("daughters_begin"), g("daughters_end"), a[f"_{M}_daughters/_{M}_daughters.index"][e])
        p2 = f["px"].astype(np.float64) ** 2 + f["py"].astype(np.float64) ** 2 + f["pz"].astype(np.float64) ** 2
        energy = np.sqrt(p2 + f["mass"].astype(np.float64) ** 2).astype(np.float32)
        d0 = np.full(n, np.nan, np.float32); z0 = np.full(n, np.nan, np.float32); vp = np.zeros(n, np.uint16)
        if e in acts:
            idx, ad0, az0, avp = acts[e]
            d0[idx], z0[idx], vp[idx] = ad0, az0, avp
        elif acts:
            n_acts_missing += 1
        cols["event_id"].append(e)
        cols["particle_id"].append(np.arange(n, dtype=np.uint64))
        cols["pdg_id"].append(np.asarray(g("PDG"), dtype=np.int64))
        for k in ("mass", "charge", "vx", "vy", "vz", "time", "px", "py", "pz",
                  "endpoint_x", "endpoint_y", "endpoint_z", "endpoint_px", "endpoint_py", "endpoint_pz"):
            cols[k].append(f[k])
        cols["energy"].append(energy)
        cols["perigee_d0"].append(d0); cols["perigee_z0"].append(z0); cols["vertex_primary"].append(vp)
        cols["parent_id"].append(np.array([p[0] if p else -1 for p in parents], dtype=np.int64))
        cols["primary"].append(((ints["simulator_status"].astype(np.int64) >> CREATED_IN_SIMULATION_BIT) & 1) == 0)
        for k in INT_BRANCHES:
            cols[k].append(ints[k])
        cols["parent_ids"].append(parents); cols["daughter_ids"].append(daughters)
        if gen is not None:
            inter = interactions(parents, n)
            rows = generator_rows(ints["generator_status"], inter, args.generator_particles)
            gen["event_id"].append(e)
            gen["particle_id"].append(rows.astype(np.uint64))
            gen["interaction"].append(inter[rows])
            gen["pdg_id"].append(np.asarray(g("PDG"), dtype=np.int64)[rows])
            gen["energy"].append(energy[rows])
            for k in ("mass", "charge", "vx", "vy", "vz", "time", "px", "py", "pz", "endpoint_x", "endpoint_y", "endpoint_z"):
                gen[k].append(f[k][rows])
            for k in ("generator_status", "color_a", "color_b"):
                gen[k].append(ints[k][rows])
            gen["parent_ids"].append([parents[i] for i in rows])
            gen["daughter_ids"].append([daughters[i] for i in rows])
    if n_acts_missing:
        raise RuntimeError(f"{n_acts_missing} events missing from the perigee/vertex_primary source")

    types = {"event_id": pa.uint32(), "particle_id": pa.list_(pa.uint64()), "pdg_id": pa.list_(pa.int64()),
             "vertex_primary": pa.list_(pa.uint16()), "parent_id": pa.list_(pa.int64()), "primary": pa.list_(pa.bool_()),
             "generator_status": pa.list_(pa.int32()), "simulator_status": pa.list_(pa.int32()),
             "color_a": pa.list_(pa.int32()), "color_b": pa.list_(pa.int32()),
             "parent_ids": pa.list_(pa.list_(pa.int64())), "daughter_ids": pa.list_(pa.list_(pa.int64()))}
    final = None
    if args.particles_full_table:
        arrays = {k: pa.array(v, type=types.get(k, pa.list_(pa.float32()))) for k, v in cols.items()}
        final = write_table(pa.table(arrays), run_dir, OBJ, n_events)
    if gen is not None:
        gtable = pa.table({k: pa.array(v, type=GEN_TYPES.get(k, pa.list_(pa.float32()))) for k, v in gen.items()})
        gfinal = write_table(gtable, run_dir, GEN, n_events)
        logger.info("%s (%s): %d rows", gfinal, args.generator_particles, sum(len(r) for r in gen["particle_id"]))
        final = final or gfinal
    return final, n_events


def write_table(table: pa.Table, run_dir: Path, obj: str, n_events: int) -> Path:
    out_dir = run_dir / obj
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f".{obj}.partial.parquet"
    pq.write_table(table, tmp)
    final = out_dir / f"{obj}_000000-{n_events:06d}.parquet"
    os.replace(tmp, final)
    return final


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)-14s %(message)s")
    args = load_config(build_parser().parse_args())
    run_dir = args.output / str(args.output_subdir) if args.output_subdir else args.output
    try:
        path, n = convert_run(run_dir, args)
    except Exception as exc:  # noqa: BLE001 - stage boundary
        logger.error("particles_full failed: %s\n%s", exc, traceback.format_exc())
        return 1
    logger.info("✓ %s: %d events, %.1f MB", path, n, path.stat().st_size / 1024**2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
