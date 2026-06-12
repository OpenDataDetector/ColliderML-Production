#!/usr/bin/env python3
"""
Convert DIGITISED calorimeter cells (k4ODD DDCaloDigi output) to Parquet.

Reads the Pandora-reco / calo-digi EDM4hep file (digi{ECal,HCal}{Barrel,Endcap}Collection
CalorimeterHits + their digiLinkCaloHit* CalorimeterHit<->SimCalorimeterHit links) and writes
one event-level Parquet row per event with list-valued cell columns plus nested truth
contributions resolved through the digi->sim links (SimCalorimeterHit contributions ->
MCParticle row index, matching the particle_id convention of the truth-particles table).

Complements convert_calorimeter.py (SIM cells): this is the detector-realistic cell view the
particle-flow reconstruction actually consumed.

Usage:
  convert_calo_digi.py --input reco_edm4hep.root --output calo_cells.parquet [--max-events N]
  convert_calo_digi.py --config convert_calo_digi.yaml   (run-directory driven, like convert_all)
"""

import argparse
import logging
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import uproot

from utils.detector_enums import CALO_DETECTOR_CODES
from utils.parquet_schemas import CALO_CELLS_PARQUET_TYPES

logger = logging.getLogger(__name__)

# (digi collection, sim collection, link collection, ecal/hcal, barrel?)
DIGI_COLLECTIONS = [
    ("digiECalBarrelCollection", "ECalBarrelCollection", "digiLinkCaloHitECALBarrel", "ecal", True),
    ("digiECalEndcapCollection", "ECalEndcapCollection", "digiLinkCaloHitECALEndcap", "ecal", False),
    ("digiHCalBarrelCollection", "HCalBarrelCollection", "digiLinkCaloHitHCALBarrel", "hcal", True),
    ("digiHCalEndcapCollection", "HCalEndcapCollection", "digiLinkCaloHitHCALEndcap", "hcal", False),
]


def detector_code(system: str, barrel: bool, z: np.ndarray) -> np.ndarray:
    if barrel:
        return np.full(len(z), CALO_DETECTOR_CODES[f"{system}_barrel"], dtype=np.uint8)
    codes = np.where(z >= 0,
                     CALO_DETECTOR_CODES[f"{system}_pos_endcap"],
                     CALO_DETECTOR_CODES[f"{system}_neg_endcap"])
    return codes.astype(np.uint8)


def ranges_to_flat_indices(begin: np.ndarray, end: np.ndarray):
    """Vectorized [b0:e0)+[b1:e1)+... -> (flat_indices, counts)."""
    counts = (end - begin).astype(np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64), counts
    # standard repeat/cumsum trick
    starts = np.repeat(begin.astype(np.int64), counts)
    offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    return starts + offsets, counts


def process_events(tree, entry_start, entry_stop):
    """Return a list of per-event row dicts (event_id filled by caller)."""
    branches = {}
    for digi, sim, link, _, _ in DIGI_COLLECTIONS:
        for b in (f"{digi}/{digi}.cellID", f"{digi}/{digi}.energy", f"{digi}/{digi}.time",
                  f"{digi}/{digi}.position.x", f"{digi}/{digi}.position.y", f"{digi}/{digi}.position.z",
                  f"_{link}_from/_{link}_from.index", f"_{link}_to/_{link}_to.index",
                  f"{sim}/{sim}.contributions_begin", f"{sim}/{sim}.contributions_end",
                  f"{sim}Contributions/{sim}Contributions.energy",
                  f"_{sim}Contributions_particle/_{sim}Contributions_particle.index"):
            branches[b] = None
    arrays = tree.arrays(list(branches), entry_start=entry_start, entry_stop=entry_stop)

    n_events = len(arrays[f"{DIGI_COLLECTIONS[0][0]}/{DIGI_COLLECTIONS[0][0]}.cellID"])
    rows = []
    for ie in range(n_events):
        cell_id, energy, x, y, z, t, det = [], [], [], [], [], [], []
        c_pid, c_en = [], []
        for digi, sim, link, system, barrel in DIGI_COLLECTIONS:
            g = lambda name, coll: np.asarray(arrays[f"{coll}/{coll}.{name}"][ie])
            gl = lambda side: np.asarray(arrays[f"_{link}_{side}/_{link}_{side}.index"][ie])
            de = g("energy", digi)
            if len(de) == 0:
                continue
            dz = g("position.z", digi).astype(np.float32)
            cell_id.append(g("cellID", digi).astype(np.uint64))
            energy.append(de.astype(np.float32))
            x.append(g("position.x", digi).astype(np.float32))
            y.append(g("position.y", digi).astype(np.float32))
            z.append(dz)
            t.append(g("time", digi).astype(np.float32))
            det.append(detector_code(system, barrel, dz))

            # digi -> sim hit index (one link per digi hit; default -1 = no link)
            sim_for_digi = np.full(len(de), -1, dtype=np.int64)
            sim_for_digi[gl("from")] = gl("to")

            cb = g("contributions_begin", sim).astype(np.int64)
            ce = g("contributions_end", sim).astype(np.int64)
            ce_d = np.where(sim_for_digi >= 0, ce[sim_for_digi], 0)
            cb_d = np.where(sim_for_digi >= 0, cb[sim_for_digi], 0)
            flat_idx, counts = ranges_to_flat_indices(cb_d, ce_d)
            contrib_en = np.asarray(arrays[f"{sim}Contributions/{sim}Contributions.energy"][ie])
            contrib_pid = np.asarray(
                arrays[f"_{sim}Contributions_particle/_{sim}Contributions_particle.index"][ie])
            c_pid.append(ak.unflatten(contrib_pid[flat_idx].astype(np.uint64), counts))
            c_en.append(ak.unflatten(contrib_en[flat_idx].astype(np.float32), counts))

        cat = lambda parts, dt: np.concatenate(parts).astype(dt) if parts else np.empty(0, dtype=dt)
        rows.append({
            "detector": cat(det, np.uint8),
            "cell_id": cat(cell_id, np.uint64),
            "energy": cat(energy, np.float32),
            "x": cat(x, np.float32), "y": cat(y, np.float32), "z": cat(z, np.float32),
            "time": cat(t, np.float32),
            "contrib_particle_ids": ak.concatenate(c_pid).to_list() if c_pid else [],
            "contrib_energies": ak.concatenate(c_en).to_list() if c_en else [],
        })
    return rows


def convert_file(input_path: Path, output_path: Path, max_events: int = -1,
                 event_id_offset: int = 0, chunk: int = 200) -> int:
    tree = uproot.open(input_path)["events"]
    n = tree.num_entries if max_events < 0 else min(max_events, tree.num_entries)
    schema = pa.schema(CALO_CELLS_PARQUET_TYPES)
    writer = pq.ParquetWriter(output_path, schema, compression="zstd")
    written = 0
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        rows = process_events(tree, start, stop)
        table = {k: [] for k in CALO_CELLS_PARQUET_TYPES}
        for i, r in enumerate(rows):
            table["event_id"].append(np.uint32(event_id_offset + start + i))
            for k, v in r.items():
                table[k].append(v.tolist() if isinstance(v, np.ndarray) else v)
        writer.write_table(pa.table(table, schema=schema))
        written += len(rows)
        logger.info("calo_digi: %s events %d-%d", input_path.name, start, stop - 1)
    writer.close()
    return written


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, help="reco/calo-digi edm4hep file")
    ap.add_argument("--output", type=Path, help="output parquet path")
    ap.add_argument("--max-events", type=int, default=-1)
    ap.add_argument("--event-id-offset", type=int, default=0)
    args = ap.parse_args()
    if not args.input or not args.output:
        ap.error("--input and --output are required")
    n = convert_file(args.input, args.output, args.max_events, args.event_id_offset)
    logger.info("wrote %s: %d events", args.output, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
