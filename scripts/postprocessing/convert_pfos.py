#!/usr/bin/env python3
"""
Convert Pandora particle-flow output (PFOs + clusters) to Parquet.

Reads GaudiPandoraPFOs / GaudiPandoraClusters from the k4ODD Pandora reco EDM4hep file and
writes two event-level Parquet tables:

  pfos.parquet          (PFOS_PARQUET_TYPES)         kinematics, PDG, charge,
                                                     cluster_ids -> clusters table,
                                                     track_ids   -> tracks table (= ActsTracks order)
  calo_clusters.parquet (CALO_CLUSTERS_PARQUET_TYPES) energy, position,
                                                     cell_ids -> CALO_CELLS.cell_id (same event)

Cluster hit references are podio ObjectIDs spanning the four digi collections; their
collectionIDs are resolved from the digiLinkCaloHit* link branches and mapped to cellIDs.

Usage:
  convert_pfos.py --input reco_edm4hep.root --output-pfos pfos.parquet \
      --output-clusters calo_clusters.parquet [--max-events N]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import uproot

from utils.parquet_schemas import PFOS_PARQUET_TYPES, CALO_CLUSTERS_PARQUET_TYPES

logger = logging.getLogger(__name__)

PFO = "GaudiPandoraPFOs"
CLU = "GaudiPandoraClusters"
DIGI_LINKS = [
    ("digiECalBarrelCollection", "digiLinkCaloHitECALBarrel"),
    ("digiECalEndcapCollection", "digiLinkCaloHitECALEndcap"),
    ("digiHCalBarrelCollection", "digiLinkCaloHitHCALBarrel"),
    ("digiHCalEndcapCollection", "digiLinkCaloHitHCALEndcap"),
]


def branches():
    out = [
        f"{PFO}/{PFO}.PDG", f"{PFO}/{PFO}.charge", f"{PFO}/{PFO}.energy",
        f"{PFO}/{PFO}.momentum.x", f"{PFO}/{PFO}.momentum.y", f"{PFO}/{PFO}.momentum.z",
        f"{PFO}/{PFO}.goodnessOfPID",
        f"{PFO}/{PFO}.clusters_begin", f"{PFO}/{PFO}.clusters_end",
        f"{PFO}/{PFO}.tracks_begin", f"{PFO}/{PFO}.tracks_end",
        f"_{PFO}_clusters/_{PFO}_clusters.index",
        f"_{PFO}_tracks/_{PFO}_tracks.index",
        f"{CLU}/{CLU}.energy",
        f"{CLU}/{CLU}.position.x", f"{CLU}/{CLU}.position.y", f"{CLU}/{CLU}.position.z",
        f"{CLU}/{CLU}.hits_begin", f"{CLU}/{CLU}.hits_end",
        f"_{CLU}_hits/_{CLU}_hits.index", f"_{CLU}_hits/_{CLU}_hits.collectionID",
    ]
    for digi, link in DIGI_LINKS:
        out += [f"{digi}/{digi}.cellID", f"_{link}_from/_{link}_from.collectionID"]
    return out


def resolve_collection_ids(arrays, n_events):
    """digi collectionID -> digi collection name, from the link 'from' branches."""
    ids = {}
    for digi, link in DIGI_LINKS:
        col = arrays[f"_{link}_from/_{link}_from.collectionID"]
        for ie in range(n_events):
            a = np.asarray(col[ie])
            if len(a):
                ids[int(a[0])] = digi
                break
    return ids


def nested(vals_per_obj):
    return [v.tolist() for v in vals_per_obj]


def process_chunk(tree, entry_start, entry_stop, coll_ids_cache):
    arrays = tree.arrays(branches(), entry_start=entry_start, entry_stop=entry_stop)
    n_events = len(arrays[f"{PFO}/{PFO}.energy"])
    if not coll_ids_cache:
        coll_ids_cache.update(resolve_collection_ids(arrays, n_events))

    pfo_rows, clu_rows = [], []
    for ie in range(n_events):
        g = lambda coll, name: np.asarray(arrays[f"{coll}/{coll}.{name}"][ie])
        gr = lambda name: np.asarray(arrays[name][ie])

        # ---- clusters
        c_en = g(CLU, "energy").astype(np.float32)
        hb = g(CLU, "hits_begin").astype(np.int64)
        he = g(CLU, "hits_end").astype(np.int64)
        hit_idx = gr(f"_{CLU}_hits/_{CLU}_hits.index")
        hit_cid = gr(f"_{CLU}_hits/_{CLU}_hits.collectionID")
        cellid_of = {}
        for digi, _ in DIGI_LINKS:
            cellid_of[digi] = g(digi, "cellID").astype(np.uint64)
        cell_ids = []
        for b, e in zip(hb, he):
            ids = np.empty(e - b, dtype=np.uint64)
            for j, k in enumerate(range(b, e)):
                digi = coll_ids_cache.get(int(hit_cid[k]))
                ids[j] = cellid_of[digi][hit_idx[k]] if digi else 0
            cell_ids.append(ids)
        clu_rows.append({
            "cluster_id": np.arange(len(c_en), dtype=np.uint32),
            "energy": c_en,
            "x": g(CLU, "position.x").astype(np.float32),
            "y": g(CLU, "position.y").astype(np.float32),
            "z": g(CLU, "position.z").astype(np.float32),
            "cell_ids": nested(cell_ids),
        })

        # ---- PFOs
        p_en = g(PFO, "energy").astype(np.float32)
        cb = g(PFO, "clusters_begin").astype(np.int64)
        ce = g(PFO, "clusters_end").astype(np.int64)
        tb = g(PFO, "tracks_begin").astype(np.int64)
        te = g(PFO, "tracks_end").astype(np.int64)
        rel_clu = gr(f"_{PFO}_clusters/_{PFO}_clusters.index")
        rel_trk = gr(f"_{PFO}_tracks/_{PFO}_tracks.index")
        pfo_rows.append({
            "pfo_id": np.arange(len(p_en), dtype=np.uint32),
            "pdg": g(PFO, "PDG").astype(np.int32),
            "charge": np.clip(np.round(g(PFO, "charge")), -127, 127).astype(np.int8),
            "energy": p_en,
            "px": g(PFO, "momentum.x").astype(np.float32),
            "py": g(PFO, "momentum.y").astype(np.float32),
            "pz": g(PFO, "momentum.z").astype(np.float32),
            "goodness_of_pid": g(PFO, "goodnessOfPID").astype(np.float32),
            "cluster_ids": nested([rel_clu[b:e].astype(np.uint32) for b, e in zip(cb, ce)]),
            "track_ids": nested([rel_trk[b:e].astype(np.uint32) for b, e in zip(tb, te)]),
        })
    return pfo_rows, clu_rows


def convert_file(input_path: Path, out_pfos: Path, out_clusters: Path,
                 max_events: int = -1, event_id_offset: int = 0, chunk: int = 200):
    tree = uproot.open(input_path)["events"]
    n = tree.num_entries if max_events < 0 else min(max_events, tree.num_entries)
    pfo_schema = pa.schema(PFOS_PARQUET_TYPES)
    clu_schema = pa.schema(CALO_CLUSTERS_PARQUET_TYPES)
    w_pfo = pq.ParquetWriter(out_pfos, pfo_schema, compression="zstd")
    w_clu = pq.ParquetWriter(out_clusters, clu_schema, compression="zstd")
    coll_ids = {}
    written = 0
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        pfo_rows, clu_rows = process_chunk(tree, start, stop, coll_ids)
        for rows, schema, writer, types in ((pfo_rows, pfo_schema, w_pfo, PFOS_PARQUET_TYPES),
                                            (clu_rows, clu_schema, w_clu, CALO_CLUSTERS_PARQUET_TYPES)):
            table = {k: [] for k in types}
            for i, r in enumerate(rows):
                table["event_id"].append(np.uint32(event_id_offset + start + i))
                for k, v in r.items():
                    table[k].append(v.tolist() if isinstance(v, np.ndarray) else v)
            writer.write_table(pa.table(table, schema=schema))
        written += len(pfo_rows)
        logger.info("pfos: %s events %d-%d", input_path.name, start, stop - 1)
    w_pfo.close()
    w_clu.close()
    return written


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output-pfos", type=Path, required=True)
    ap.add_argument("--output-clusters", type=Path, required=True)
    ap.add_argument("--max-events", type=int, default=-1)
    ap.add_argument("--event-id-offset", type=int, default=0)
    args = ap.parse_args()
    n = convert_file(args.input, args.output_pfos, args.output_clusters,
                     args.max_events, args.event_id_offset)
    logger.info("wrote %s + %s: %d events", args.output_pfos, args.output_clusters, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
