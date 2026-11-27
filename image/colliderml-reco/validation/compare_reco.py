#!/usr/bin/env python3
"""Compare the reconstruction output of two EDM4hep files, value by value.

Three checks, all with zero tolerance and in collection order:
  1. strict_compare_pandora.py: PFO, cluster and vertex kinematics and link counts.
  2. Every stored field of GaudiPandoraPFOs, GaudiPandoraClusters and
     GaudiPandoraStartVertices, and the targets of their relations: collection, index, cell
     ID and energy of each hit of a cluster; collection and index of each cluster, track,
     particle and decay vertex of a PFO, and of each particle of a vertex. Floats are
     compared by bit pattern, so NaN equals only the same NaN.
  3. Digitised calorimeter hits (the four digi*Collection collections, which must exist, and
     any other edm4hep::CalorimeterHit collection): cell ID, energy, energy error, time,
     type and position.

Not compared: the hit-to-SimHit link collections and the collections copied from the input.

The reference may be a merged file: --ref-first selects the entries that correspond to
the candidate file. --expect-events fails the comparison if the candidate has another
number of events.

Exit code 0 if no check finds a difference, 1 otherwise. Prints one JSON object.
"""
import argparse
import json
import math
import struct
import sys

from cppyy.gbl.std import string as std_string
from podio import root_io

import strict_compare_pandora as strict

PANDORA_COLLECTIONS = ("GaudiPandoraPFOs", "GaudiPandoraClusters", "GaudiPandoraStartVertices")
DIGI_COLLECTIONS = ("digiECalBarrelCollection", "digiECalEndcapCollection",
                    "digiHCalBarrelCollection", "digiHCalEndcapCollection")


def bits(value):
    """Bit pattern of a float (as a double), so that comparison is exact also for NaN."""
    return struct.pack("<d", float(value)).hex()


def vec3(vector):
    return (bits(vector.x), bits(vector.y), bits(vector.z))


def floats(values):
    return tuple(bits(value) for value in values)


def target(obj):
    """Position of a related object in its collection."""
    object_id = obj.getObjectID()
    return (int(object_id.collectionID), int(object_id.index))


def pfo_record(pfo):
    return (
        pfo.getPDG(), bits(pfo.getEnergy()), vec3(pfo.getMomentum()), vec3(pfo.getReferencePoint()),
        bits(pfo.getCharge()), bits(pfo.getMass()), bits(pfo.getGoodnessOfPID()),
        floats(pfo.getCovMatrix().values),
        tuple(target(x) for x in pfo.getClusters()),
        tuple(target(x) for x in pfo.getTracks()),
        tuple(target(x) for x in pfo.getParticles()),
        target(pfo.getDecayVertex()),
    )


def cluster_record(cluster):
    return (
        cluster.getType(), bits(cluster.getEnergy()), bits(cluster.getEnergyError()),
        vec3(cluster.getPosition()), floats(cluster.getPositionError().values),
        bits(cluster.getITheta()), bits(cluster.getPhi()), vec3(cluster.getDirectionError()),
        floats(cluster.getShapeParameters()), floats(cluster.getSubdetectorEnergies()),
        tuple(target(hit) + (int(hit.getCellID()), bits(hit.getEnergy())) for hit in cluster.getHits()),
        tuple(target(x) for x in cluster.getClusters()),
    )


def vertex_record(vertex):
    return (
        vertex.getType(), bits(vertex.getChi2()), vertex.getNdf(), vec3(vertex.getPosition()),
        floats(vertex.getCovMatrix().values), vertex.getAlgorithmType(), floats(vertex.getParameters()),
        tuple(target(x) for x in vertex.getParticles()),
    )


RECORD = dict(zip(PANDORA_COLLECTIONS, (pfo_record, cluster_record, vertex_record)))


def pandora_records(frame, path):
    """Return {collection name: [records]}; a missing Pandora collection is an error."""
    out = {}
    available = {str(name) for name in frame.getAvailableCollections()}
    for name in PANDORA_COLLECTIONS:
        if name not in available:
            raise SystemExit(f"{path}: collection {name} is missing")
        out[name] = [RECORD[name](obj) for obj in frame.get(name)]
    return out


def calo_hits(frame, path):
    """Return {collection name: [hit tuples]} for all CalorimeterHit collections."""
    out = {}
    available = sorted(str(name) for name in frame.getAvailableCollections())
    for name in DIGI_COLLECTIONS:
        if name not in available:
            raise SystemExit(f"{path}: collection {name} is missing")
    for name in available:
        coll = frame.get(name)
        if coll is None or str(std_string(coll.getValueTypeName())) != "edm4hep::CalorimeterHit":
            continue
        out[name] = [
            (int(hit.getCellID()), bits(hit.getEnergy()), bits(hit.getEnergyError()), bits(hit.getTime()),
             hit.getType(), vec3(hit.getPosition()))
            for hit in coll
        ]
    return out


def load(path, first=0, count=None):
    """Return per-event lists (strict summary, Pandora records, calorimeter hits)."""
    frames = root_io.Reader(str(path)).get("events")
    stop = len(frames) if count is None else first + count
    if stop > len(frames):
        raise SystemExit(f"{path}: entries {first}..{stop - 1} requested, file has {len(frames)}")
    summaries, records, hits = [], [], []
    for i in range(first, stop):
        frame = frames[i]
        summaries.append(strict._extract_event(frame))
        records.append(pandora_records(frame, path))
        hits.append(calo_hits(frame, path))
    return summaries, records, hits


def compare_collections(ref, cand):
    """Compare per-event {name: [records]} dictionaries.

    Returns (number of records compared, events with a difference, messages).
    """
    n_records, events, messages = 0, set(), []
    for i_event, (ref_event, cand_event) in enumerate(zip(ref, cand)):
        if ref_event.keys() != cand_event.keys():
            events.add(i_event)
            messages.append(f"event {i_event}: collections {sorted(ref_event)} != {sorted(cand_event)}")
            continue
        for name in ref_event:
            a, b = ref_event[name], cand_event[name]
            n_records += len(a)
            if a != b:
                events.add(i_event)
                where = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                messages.append(f"event {i_event} {name}: {len(a)} vs {len(b)} objects, first difference at {where}")
    return n_records, sorted(events), messages


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ref")
    parser.add_argument("cand")
    parser.add_argument("--ref-first", type=int, default=None,
                        help="first reference entry; the candidate's event count is read from there")
    parser.add_argument("--expect-events", type=int, default=None)
    args = parser.parse_args()

    cand_summaries, cand_records, cand_hits = load(args.cand)
    n_events = len(cand_summaries)
    if args.expect_events is not None and n_events != args.expect_events:
        raise SystemExit(f"{args.cand}: {n_events} events, expected {args.expect_events}")
    if args.ref_first is None:
        ref_summaries, ref_records, ref_hits = load(args.ref)
    else:
        ref_summaries, ref_records, ref_hits = load(args.ref, args.ref_first, n_events)
    if len(ref_summaries) != n_events:
        raise SystemExit(f"{args.ref} has {len(ref_summaries)} events, {args.cand} has {n_events}")

    _, max_diff, strict_message = strict.compare_events(ref_summaries, cand_summaries, order_invariant=False)
    metrics_ref, metrics_cand = strict.compute_metrics(ref_summaries), strict.compute_metrics(cand_summaries)
    n_objects, pandora_events, pandora_messages = compare_collections(ref_records, cand_records)
    n_hits, hit_events, hit_messages = compare_collections(ref_hits, cand_hits)

    result = {
        "events": n_events,
        "pfos": [metrics_ref["npfo"], metrics_cand["npfo"]],
        "clusters": [metrics_ref["nclu"], metrics_cand["nclu"]],
        "strict_max_abs_diff": max_diff if math.isfinite(max_diff) else "inf",
        "strict_message": strict_message,
        "pandora_objects_compared": n_objects,
        "pandora_events_differing": pandora_events,
        "pandora_messages": pandora_messages[:5],
        "calo_hits_compared": n_hits,
        "calo_events_differing": hit_events,
        "calo_hit_messages": hit_messages[:5],
    }
    # Files without any Pandora object or any hit must not count as agreement.
    result["pandora_identical"] = (n_objects > 0 and not pandora_events
                                   and max_diff == 0.0 and not strict_message)
    result["calo_hits_identical"] = n_hits > 0 and not hit_events
    result["identical"] = result["pandora_identical"] and result["calo_hits_identical"]
    print(json.dumps(result))
    sys.exit(0 if result["identical"] else 1)


if __name__ == "__main__":
    main()
