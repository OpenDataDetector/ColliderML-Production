#!/usr/bin/env python3
"""List the PFOs and clusters that are present in only one of two reconstruction outputs.

Usage: describe_difference.py REF.root CAND.root [--ref-first N]

Objects are matched on all their stored values. Relations enter the match by content and
not by index, because indices shift when an object is added or removed: a cluster is matched
with its list of hits (cell ID and energy), a PFO with the number of its clusters and tracks.
For every event with a difference, prints the number of matched objects and a summary of
the unmatched ones.
"""
import argparse
import collections
import json

from podio import root_io

import compare_reco


def pfo_key(pfo):
    record = compare_reco.pfo_record(pfo)
    # Fields 8 to 11 are relation targets (clusters, tracks, particles, decay vertex).
    return record[:8] + tuple(len(x) for x in record[8:11])


def cluster_key(cluster):
    record = compare_reco.cluster_record(cluster)
    # Field 10 is the hit list (collection, index, cell ID, energy); field 11 the sub-clusters.
    return record[:10] + (tuple(hit[2:] for hit in record[10]), len(record[11]))


def pfo_summary(pfo):
    return {"pdg": pfo.getPDG(), "energy": pfo.getEnergy(), "n_tracks": len(pfo.getTracks()),
            "n_clusters": len(pfo.getClusters())}


def cluster_summary(cluster):
    position = cluster.getPosition()
    return {"energy": cluster.getEnergy(), "position": [position.x, position.y, position.z],
            "n_hits": len(cluster.getHits())}


def unmatched(ref, cand, key, summary):
    """Return (number of matched objects, summaries only in ref, summaries only in cand)."""
    ref_keys = [key(obj) for obj in ref]
    cand_keys = [key(obj) for obj in cand]
    only_ref = collections.Counter(ref_keys) - collections.Counter(cand_keys)
    only_cand = collections.Counter(cand_keys) - collections.Counter(ref_keys)

    def pick(objects, keys, counter):
        counter, out = counter.copy(), []
        for obj, k in zip(objects, keys):
            if counter[k] > 0:
                counter[k] -= 1
                out.append(summary(obj))
        return out

    in_ref, in_cand = pick(ref, ref_keys, only_ref), pick(cand, cand_keys, only_cand)
    return len(ref_keys) - len(in_ref), in_ref, in_cand


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ref")
    parser.add_argument("cand")
    parser.add_argument("--ref-first", type=int, default=0)
    args = parser.parse_args()

    cand_frames = root_io.Reader(args.cand).get("events")
    ref_frames = root_io.Reader(args.ref).get("events")
    if args.ref_first + len(cand_frames) > len(ref_frames):
        raise SystemExit(f"{args.ref} has {len(ref_frames)} events, too few for the candidate")
    for i in range(len(cand_frames)):
        ref, cand = ref_frames[args.ref_first + i], cand_frames[i]
        n_pfos, pfos_ref, pfos_cand = unmatched(
            ref.get("GaudiPandoraPFOs"), cand.get("GaudiPandoraPFOs"), pfo_key, pfo_summary)
        n_clusters, clusters_ref, clusters_cand = unmatched(
            ref.get("GaudiPandoraClusters"), cand.get("GaudiPandoraClusters"), cluster_key, cluster_summary)
        if not (pfos_ref or pfos_cand or clusters_ref or clusters_cand):
            continue
        print(json.dumps({
            "event": i,
            "pfos_matched": n_pfos, "pfos_only_in_ref": pfos_ref, "pfos_only_in_cand": pfos_cand,
            "clusters_matched": n_clusters,
            "clusters_only_in_ref": clusters_ref, "clusters_only_in_cand": clusters_cand,
        }))


if __name__ == "__main__":
    main()
