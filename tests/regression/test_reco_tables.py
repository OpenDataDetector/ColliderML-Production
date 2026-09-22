"""Release-2 calorimeter / particle-flow table contract (calo_cells,
calo_clusters, pfos), as written per run by reco_tables and published by
package_native_parquet.

calo_cells     one entry per digitised cell: detector code, cell_id, energy,
               position, time, nested truth contributions (particle_id, energy).
calo_clusters  one entry per Pandora cluster: energy, position, cell_ids ->
               calo_cells.cell_id of the same event.
pfos           one entry per Pandora PFO: kinematics, pdg, charge, cluster_ids ->
               calo_clusters row index, track_ids -> tracks row index (ActsTracks
               order) of the same event.

Reads the per-run tree at $COLLIDERML_ACTSNATIVE_PARQUET_DIR like the tracker
suite; skips cleanly when unset.
"""

import math


def _rows(df):
    return list(df.iter_rows(named=True))


def test_same_event_set_as_particles(acts_particles, acts_calo_cells, acts_calo_clusters, acts_pfos):
    """The three reco tables cover exactly the events the truth table covers;
    a k4run that stopped early would show up here, not downstream."""
    truth = set(acts_particles["event_id"].to_list())
    for name, df in (("calo_cells", acts_calo_cells), ("calo_clusters", acts_calo_clusters), ("pfos", acts_pfos)):
        ev = set(df["event_id"].to_list())
        assert ev == truth, (
            f"{name}: event set differs from particles (missing {len(truth - ev)}, extra {len(ev - truth)})"
        )


def test_cell_contrib_particle_ids_resolve(acts_calo_cells, acts_particles):
    """Truth contributions carry the MCParticle row index. The particles table is
    SPARSE (the ACTS writer keeps generator particles and those with tracker hits,
    not every Geant4 secondary), so a contribution may legitimately point outside
    it. What must hold: no negative ids, and most of the deposited energy resolves
    to a published particle (pilot 2026-09-22, hard-scatter ttbar: 72% by energy,
    70% by count). The bound below is deliberately loose; it guards against a
    broken index convention, not against physics."""
    pid_sets = {int(r["event_id"]): set(r["particle_id"]) for r in _rows(acts_particles)}
    tot_e = res_e = 0.0
    neg = 0
    for r in _rows(acts_calo_cells):
        s = pid_sets[int(r["event_id"])]
        for pids, es in zip(r["contrib_particle_ids"], r["contrib_energies"]):
            for pid, e in zip(pids, es):
                tot_e += e
                if pid < 0:
                    neg += 1
                elif pid in s:
                    res_e += e
    assert neg == 0, f"{neg} negative contribution particle ids"
    assert tot_e > 0
    frac = res_e / tot_e
    assert frac > 0.5, f"only {frac:.3f} of contribution energy resolves into the particles table"


def test_cell_energies_nonnegative_finite(acts_calo_cells):
    """Digitised cell energies are finite and not negative. Exactly-zero cells DO
    occur (about 1% in the pilot: realistic ECAL digitisation can round a
    sub-threshold deposit to zero and DDCaloDigi still writes the cell); they are
    reported, not failed."""
    zero = total = 0
    for r in _rows(acts_calo_cells):
        for e in r["energy"]:
            assert math.isfinite(e) and e >= 0, f"event {r['event_id']}: cell energy {e}"
            total += 1
            zero += e == 0
    print(f"zero-energy cells: {zero} of {total} ({zero / max(total, 1):.4f})")


def test_cluster_cell_ids_resolve(acts_calo_clusters, acts_calo_cells):
    """Every cell a cluster claims exists in the same event's calo_cells table,
    and no cluster is empty."""
    cells = {int(r["event_id"]): set(r["cell_id"]) for r in _rows(acts_calo_cells)}
    bad, empty = 0, 0
    for r in _rows(acts_calo_clusters):
        known = cells[int(r["event_id"])]
        for ids in r["cell_ids"]:
            if len(ids) == 0:
                empty += 1
            bad += sum(1 for c in ids if c not in known)
    assert empty == 0, f"{empty} clusters without cells"
    assert bad == 0, f"{bad} cluster cell references not in calo_cells"


def test_pfo_links_resolve(acts_pfos, acts_calo_clusters, acts_tracks):
    """PFO cluster_ids index calo_clusters and track_ids index tracks (ActsTracks
    order) within the event; a PFO has at least one of the two."""
    n_clu = {int(r["event_id"]): len(r["energy"]) for r in _rows(acts_calo_clusters)}
    n_trk = {int(r["event_id"]): len(r["track_id"]) for r in _rows(acts_tracks)}
    bad_clu = bad_trk = orphan = 0
    for r in _rows(acts_pfos):
        ev = int(r["event_id"])
        for cids, tids in zip(r["cluster_ids"], r["track_ids"]):
            if len(cids) == 0 and len(tids) == 0:
                orphan += 1
            bad_clu += sum(1 for c in cids if not (0 <= c < n_clu[ev]))
            bad_trk += sum(1 for t in tids if not (0 <= t < n_trk[ev]))
    assert bad_clu == 0, f"{bad_clu} PFO cluster references out of range"
    assert bad_trk == 0, f"{bad_trk} PFO track references out of range"
    assert orphan == 0, f"{orphan} PFOs with neither clusters nor tracks"


def test_pfo_kinematics_consistent(acts_pfos):
    """Charged PFOs carry a track, neutral ones do not; energy is finite and
    at least |p| within float tolerance."""
    charged_without_track = neutral_with_track = bad_energy = 0
    for r in _rows(acts_pfos):
        for q, e, px, py, pz, tids in zip(r["charge"], r["energy"], r["px"], r["py"], r["pz"], r["track_ids"]):
            p = math.sqrt(px * px + py * py + pz * pz)
            if not math.isfinite(e) or e < 0.999 * p:
                bad_energy += 1
            if q != 0 and len(tids) == 0:
                charged_without_track += 1
            if q == 0 and len(tids) > 0:
                neutral_with_track += 1
    assert bad_energy == 0, f"{bad_energy} PFOs with E < |p| or non-finite E"
    assert charged_without_track == 0, f"{charged_without_track} charged PFOs without a track"
    assert neutral_with_track == 0, f"{neutral_with_track} neutral PFOs with a track"
