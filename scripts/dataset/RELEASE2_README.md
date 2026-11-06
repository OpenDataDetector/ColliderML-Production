---
license: cc-by-4.0
pretty_name: ColliderML Dataset Release 2
tags:
- physics
- high-energy-physics
- particle-physics
- collider-physics
- particle-flow
- tracking
- calorimetry
- machine-learning
- simulation
size_categories:
- 10K<n<100K
---

# ColliderML Release 2

Release 2 adds particle flow to ColliderML. The events are the Release 1 simulations
([CERN/ColliderML-Release-1](https://huggingface.co/datasets/CERN/ColliderML-Release-1)), reconstructed again
with ACTS tracking and Pandora particle flow on the Open Data Detector (ODD). New tables: Pandora particle-flow
objects (PFOs), calorimeter clusters, digitised calorimeter cells, tracker sim hits and truth-seeded tracks.

This is a pilot release: expect changes. Every change is listed under "Changes" below.

## What is here

One config per process, pileup and object: `{process}_{pu0|pu200}_{object}`, e.g. `ttbar_pu200_pfos`.

| Process | Pileup | Events | Version |
|---|---|---|---|
| ttbar | 0 | 10,240 | v22 |
| ttbar | 200 | 10,048 | v22 |

| Object | One row per event, list columns per | Main join keys |
|---|---|---|
| `particles` | MCParticle (PU0: the full record, generator history included; PU200: the particles Geant4 tracked, as in Release 1) | `particle_id`, `parent_id` / `parent_ids` |
| `tracker_simhits` | Geant4 tracker hit | particle id |
| `tracker_hits` | digitised tracker measurement | particle ids, sim hit ids |
| `tracks` | ACTS track (CKF + ambiguity resolution), 5x5 covariance, chi2, ndf, holes, outliers, shared hits | `track_id` |
| `truth_tracks` | truth-seeded ACTS fit | particle id |
| `calo_cells` | digitised calorimeter cell (GeV, ns) with per-particle Geant4 energy | `cell_id`, `contrib_particle_ids` |
| `calo_clusters` | Pandora cluster | `cell_ids` -> `calo_cells.cell_id` |
| `pfos` | Pandora PFO | `cluster_ids` -> `calo_clusters`, `track_ids` -> `tracks.track_id` |

```python
from datasets import load_dataset

pfos = load_dataset("CERN/ColliderML-Release-2", "ttbar_pu200_pfos", split="train")
```

## Conventions

- `event_id` joins every table of a process and pileup, and the same event in Release 1:
  event_id = run * 1280 + position for PU0, run * 64 + position for PU200 (position in the run's
  simulation file).
- Every particle id is the index of the particle in the simulation's MCParticle list, the Release 1
  `particle_id`.
- Tracks with no majority particle carry particle id 18446744073709551615.
- Calorimeter truth is the particle that deposited the energy, often a shower secondary. Follow `parent_id`
  to the particle that entered the calorimeter.
- An event with no objects of a type has no row in that table. Join on `event_id` with a left join from
  `particles`.
- For ttbar PU200, `particles` comes from Release 1 in files of 100 events and covers events 0-10,099; the
  other tables cover 0-10,047.

## Detector and software

- Simulation: Release 1 (Geant4 via DD4hep, ODD).
- Tracking: ACTS on ODD v6.
- Calorimeter digitisation and particle flow: DDCaloDigi and Pandora (k4GaudiPandora, k4ODD) with CLD-style
  settings refit for ODD. Two bug fixes are included: ACTS tracks now reach Pandora's track-cluster matching,
  and the ODD calorimeter no longer reports each layer's material twice.

## Effect of the fixes (ttbar)

| | before (v20) | Release 2 (v22) |
|---|---|---|
| tracks not used by Pandora, PU0 | 16.0% | 0.5% |
| tracks not used by Pandora, PU200 | 14.2% | 0.2% |
| jet energy resolution, RMS90 / mean, PU0 | 0.088 | 0.064 |
| W-decay electrons labelled e, PU0 | 0.32 | 0.89 |
| tracked W-decay electrons labelled e, PU200 | 0.20 | 0.83 |
| double-counted energy / PFO energy, PU200 | 12.4% | 8.7% |

PU0: 2,560 events; PU200: 1,024 events with the same tracking in both versions.

## Known limitations

- **Last calorimeter layer.** The digitisation gives the last ECal layer and the last HCal layer zero energy
  (about 0.5-1.6% of the calorimeter energy at PU200, most in the endcaps). A fix is being prepared; the files
  will be replaced and listed under "Changes".
- **Jets at PU200.** Subtract pileup with an eta-strip rho (kT jets, R = 0.4, active areas, median pT/area in
  strips of eta). A global rho over-subtracts central jets (response 0.81 vs 0.91).
- **Photons at PU200.** Photon-PFO purity is 0.67 overall and 0.735 for photons above 5 GeV within abs eta < 2.5.
  Many soft and forward photon PFOs are fragments of charged-hadron showers.
- **Electron PFOs at PU200** are not pure (purity 0.20 above 5 GeV). Check the truth link before treating a
  PFO with pdg 11 as a prompt electron.
- **Muons.** Muon ID uses no muon system; at PU200, 0.34 of tracked W-decay muons are labelled mu.
- Charged particles below the ACTS pT threshold (0.7 GeV) have no track and are reconstructed as neutrals.
- Neutral hadrons at 50-100 GeV: energy resolution 9-11%.

## Citation and contact

ColliderML: https://github.com/OpenDataDetector/ColliderML. Contact: daniel.thomas.murnane@cern.ch.

## Changes

- First upload: ttbar PU0 and PU200, v22. `calo_cells` truth holds one entry per particle per cell.
