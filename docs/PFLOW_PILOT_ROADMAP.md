# Particle-flow pilot roadmap

Living todo list for the Pandora particle-flow pilot data. Updated as work lands; decisions are dated.
Feedback from the ML-pflow group and the calorimeter experts comes through Daniel only.

## Goal

Pandora outputs (calorimeter cells, clusters, PFOs) in parquet, released on top of the existing
Release 1 ttbar events, served from the NERSC world-readable ColliderML data directory.

| Round | Sample | Events | Release 1 runs | Release 1 event_ids |
|---|---|---|---|---|
| Pilot 1 | `hard_scatter/ttbar/v1` (PU0) | 10,240 | 0 to 7 (1280 events each) | 0 to 10,239 |
| Pilot 1 | `full_pileup/ttbar/v1` (PU200) | 1,024 | 0 to 15 (64 events each) | 0 to 1,023 |
| Pilot 2 | same events | same | same | same |
| Later | 1M PU0, 100k PU200 | | | |

Cycle: produce pilot, share, collect feedback (about 2 weeks), apply changes, produce the next pilot.

## Decisions

- 2026-09-26: calorimeter truth is complete; every hit has truth by construction. Pandora tables use the
  Release 1 particle ids (edm4hep MCParticle index), which match `truth/particles` for 100% of the
  contribution energy (32 PU0 events checked).
- 2026-09-26: distribution is the NERSC public directory; no Hugging Face upload for pilots.
- 2026-10-01: re-run ACTS tracking for the pilot events (Release 1 kept no fitted track states for PU200,
  and Pandora needs states at the IP, first hit and last hit). Fold a track-parameter validation into it:
  the Release 1 track parameter resolutions (for example vs d0 and z0) look different in shape from ATLAS.
- 2026-10-01: pilot events are exactly the first runs listed in the Goal table.

## Todo

### A. Pipeline fixes before Pilot 1

- [ ] A1. Particle ids in the re-run tracker tables. The `digi_and_reco` parquet writer numbers particles
      0 to N-1 (ACTS index), and unmatched tracks carry the uint64 maximum. Release 1 uses the edm4hep
      MCParticle index. Map the re-run tracker_hits/tracks particle ids onto the Release 1 ids, so the new
      tables join with Release 1 `truth/particles`. Do not ship the re-run particles table.
- [ ] A2. Event-number map for every re-run table (done for calo_cells/calo_clusters/pfos in
      `convert_reco_tables.py`; check tracks and tracker_hits use the ddsim file position).
- [ ] A3. Global event_ids: re-run tables must use `run * run_size + ddsim position`
      (run_size 1280 for PU0, 64 for PU200), the Release 1 convention.
- [ ] A4. Pandora: one working directory per process (shared `ddcalodigi_hist.root` killed 18 of 64
      processes in the pilot wave).
- [ ] A5. Rebuild the reco container image from `docker/colliderml-reco/Dockerfile` (the saved tarball is gone).
      Pilot 1 can use `pandora_shifter.sh` meanwhile.
- [ ] A6. PU200 configs for digitization, Pandora and reco_tables (none exist yet), and a measurement of
      ACTS time, handoff file size and Pandora memory at PU200 on one run before the 16-run job.
- [ ] A7. Track converter covariance: `MaxTrackSigmaPOverP` is set to 999 because the ACTS to edm4hep
      conversion inflates the omega covariance. Find the cause or record it as a known issue.

### B. Track validation (with the ACTS re-run)

- [ ] B1. Residuals and pulls of d0, z0, phi, theta, q/p vs pT, eta, and truth d0/z0, from
      `tracksummary_ambi.root`: Release 1 run 0 vs the re-run, PU0 and PU200.
- [ ] B2. Compare shapes with published ATLAS Inner Detector performance (d0 and z0 resolution vs pT and eta).
- [ ] B3. Truth-tracking baseline (kf + gx2f) on the same events, to separate fitter effects from
      pattern recognition.
- [ ] B4. Write up findings; decide whether anything changes before Pilot 1.

### C. Pilot 1 production

- [ ] C1. ACTS re-run: PU0 runs 0 to 7, PU200 runs 0 to 15, with the `sim_with_tracks.root` handoff.
- [ ] C2. Pandora on the same runs.
- [ ] C3. reco_tables, plus `tests/regression/test_reco_tables.py` on each run.
- [ ] C4. Physics check: jet and event-level energy response vs Pilot 0 numbers in the September brief.
- [ ] C5. Publish to the public directory (path to be agreed) with a data card.

### D. Documentation

- [ ] D1. Data card for each pilot: events, configs, git SHA, image digest, schema, known issues.
- [ ] D2. Correct the September brief: the "72% of calorimeter energy resolves to particles" statement was an
      id-space mismatch in the re-run particles table, not missing truth.
- [ ] D3. Keep `docs/PRODUCTION_WORKFLOW.md` in step with the chain actually used.

### E. After Pilot 1

- [ ] E1. Collect feedback (via Daniel), list changes here, produce Pilot 2.
- [ ] E2. Scale to 1M PU0 and 100k PU200 (estimates: about 150 node-hours PU0, about 1,100 PU200).

## Open questions

Tracked here until answered; answered ones move to Decisions.

1. ACTS configuration for the re-run: the current stack (sw image, ODD v6, geometric digitization, as in
   Pilot 0) or the Release 1 configuration (ODD v4, Release 1 digitization). The current stack produces a
   new tracker_hits table as well as new tracks.
2. Public path: a separate pilot directory such as `public/hard_scatter/ttbar/pflow_pilot_v1/`, rather than
   adding objects inside the Release 1 `v1` tree.
3. Track validation: is there a specific plot from the team showing the odd shape, to reproduce first?
