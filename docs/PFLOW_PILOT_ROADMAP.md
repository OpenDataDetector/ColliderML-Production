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
- 2026-10-02: the ACTS re-run uses the current stack (sw image, ODD v6, geometric digitization). The pilot
  therefore ships new tracker_hits and tracks tables. This structure will later become an update to
  Release 1 or a separate Release 2.
- 2026-10-02: track validation (group B) is not a blocker; return to it once the Pandora production runs.
- 2026-10-02: versions are two digits, `v<release><minor>`: the first digit is the release, the second a
  test or pilot round. Pilot 1 is `v20` in the same campaigns as Release 1:
  `simulation/hard_scatter/ttbar/v20/` (PU0) and `simulation/full_pileup/ttbar/v20/` (PU200); Pilot 2 is
  `v21`. Unchanged v1 objects (edm4hep, Release 1 particles) are symlinked from v1, not copied. Existing
  v2, v3, v5 directories stay as they are (internal test passes, not releases). Release 1 stays `v1`.

## Todo

### A. Pipeline fixes before Pilot 1

- [x] A1. (2026-10-01, 23ca725) Particle ids in the re-run tracker tables. The `digi_and_reco` parquet writer numbers particles
      0 to N-1 (ACTS index), and unmatched tracks carry the uint64 maximum. Release 1 uses the edm4hep
      MCParticle index. Map the re-run tracker_hits/tracks particle ids onto the Release 1 ids, so the new
      tables join with Release 1 `truth/particles`. Do not ship the re-run particles table.
      Done in `package_native_parquet.py` (`remap_particle_ids`): exact (PDG, momentum, vertex) match to
      edm4hep; 100% of remapped ids found in Release 1 particles on 64 pilot events. Release 1 PU200
      particles checked to use the same convention (4 events, 3 runs).
- [x] A2. Event-number map for every re-run table (done for calo_cells/calo_clusters/pfos in
      `convert_reco_tables.py`; the ACTS tables already use the ddsim file position: pilot and
      Release 1 particle counts agree event by event).
- [x] A3. Global event_ids (done by the packager): re-run tables must use `run * run_size + ddsim position`
      (run_size 1280 for PU0, 64 for PU200), the Release 1 convention.
- [x] A4. (45451f9) Pandora: one working directory per process, and `processes: N` slices a run (shared `ddcalodigi_hist.root` killed 18 of 64
      processes in the pilot wave). Tested: 4 slices x 2 events, merged order identical to input.
- [x] A5. (2026-10-01) Reco image rebuilt: `colliderml/reco:20261001b`, tarball `ColliderML/images/colliderml_reco_20261001b.tar`
      (20261001 shipped without pyarrow: pip --user refused in the spack venv). Also fixed: container env quoting
      (host $PYTHONPATH leaked into containers), reco_tables env sourcing the sim-image assembler. Old note: rebuild the reco container image from `docker/colliderml-reco/Dockerfile` (the saved tarball is gone).
      Pilot 1 can use `pandora_shifter.sh` meanwhile.
- [ ] A6. PU200 configs drafted (`configs_production/full_pileup/ttbar/v20/`); still to do: a measurement of
      ACTS time, handoff file size and Pandora memory at PU200 on one run before the 16-run job.
- [ ] A7. Track converter covariance: `MaxTrackSigmaPOverP` is set to 999 because the ACTS to edm4hep
      conversion inflates the omega covariance. Find the cause or record it as a known issue.

### B. Track validation (after Pilot 1 production starts)

- [ ] B1. Residuals and pulls of d0, z0, phi, theta, q/p vs pT, eta, and truth d0/z0, from
      `tracksummary_ambi.root`: Release 1 run 0 vs the re-run, PU0 and PU200.
- [ ] B2. Compare shapes with published ATLAS Inner Detector performance (d0 and z0 resolution vs pT and eta).
- [ ] B3. Truth-tracking baseline (kf + gx2f) on the same events, to separate fitter effects from
      pattern recognition.
- [ ] B4. Write up findings; decide whether anything changes before Pilot 1.

### C. Pilot 1 production

- [x] C1. ACTS re-run: PU0 runs 0 to 7 (15 min, 1 node, 860 s/run at 16 threads), PU200 runs 0 to 15 (803 s/run at
      8 threads; handoff 9.3 GB per 64-event run).
- [x] C2. Pandora: PU0 done (8 runs in 13 min on 1 node, ~47k ev/node-h). PU200: 8 slices x 8 events did not finish
      a run in 20 min; with 64 single-event slices per run, one run per node: 16-27 min per run (~150-240 ev/node-h),
      slices kept unmerged (`merge_slices: false`; podio-merge-files was slower than the reco).
- [x] C3. reco_tables (PU0 6 min; PU200 8 min per 3 runs/node), plus `tests/regression/test_reco_tables.py` on each run.
- [ ] C4. Physics check: jet and event-level energy response vs Pilot 0 numbers in the September brief.
- [x] C0. (approved 2026-10-01, plus tracker_simhits) Review the v20 configs with Daniel (`configs_production/{hard_scatter,full_pileup}/ttbar/v20/`).
- [x] C0b. Create v20 symlinks to v1 (`scripts/cli/link_version_objects.py`, dry run checked: PU0 19 links,
      PU200 27 links).
- [ ] C5. PU0 published 2026-10-01: `hard_scatter/ttbar/v20/parquet`, 8 objects x 11 files, all 10,240 events;
      ids join Release 1 particles 100% (events 0-63, 9000-9063); calo_cells 17.6 GB (1.7 MB/event, the size driver).
      PU200 published 2026-10-01: `full_pileup/ttbar/v20/parquet`, 8 objects x 11 files, 1,024 events; ids join
      100% (events 0-7, 960-967); calo_cells 40 GB (39 MB/event). Data cards: `v20/README.md`. Write outputs to `v20` (see Decisions) with a data card; symlink unchanged v1 objects.

### D. Documentation

- [x] D1. (v20 README.md written) Data card for each pilot: events, configs, git SHA, image digest, schema, known issues.
- [ ] D2. Correct the September brief: the "72% of calorimeter energy resolves to particles" statement was an
      id-space mismatch in the re-run particles table, not missing truth.
- [ ] D3. Keep `docs/PRODUCTION_WORKFLOW.md` in step with the chain actually used.

### E. After Pilot 1

- [ ] E1. Collect feedback (via Daniel), list changes here, produce Pilot 2.
- [ ] E2. Scale to 1M PU0 and 100k PU200 (estimates: about 150 node-hours PU0, about 1,100 PU200).

## Open questions

0. Delete the incomplete 10.5 GB `full_pileup/ttbar/v20/runs/0/reco_edm4hep.root` (killed merge)? Asked 2026-10-01.
0b. calo_cells is the size driver (1.7 MB/event PU0, 39 MB/event PU200): format/threshold choice before scaling.

Tracked here until answered; answered ones move to Decisions.

1. Track validation: is there a specific plot from the team showing the odd shape, to reproduce first?

## Data locations (checked 2026-10-02)

- Web portal: `/global/cfs/cdirs/m4958/www/ColliderML` is a symlink to
  `/global/cfs/cdirs/m4958/data/ColliderML/simulation`, so the portal serves the `simulation` tree.
- Release 1 PU0 ttbar: `simulation/hard_scatter/ttbar/v1/` (`runs/<n>/` ROOT files, `parquet/{reco,truth}/`,
  1000 files of 1000 events).
- Release 1 PU200 ttbar: `simulation/full_pileup/ttbar/v1/` (`runs/<n>/edm4hep.root` only, `parquet/` with
  files of 100 events).
- `data/ColliderML/public/`: older pilot and taster sets plus `manifest.json`; it has no Release 1 PU200 parquet.
- September Pilot 0 output: `data/ColliderML/staging/release2_pilot/ttbar_pu0/v1/` (not served by the portal).
