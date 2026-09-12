# ColliderML production workflow — the campaign runbook

The end-to-end recipe for producing a dataset, written down so nobody reinvents
it (or re-breaks it) per campaign. Everything here was learned the expensive way
during the `drift_beamspot` campaign (2026-07/08); incidents are cited inline so
the "why" survives.

**The prime directive: every stage submission gets its config discussed and
approved BEFORE it runs — including recovery/redo jobs.** Proven values are
workload-specific: a packing proven on muons is not proven on ttbar
(`57452905`: 32 runs/node, fine for muons, corrupted ~709/786 ttbar sims).

## Stage chain

```
gun datasets  : particlegun_generation -> simulation -> digitization -> package_parquet
madgraph      : madgraph_init -> madgraph_generation -> merge_smear ->
                [normalise_runs] -> simulation -> digitization -> package_parquet
```

All driven by `scripts/cli/run_stage.py <config> [--run-range A B | --run-list ...]`.
Gate every generated/edited config with `scripts/cli/verify_config_intent.py`
plus a zero-overwrite scan of the target run dirs before submitting.

## Proven job shapes (as of 2026-08)

| stage | workload | runs/node | time | provenance |
|---|---|---|---|---|
| particlegun_generation | muons 100k ev/run | 32 | 45m | 12.8M pilot + 200M campaign |
| simulation | muons 100k ev/run | 32 | 2h | 1888-run campaign |
| simulation | **ttbar 1280 ev/run** | **16** | **3h** | hard_scatter/ttbar 1000-run campaign. NOT 32 (57452905 mass failure) |
| digitization | muons 100k ev/run | 32 | 1h | 2017-run campaign |
| digitization | ttbar 1280 ev/run | 16 | 1h | hard_scatter/ttbar |
| merge_smear | ttbar | 32 | 30m | light, single-threaded |
| madgraph_generation | ttbar 17.5k ev/MG-run | **8** (`mg_nb_core: 32`) | **4h** | packing test 2026-08-28: A/B/C = 1/4/8 runs/node -> 159.6 / 64.5 / 33.6 node-h per 1M showered events (C = 4.75x cheaper, outputs identical-quality: 6 full files/run, same survival). Single-run utilization measured 3.3% - MG's shower/tail phases are near-serial, so whole-node-per-run pays ~97% idle. ALWAYS set mg_nb_core = 256/runs_per_node. 2h wall killed slow-channel runs (9, 34) at 1/node; C arm ran 2h04 for 8 runs |
| package_parquet | any | 32 chunks/node | 30-60m | ~36s/chunk (muons) |

`qos: debug` caps at 8 nodes; anything larger goes `regular`. Prefer regular for
jobs that write into web-served directories (a mid-write kill leaves partial
files in public view).

## Non-negotiable config keys

- `common.cache_dir: <repo>/.cache` — the sw:pr-8 image ships an EMPTY Geant4
  data dir. Without the cache every task downloads ~2 GB; it held at 128 tasks
  and collapsed at 1888 (sim job 56955129: 82/1888 runs survived).
- Simulation (tracker-only campaigns): `odd_compact_files` with Defs + Tracker +
  the campaign's fields/plugins XML. Without it ddsim silently simulates the
  full detector.
- Simulation log hygiene: `ddsim_printLevel: 6`, `filter_g4_warnings: true`.
- Digitization (native parquet era): `output_parquet_arrow: true`,
  `truth_tracking: true` (+ `truth_tracking_fitter: kf`,
  `truth_tracking_prefit: gx2f`), `parquet_events_per_shard: 100000`,
  `parquet_events_per_row_group: 1000` (row group bounds write memory — never
  let it follow a large shard).
- Truth-seed window: `truth_tracking_delta_r: [10.0, 1.0e6]` and
  `truth_tracking_abs_delta_z: [0.0, 1.0e6]` (code default since 74a9b94,
  2026-09-07; write them out so the intent is visible). ACTS' own defaults
  (200 mm / 500 mm) made the seed start at layer 4 or a disc for forward tracks
  from the wide-z beamspot, and the fit never visits hits inside the bottom
  space point: v1 truth_tracks of every dataset digitized before 2026-09-07
  (uniform, loguniform, ttbar) lack the innermost pixel hit on 1.2% of tracks,
  ~10% in 2.5<|eta|<2.75. The four discrete muon bins (2/10/50/100 GeV) were
  re-digitized with the fix. Audit after any digitization: innermost hit on
  track must be 100% (per-track hit accounting, not job status).
- `run_stage.py` needs the `collider-env` conda python
  (`~/.conda/envs/collider-env/bin/python`): the system python3 has a
  simple_slurm without `Slurm.add_cmd` and every submission fails with an
  AttributeError. `verify_packaged_parquet.py` needs python >= 3.10 (same env).

## MadGraph layout facts

- MG run `N` writes its split output dirs at `N*max_files_per_mg_run + chunk`
  (see `madgraph_gen.py:_calculate_split_config`). With `max_files_per_mg_run:
  50` and ~6 surviving files per run, the runs tree is SPARSE: 0-5, 50-55,
  100-105, ... This is by design.
- Each MG run leaves one trailing dir containing a ZERO-BYTE `events.hepmc`
  (the splitter discards the final partial chunk after creating the dir). These
  are junk: move them to staging before normalising — `normalise_runs.py` only
  removes truly empty dirs, and a 0-byte file makes a dir non-empty.
- Packing: 8 MG runs/node with `mg_nb_core: 32` is the proven-efficient shape
  (see table); madgraph_gen.py caps MG's core auto-detection via the
  `mg_nb_core` config key - without it every co-scheduled instance grabs all
  256 cores.
- FxFx survival is ~44-58%, so downstream dir counts are estimates until
  generation finishes. Build sim/digi run lists from what exists (non-empty
  `events.hepmc` above ~140 MB for 1280-event files).

## Making runs contiguous

`scripts/postprocessing/normalise_runs.py <runs_dir> [--dry-run]` — removes
empty dirs and renumbers the rest contiguous from 0. Run it AFTER generation
(and after clearing zero-byte trailing dirs), BEFORE digitization, so digi and
packaging see dense numbering. Global event ids are `run * run_size`, and the
packager refuses gappy numbering on purpose.

Renumbering shifts seed provenance: dir `N` may hold data generated with seed
string `..._run<M>` for M != N. The data is valid (seeds are arbitrary and
deterministic); note the mapping in the dataset README if regeneration matters.

## Packaging (`package_parquet` stage)

`scripts/postprocessing/package_native_parquet.py`, registered in
`cli_utils.STAGE_SCRIPT_MAP`, driven like convert_all with `--chunk-index`.

- Layout/naming matches convert_all and the published datasets:
  `parquet/{truth,reco}/<object>/<campaign>.<dataset>.<version>.<group>.<object>.events<A>-<B>.parquet`
  with truth = {particles, tracker_simhits}, reco = {tracker_hits, tracks,
  truth_tracks}.
- Global event numbering: `event_id += run * run_size`, NO run_id column.
- `chunk_size` conventions: single-particle 100k-ev runs -> `1000000`
  (10 runs/file, ~430 MB tracker_hits); ttbar-class 1280-ev runs -> `1000`
  (the hard_scatter/ttbar convention, ~200 MB files).
- **Time units**: the ACTS-native writer emits time in ACTS native units
  (mm, c=1). The packager converts to ns (`TIME_COLUMNS` map). Caught
  downstream as "pixel time median 271 ns" (really 271 mm = 0.91 ns). Strips
  legitimately have time=0 (no time in their ODD measurement subspace); true
  times join via `simhit_ids -> tracker_simhits.true_time`.
- Verify with `scripts/postprocessing/verify_packaged_parquet.py --config <cfg>`
  (exact ranges, row/distinct counts, cross-table event-set equality, gap-free
  tiling; final chunk may be legitimately short).
- `output_base_dir` = `data/ColliderML/simulation` — NOTE
  `www/ColliderML -> data/ColliderML/simulation`, so packaged files are
  web-public the moment they are written. `public/` is a separate curated tree
  (filtered copies, not hardlinks).

## Failure modes and how to read them

- **pasta failures** (`pasta failed ... netlink`): podman startup race when many
  containers launch at once on a node. A few % of tasks, dies BEFORE the
  payload, leaves nothing. Fix: rerun the missing runs. Reruns are bit-identical
  to what the first attempt would have produced — seeds are
  `<dataset>_<version>_run<N>` hashed, a pure function of run number (verified
  by byte-identical regeneration of recovered runs).
- **Validator false alarms**: generation validators glob `events.hepmc3` but
  MG writes `events.hepmc` -> FAILED with correct data. merge_smear reports
  CONFIGURATION_ERROR similarly. Check outputs, not job state.
- **Validator false negatives**: the gen validator counts dirs it FINDS, not
  dirs expected (reported 0 failed when 52 runs were missing). Always count
  outputs against the run list independently.
- **TIMEOUT**: kills tasks mid-write. Audit by CONTENT (uproot: events tree,
  entry count), not by file presence or size alone.
- **Filename coexistence**: old- and new-format parquet have different shard
  names; re-digitizing does NOT overwrite old shards, it adds alongside, and
  the packager concatenates everything in a table dir -> double-counted events.
  Move superseded parquet to `data/ColliderML/staging/` (mv, reversible) before
  re-digitizing.
- **Threaded digitization is not bit-reproducible** (RNG consumption order at
  threads>1); seed still fixes the ensemble. Gen/merge are single-threaded and
  bit-reproducible.

## Verification ladder (per stage, before the next consumes it)

1. Count outputs against the submitted run list (never trust the validator).
2. Content-check a sample (and every file after a TIMEOUT): readable trees,
   expected entries, expected branches (calo=0 for tracker-only).
3. After packaging: `verify_packaged_parquet.py` over all chunks.
4. Physics spot-checks: truth-track validation harness
   (`tests/regression/validate_truth_tracks.py`), time medians ~ns scale.

## Digitisation: GEOMETRIC, always

Every ColliderML dataset uses `digi_config: /opt/odd/config/odd-digi-geometric-config.json`
(real segmentation, charge deposition, clustering). The Gaussian-smearing alternative is a
gross simplification and must never reach published data. The code default in
`digi_and_reco.py` was smearing until 2026-09-11, which silently smeared any config that
omitted the key; the whole drift_beamspot campaign was digitised that way and re-run on
2026-09-12. Audit any new digitisation: `size_loc0`/`size_loc1`/`n_channels`/
`sum_activation` must be non-zero. `local_eta`, `local_phi`, `eta_angle`, `phi_angle` stay
zero either way (Athena-dump-only fields).

Geometric emits NO reco-level time, by design: ColliderML has no calibrated time baseline,
so truth time is published in `tracker_simhits.true_time` and users smear it themselves.
The stored hit variance under geometric is a charge-weighted pitch^2/12 propagation, not the
truth covariance it was under smearing: measured actual/stored is 1.35 in the pixel barrel
r-phi, 0.75 in the short-strip barrel, giving truth-KF pull widths of 1.2 to 1.6.

## Seed acceptance vs the beamspot

`seed_impact_max` defaults to 3.0 mm, the benchmark value for hard_scatter and full_pileup.
drift_beamspot spreads vertices uniformly over +-5 mm transverse, so |d0| reaches 7.1 mm and
at 3 mm the CKF loses ~40% of particles by construction (efficiency 0.97 within 2 mm, 0.09 at
5 mm). That campaign runs at 8.0 mm: efficiency flat at ~0.96 across the whole beamspot, no
rise in fakes, +2.5% CPU. Never widen it for the benchmark campaigns.

## Track covariance in the parquet

The ACTS Arrow track writer emits only the five perigee parameters. The fitted
covariance reaches the published tables through the packaging stage:

- Digitization must set `performance_metrics: true` (CKF/ambi summary) and
  `truth_tracking_root_summary: true` (truth-track summary). Without them the
  ROOT summaries carry only the diagonal `err_*`, and the covariance is
  unrecoverable without re-digitizing. This was missed on the 2026-09-12
  geometric re-run and cost a second 195 node-hour pass: if a re-digitization is
  running, these two keys cost nothing and preserve every option.
- Packaging must set `track_covariance: true`. The packager joins on
  (event, track_id): the ROOT writer stores `track_nr = track.index()` and the
  Arrow writer stores the same value as `track_id`, so it is a key, not an
  ordering assumption (verified: d0/phi/t agree to exactly 0.0, 0 unmatched).
- Published as 15 columns `cov_d0_d0` ... `cov_qop_qop`, the upper triangle of
  the 5x5, in ACTS native units (mm^2, mm*rad, rad^2, rad/GeV, 1/GeV^2).
- The time row and column are dropped on purpose: geometric digitization makes no
  time measurement, so they carry only the seeding prior (`err_t` is a constant
  10 ns for every CKF track, 33 ps for every truth track).
- Sanity values (2026-09-12): sigma(d0) 32.50 / 11.88 / 7.31 / 6.92 um for the
  2/10/50/100 GeV bins, rho(d0,phi) -0.977 -> -0.847 as tracks straighten.
  About 0.5% of CKF covariances are not positive definite as ACTS produces them;
  truth-track covariances are all positive definite.
