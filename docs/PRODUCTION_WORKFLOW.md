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
| madgraph_generation | ttbar 17.5k ev/MG-run | 1 | **3h** | 2h wall killed slow-channel runs (9, 34) |
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

## MadGraph layout facts

- MG run `N` writes its split output dirs at `N*max_files_per_mg_run + chunk`
  (see `madgraph_gen.py:_calculate_split_config`). With `max_files_per_mg_run:
  50` and ~6 surviving files per run, the runs tree is SPARSE: 0-5, 50-55,
  100-105, ... This is by design.
- Each MG run leaves one trailing dir containing a ZERO-BYTE `events.hepmc`
  (the splitter discards the final partial chunk after creating the dir). These
  are junk: move them to staging before normalising — `normalise_runs.py` only
  removes truly empty dirs, and a 0-byte file makes a dir non-empty.
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
