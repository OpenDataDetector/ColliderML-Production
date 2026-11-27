# Docker pipeline: the full chain on one machine

`scripts/cli/run_pipeline_docker.sh` runs the whole ColliderML chain for one
channel in containers, with Docker, Podman or podman-hpc. It needs no site
config, no Slurm and no CVMFS. The public library (`colliderml.simulate`)
runs the same stage scripts and configs from a clone of this repository at a
pinned tag.

```bash
./scripts/cli/run_pipeline_docker.sh --channel ttbar --events 10 --output ./out
```

## Stages

| # | Stage | Script | Image | Output in `<output>/runs/0/` |
|---|---|---|---|---|
| 1 | `madgraph_init` (ttbar) | `simulation/madgraph_init.py` | sim | `<output>/docker_test/ttbar/v1/madgraph_process.tgz` |
| 2 | `madgraph_generation` (ttbar) | `simulation/madgraph_gen.py` | sim | `<output>/runs/all/0/events.hepmc.gz` |
| 3 | `pythia_generation` | `simulation/pythia_gen.py` | sim | `merged_events.hepmc3` |
| 4 | `simulation` | `simulation/ddsim_run.py` | sim | `edm4hep.root` |
| 5 | `digitization` | `simulation/digi_and_reco.py` | sim | tracker tables, `sim_with_tracks.root`, ROOT summaries |
| 6 | `particles_full` | `postprocessing/convert_particles_full.py` | sim | `particles_full/`, `generator_particles/` |
| 7 | `pandora_reco` | `simulation/pandora_reco.py` | reco | `reco_edm4hep.root` or `reco_parts/` |
| 8 | `reco_tables` | `postprocessing/convert_reco_tables.py` | reco | `calo_cells/`, `calo_clusters/`, `pfos/` |
| 9 | `package_parquet` | `postprocessing/package_native_parquet.py` | sim | `<output>/parquet/{truth,reco}/<table>/` |

`higgs_portal` generates with Pythia8 alone and starts at stage 3.

The stage configs are in `configs_development/docker_test/<channel>/`. They
hold container paths only (`/output`, `/cache`, `/opt/...`). The tracking
settings are those of the v23 production configs: seed finder at 3 T, seed
minimum pT 0.65 GeV, AMVF vertexing, KF truth tracks with a GX2F pre-fit.

## Images

The two image names are set in `scripts/cli/docker_images.env` and nowhere
else. Override them per run with `--sim-image` / `--reco-image`.

| Image | Runs | Size |
|---|---|---|
| sim: `ghcr.io/opendatadetector/sw:pr-8` | generation, Geant4, ACTS, truth tables, packaging | 6 GB unpacked |
| reco: `ghcr.io/opendatadetector/colliderml-reco:20261127` | calorimeter digitisation, Pandora, calo and PFO tables | 27 GB unpacked |

The reco image is built from `image/colliderml-reco/` (recipe, pinned sources,
patches and its validation against the image that produced the v23 pilot runs).
It needs an x86-64 CPU with AVX2 (x86-64-v3).

`scripts/cli/docker_stage_env.sh sim|reco` sets up the environment inside the
container. Both images carry the Open Data Detector under `/opt/odd-install`.
The two builds have the same detector description: the files differ only in
`DetType` flags and in the DDRec extensions the reconstruction reads
(compared file by file on 2026-11-27, `sw:pr-8` against `reco:20261106`).

### Vertex-to-track links

`reco/vertices` gets the columns `track_ids`, `track_weights`,
`truth_interaction` and `truth_interaction_dz` only when digitisation runs in
a sim image with the `colliderml_vertex_links` module. That image is not
published: build it from `image/vertex_track_links/` on top of `sw:pr-8` and
pass it with `--sim-image`. The development configs set
`vertexing_track_links: auto` and `vertex_links: auto`: with the public
`sw:pr-8` the chain runs without the links, with the patched image it writes
them.

```bash
./scripts/cli/run_pipeline_docker.sh --channel ttbar --sim-image <image with the writer>
```

## Options

```
--events N             events to produce (default 10)
--pileup MU            mean pile-up interactions per event (ttbar default 0)
--threads N            Geant4 and ACTS threads (default 2, the minimum)
--pandora-processes N  concurrent Pandora processes (default 1)
--seed N               random seed (default 42)
--output DIR           output directory (default ./output); must not hold a run yet
--cache DIR            cache directory (default <repo>/.cache)
--runtime NAME         docker, podman-hpc or podman (default: the first one found)
--from STAGE, --to STAGE   run part of the chain in an existing output directory;
                       events and pile-up default to those of the run it continues
--list-stages          print the stages of the channel
```

A single stage: `./scripts/cli/run_docker.sh <script> <config> [args]`.

`--pandora-processes` also sets how the events are split between Pandora
processes, and the output depends on the split: in an earlier test, 4 of 16
events changed by less than 1% in summed PFO energy with a different split
(`docs/PFLOW_PILOT_ROADMAP.md`, open questions). Keep the number of processes
fixed when you compare runs.

On a host with SELinux enforcing (Fedora, RHEL), the bind mounts of the
checkout, the cache and the output directory need a label that the
containers may read. The scripts do not add `:z` to the mounts; we have not
tested such a host.

### Two threads at least

With `threads: 1` the ddsim of `sw:pr-8` writes every event twice. A request
for 10 events gave 20 entries with event numbers 0,0,1,1 and so on. The log
shows Geant4 falling back to its serial run manager; the cause inside DD4hep
has not been looked into. The script refuses
`--threads 1`, and the simulation configs set `verify_output: true`, which makes
`ddsim_run.py` exit non-zero unless each requested event is in the output once.

## Cache and first run

The cache directory holds the Geant4 datasets (2.1 GB, about 43,000 files),
downloaded inside the container on the first run. On a parallel file system
this copy is slow (28 minutes to CFS in the test below); put the cache on
local disk where possible.

## Timing

Measured on 2026-11-27 on one Perlmutter CPU node (AMD EPYC 7763) with
podman-hpc, output on CFS, Geant4 cache in place, at commit `c4b6493`.

| Stage | ttbar, 10 events, no pile-up | ttbar, 2 events, pile-up 200 |
|---|--:|--:|
| settings | 2 threads, 1 Pandora process | 2 threads, 2 Pandora processes |
| `madgraph_init` | 396 s | 475 s |
| `madgraph_generation` | 15 s | 16 s |
| `pythia_generation` | 1 s | 19 s |
| `simulation` | 143 s | 768 s |
| `digitization` | 17 s | 97 s |
| `particles_full` | 3 s | 9 s |
| `pandora_reco` | 38 s | 736 s |
| `reco_tables` | 7 s | 13 s |
| `package_parquet` | 3 s | 7 s |
| whole run | 744 s | 2269 s |
| largest single process | 2.2 GB (simulation) | 11.3 GB (simulation); 5.6 GB per Pandora process |
| output directory | 0.42 GB (tables: 17 MB) | 1.8 GB (tables: 160 MB) |

MadGraph uses every core it is given (128 here).

`<output>/pipeline_timing.csv` has one row per stage: wall seconds, exit
code, the largest resident set of a single process (`max_process_rss_mb`) and
the peak of the cgroup the stage ran in (`cgroup_peak_mb`). Under Slurm with
podman-hpc the cgroup is the job step, so that column covers everything run
in the step so far and includes the page cache.

## podman-hpc on NERSC Perlmutter

- Images must be in the shared store before a compute node can use them:
  `podman-hpc pull <image>`, or `podman-hpc load -i <tar>` followed by
  `podman-hpc migrate <image>`.
- `podman-hpc` is preferred over `podman` when both are on `$PATH`.
- Start containers of one image from one job step at a time. While a step ran
  an `sw:pr-8` container, `podman-hpc run` of the same image from a second
  `srun --overlap` step failed with "creating read-write layer ... transport
  endpoint is not connected".
- podman-hpc passes no host environment into the container; the scripts set
  what the stages need explicitly.
- Compute nodes have outbound network, so the first-run downloads work there.

## Limits

- One run (`runs/0`) per output directory, one parquet file per table.
- Not yet run with Docker or plain Podman: every test used podman-hpc.
- The ttbar process is `p p > t t~` at NLO without extra jets. The released
  samples use FxFx-merged multi-jet matrix elements.
- Both images are x86-64 only.
- With Docker the output files belong to root.
- Production runs (`run_stage.py`, `configs_production/`) do not use these
  scripts; see `PRODUCTION_WORKFLOW.md` and `TWO_CONTAINER_PIPELINE.md`.

## Tests

`pytest tests/unit/test_docker_pipeline.py` checks the stage list against the
scripts and configs, that the development configs hold no host paths or batch
settings, the two `auto` switches and the stage timer. It needs no container.
