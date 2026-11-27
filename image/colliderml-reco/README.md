# ColliderML reconstruction image

Calorimeter digitisation and Pandora particle flow for the Open Data Detector (ODD), as used
for the ColliderML release 2 datasets. The image takes simulated ODD events with tracks and
writes particle-flow objects, calorimeter clusters and digitised calorimeter hits in EDM4hep.

It contains:

- the Key4hep stack (ROOT, Gaudi, k4FWCore, DD4hep, Geant4) from the Key4hep base image;
- PandoraSDK, LCContent and k4GaudiPandora with the ColliderML speed changes, from the tag
  `colliderml-reco-20261127` of the OpenDataDetector forks (see `patches/README.md`);
- k4ODD: the Gaudi options, the Pandora settings (`PandoraSettingsCLD.xml`), the photon
  likelihood data trained on Z->uds events, and the calorimeter calibration constants;
- the ODD geometry with layered calorimeters and a 3 T solenoid field.

Every source is a public repository pinned to a commit in `sources.env`. Licences are in
`THIRD_PARTY.md`: PandoraSDK and LCContent are GPL-3.0, so the reconstruction built from them
is distributed under GPL-3.0 terms.

## Requirements

- x86-64 CPU with AVX2 (x86-64-v3: Intel Haswell, AMD Zen, or newer).
- docker or podman. The image is about 27 GB on disk.
- Memory per process: up to 2.2 GiB observed for ttbar events without pileup, 6.3 GiB at
  200 pileup.

## Quick start

```bash
IMAGE=ghcr.io/opendatadetector/colliderml-reco:20261127
# After a local build (see Building): IMAGE=colliderml/reco:20261127

# 1. Check the image on your machine: simulates 5 photons and reconstructs them (about a minute).
docker run --rm $IMAGE colliderml-reco selftest

# 2. Reconstruct your events, writing the output as yourself.
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work" $IMAGE \
    colliderml-reco run --input /work/sim_with_tracks.root --output /work/reco.root
```

With rootless podman, replace `--user "$(id -u):$(id -g)"` by `--userns=keep-id`, so that
the container user maps to your own account. On hosts with SELinux, mount with
`-v "$PWD:/work:Z"`.

The entrypoint loads the environment and then runs the command it is given, so
`docker run -it $IMAGE bash` gives a shell with `k4run`, `ddsim` and the Pandora stack set
up. In a shell started another way (for example `apptainer exec`, which skips the
entrypoint; we have not tested apptainer), `source /opt/colliderml-reco/env/setup.sh`.

## `colliderml-reco`

```
colliderml-reco run --input SIM.root --output RECO.root [options] [-- K4RUN_ARGS...]
  --events N         number of events (default: all)
  --first-event K    index of the first input event (default: 0)
  --tracks NAME      track collection given to Pandora (default: ActsTracks)
  --no-tracks        calorimeter-only reconstruction
  --settings FILE    Pandora settings XML (default: PandoraSettingsCLD.xml)
  --work-dir DIR     directory for k4run side files (default: RECO.root.work)
  --no-jemalloc      do not preload jemalloc
colliderml-reco selftest [--events N] [--work-dir DIR]
colliderml-reco info
```

Without input of your own, the self-test gives you a small one. It keeps its files when it
is given a directory, and its `sim.root` (photons, no tracks) can be reconstructed again:

```bash
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work" $IMAGE \
    colliderml-reco selftest --work-dir /work/selftest
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work" $IMAGE \
    colliderml-reco run --no-tracks --input /work/selftest/sim.root --output /work/reco.root
```

`run` calls `k4run /opt/k4ODD/k4ODD/options/ODDreconstruction.py`. Calibration constants and
digitisation switches are `K4ODD_*` environment variables read by that file; the defaults are
the calibrated values (`/opt/k4ODD/k4ODD/options/calib_constants_real.env` lists them).

One `k4run` process uses one core. To use a whole machine, start one process per event range
(`--first-event`, `--events`), each with its own `--output`, and merge the outputs with
`podio-merge-files`. At 200 pileup we run one event per process.

### Input

An EDM4hep file (podio 1.4 or compatible) with, per event:

| Collection | Type | Notes |
|---|---|---|
| `ECalBarrelCollection`, `ECalEndcapCollection`, `HCalBarrelCollection`, `HCalEndcapCollection` | `SimCalorimeterHit` | from `ddsim` with the ODD geometry of this image |
| `MCParticles` | `MCParticle` | |
| `EventHeader` | `EventHeader` | one entry per event; the digitisation reads the run and event number |
| `ActsTracks` (or the name given to `--tracks`) | `Track` | with track states; not needed with `--no-tracks` |

The file must carry the cell-ID encoding metadata that `ddsim` writes for the calorimeter
collections. The simulation has to use the same geometry as the reconstruction: the compact
file is `/opt/odd-install/share/OpenDataDetector/xml/OpenDataDetector.xml` in this image, and
`ddsim` is available in it. The ColliderML simulation and tracking chain that produces such
files is in https://github.com/OpenDataDetector/ColliderML-Production.

### Output

The input collections plus:

| Collection | Type |
|---|---|
| `GaudiPandoraPFOs` | `ReconstructedParticle` |
| `GaudiPandoraClusters` | `Cluster` |
| `GaudiPandoraStartVertices` | `Vertex` |
| `digiECalBarrelCollection`, `digiECalEndcapCollection`, `digiHCalBarrelCollection`, `digiHCalEndcapCollection` | `CalorimeterHit` |
| `digiLinkCaloHit*`, `digiRelationCaloHit` | links from digitised hits to `SimCalorimeterHit` |

## Building

```bash
./build.sh 20261127                      # podman; CONTAINER_ENGINE=docker for docker
                                         # result: colliderml/reco:20261127
```

The build needs network access (base image from ghcr.io, sources from github.com and
gitlab.cern.ch, four wheels from PyPI), about 60 GB of disk and 2 GB of memory per compile
job. Once the base image is pulled it takes about 10 minutes on 64 cores. `build.sh` passes
extra arguments to the build command.

What is pinned: the base image by digest, every git source by commit hash, every patch by
SHA-256 (`patches/SHA256SUMS`), every wheel by hash (`requirements.txt`). Repeated builds on the
same machine gave bit-identical Pandora libraries; the image ID differs between builds.

### Variants

The Pandora libraries can be built in two ways (`--build-arg PANDORA_VARIANT=...`). Times are
for one AMD EPYC 7763 core:

| Variant | Sources | Compiler target | PU200 ttbar event |
|---|---|---|---|
| `optimised` (default) | `*_OPT_COMMIT` in `sources.env`: the tag `colliderml-reco-20261127` of the three forks | x86-64-v3 | 22 s (one process), 37 s (64 processes on the node) |
| `reference` | pinned commits + `*_reference` patches: the sources of the image that produced the ColliderML v23 pilot | generic x86-64 | about 18 minutes (28 processes on the node) |

```bash
./build.sh 20261127-reference --build-arg PANDORA_VARIANT=reference
```

## Validation

We reconstructed the same events with this image and with the earlier image that produced
the ColliderML v23 pilot runs, and compared every PFO, cluster, vertex and digitised
calorimeter hit, with all stored fields and relations, at zero tolerance
(`validation/compare_reco.py`).

| Event set | `optimised` (published image): equal / run | `reference` (built with the gap correction on): equal / run |
|---|---|---|
| single particles (photon, pion, K0L; 10 and 100 GeV) | 300 / 300 | 300 / 300 |
| ttbar, no pileup | 1280 / 1280 | 110 / 110 |
| ttbar, 200 pileup | 192 / 192 | 20 / 20 |

Both variants reproduce the earlier output in every event we ran. The `reference` column was
measured on a build made before the gap correction was switched off; with the switch off, the
reference variant gave equal output on the two PU200 events we ran. The `optimised` column is
the published image (`ghcr.io/opendatadetector/colliderml-reco:20261127`), which runs with
the ECAL gap correction off while `reco:20261106` ran with it on. `validation/RESULTS.md` has
the event sets, the timing, and the history of an earlier optimised build that differed in six
events.

## Known limitations

- The ECAL gap correction of `DDCaloDigi` is switched off (`ECALGapCorrection = 0`,
  `patches/k4odd_ecal_gap_correction_off.patch`), where the ColliderML production ran with
  it on. ODD has no dead gaps between ECAL cells, and the correction in k4GaudiPandora reads
  the hits through dangling pointers (key4hep/k4GaudiPandora#68). In our builds it never
  changed a hit energy, and this image gives the same output as the production image
  (Validation).
- Tracks beyond |eta| of about 2 are not extended to the endcap calorimeter face by the
  track creator; `k4run` prints warnings for them.
- The base image tag used for the pin is a moving tag upstream. The digest in `Containerfile`
  stays valid only as long as the Key4hep project keeps that manifest.

## Layout

| Path | Content |
|---|---|
| `Containerfile`, `build.sh` | image recipe |
| `sources.env` | pinned source repositories and commits |
| `steps/` | build steps, run in order inside the image |
| `patches/` | patches and their checksums |
| `env/`, `bin/` | runtime environment, entrypoint and `colliderml-reco` |
| `validation/` | output comparison, image checks, results |
| `nersc/` | how the released image was built and validated on NERSC Perlmutter (site paths) |
| `THIRD_PARTY.md`, `LICENSE`, `LICENSES/` | sources and licences |
