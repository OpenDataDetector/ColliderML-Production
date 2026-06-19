# Two-Container Production Pipeline (sim + reco)

Status: **working** (branch `feat/two-container-pipeline`). Both containers build and the
full chain runs end-to-end on Perlmutter via `podman-hpc` — **no cvmfs, shifter, or conda**.
This replaces the old 5-mechanism patchwork (shifter-ATLAS + LCG cvmfs + key4hep cvmfs +
bare-metal acts/dd4hep + host-conda).

## Why two containers (not one)

We first tried to fold the whole pipeline (including Pandora particle flow) into the single
`OpenDataDetector/sw` spack image. It hit a hard wall: that image's **ROOT 6.38 is built
without the Eve component**. Production charged-PF Pandora needs
`k4GaudiPandora → k4Reco::GaudiTrkUtils → DDKalTest → KalTest`, and KalTest's `TRKTrack`
*inherits* `TEveTrackPropagator`, so KalTest cannot build against an Eve-less ROOT. The
key4hep stack is a *different ROOT build* (`+eve`) that production Pandora is welded to.

Decision: **call Pandora a reco task.** Two images:

| Image | Base | Does |
|-------|------|------|
| **sim** `colliderml/acts-arrow` | OpenDataDetector/sw spack image + arrow-ACTS overlay | gen, ddsim, ACTS digitisation + tracking, arrow parquet, all uproot-based converters |
| **reco** `colliderml/reco` | `ghcr.io/key4hep/key4hep-sim-reco-ubuntu24` (no-cvmfs key4hep) | `calo_digitization`, `pandora_reco` (k4run + the calibrated Pandora stack) |

## Geometry: use the calibrated `azaborow/addLayeredCalo_MuonCoil` ODD — for BOTH containers

The calibrated Pandora reco was benchmarked against
`gitlab.cern.ch/azaborow/OpenDataDetector` branch **`addLayeredCalo_MuonCoil`** (the same
geometry `k4ODD`'s CI `clone_ODD` test pins). Stock acts ODD does **not** work:
- v4.0.4 calorimeters set no `DetType` flags (so Pandora's `getExtension(CALORIMETER|...)`
  finds nothing under DD4hep 1.32.1);
- v6.0.2's muon is a *spectrometer* (`ODDMuonBarrel`) lacking the
  `dd4hep::rec::LayeredCalorimeterData` extension `DDGeometryCreator` hard-requires.

**Sim and reco must use the SAME geometry** so the calo cell-IDs decode across the handoff.
The reco image bakes azaborow at `/opt/odd-install`; the sim side builds it to CFS
(`stress_mu200/odd-azaborow-sim`) and `ddsim_run.py`/`digi_and_reco.py` honor the
`ODD_COMPACT_FILE` / `ODD_GEO_DIR` / `ODD_PATH` env overrides.

## Per-stage container routing

`scripts/cli/cli_utils.py::resolve_stage_container` reads `common.stage_containers` from
`env_setup.yaml`: `calo_digitization` + `pandora_reco` → the reco image; everything else
(gen/sim/digi/tracking + the uproot-based converters) → the sim image. Falls back to
`common.container` (the sim default). `job_submission.py` loads the right tarball per stage.

## Building the reco image

```bash
cd docker/colliderml-reco
podman-hpc build --jobs 4 -t colliderml/reco:<date> .
```
Build on the **login node** (the steps `git clone` + `apt`, which need internet; compute
nodes have none). It's light (4 small Pandora packages + k4ODD + ODD; the heavy
Gaudi/tracking stack is in the base). Steps (`steps/*.sh`, each run with explicit `bash`
because podman-hpc's builder uses dash and ignores `SHELL`):
1. `key4hep-shim.sh` — native activation (`source /opt/setup_spack.sh; spack load
   key4hep-stack`) + the source-built k4Reco prefix + a cvmfs-style merged-include `CPATH`.
2. `build_k4reco.sh` — **k4Reco v0.3.0** `GaudiTrkUtils` from source (the base's spack
   k4reco-0.2.1 has the wrong `GaudiDDKalTestTrack` API *and* ships no CMake export). Builds
   only `GaudiTrkUtils` (relaxes `find_package(k4FWCore 1.4)`; drops the `k4RecoPlugins`
   module that needs 1.4) and enables the commented-out config export.
3. `build_pandora.sh` — the calibrated stack (PandoraSDK / optimized LCContent /
   k4GaudiPandora / k4DetectorPerformance) at the pinned org-fork refs.
4. `build_k4odd.sh` — k4ODD plugins + the calibrated options.
5. `patch_k4fwcore.sh` — guards k4FWCore 1.3's `EventLoopMgr(Warnings=False)` (Gaudi 40
   dropped that property, which otherwise kills *every* `k4run`).
6. `build_odd.sh` — the azaborow ODD (kept LATE so re-bumping the ODD ref doesn't
   invalidate the cached Pandora layers).

## Running the pipeline

### Calo-only PF (neutral, no tracks)
```
ddsim(azaborow) → sim_edm4hep.root → pandora_reco (K4ODD_TRACK_COLLECTION=EmptyTracks)
```

### Charged PF (the production mode — needs ACTS tracks)
The proven recipe (testbed scripts in `<release-2>/testbed/scripts/`):
```
SIM container:
  ddsim (ODD_COMPACT_FILE=azaborow)                       -> sim.root
  testbed/scripts/acts_tracking.py  (ODD_PATH=azaborow)   -> sim_with_tracks.root
        # its PodioWriter(inputFrame="events") carries calo+MCParticles through
        # AND adds the ActsTracks collection in ONE frame — no merge step
  testbed/scripts/add_metadata_frames.py --with-tracks <f> --sim <sim>
        # REQUIRED: restores the calo cellID-encoding metadata the PodioWriter drops;
        # without it ECalBarrelDigi dies with "bad optional access"
RECO container:
  k4run ODDreconstruction.py --inputFile sim_with_tracks.root
        K4ODD_TRACK_CREATOR=DDTrackCreatorCLIC
        K4ODD_TRACK_COLLECTION=ActsTracks
        K4ODD_PANDORA_SETTINGS=.../PandoraSettingsCLD.xml
  -> GaudiPandoraPFOs / GaudiPandoraClusters / GaudiPandoraStartVertices
```
The reco env block (`env_setup.yaml`) sources, in order: the native key4hep shim → the
pandora stack `setup_stack.sh` → ODD `this_odd.sh` → exports `K4ODD_PATH`, `ODD_INSTALL_DIR`,
and `K4ODD_INSTALL/{lib,python}` on `LD_LIBRARY_PATH`/`PYTHONPATH`. `pandora_reco.py` always
passes an **absolute** `K4ODD_PANDORA_SETTINGS` (k4run `exec()`s the options file, so
`__file__`-relative defaults resolve to the wrong dir).

## Gotchas (hard-won)

- **Sim image goes ENOEXEC** ("cannot execute binary file" on every binary) when the
  podman-hpc overlay store fills from big image builds. Fix: `podman-hpc system prune -f`
  (reclaimed ~1.1 TB), then reload. NOT an architecture problem.
- **Old testbed `sim_with_tracks.root` fixtures** were written with a cvmfs podio that the
  reco image's **podio 1.4.1 cannot read** (`bad_function_call` on `readEntry`). Regenerate
  via the sim container instead of reusing fixtures.
- `DDTrackCreatorCLIC` emits forward-disk-extension WARNINGs for the endcap tracker — the
  known η≥2 endcap-tracking gap; non-fatal, PFOs are still produced.
- podio cross-version: sim writes podio 1.4.1 / edm4hep 0.99.2; reco reads with the same →
  no cross-version issue for sim-container-written files.

## Not yet wired (deliberate follow-ups, not blockers)

- Fold `acts_tracking.py` + `add_metadata_frames.py` into the production `digitization`
  stage so the CLI emits `sim_with_tracks.root` directly (currently a manual two-step).
- Arrow-parquet writing for the reco-side PFO/cluster tables in the full-CLI run.
- Rename the sim tag `colliderml/acts-arrow` → `colliderml-sim` and drop the aborted
  single-image Gaudi/tracking layers from `docker/colliderml-prod/`.
