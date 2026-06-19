# colliderml/reco — the reco container (Container B)

The key4hep-based image that runs `calo_digitization` + `pandora_reco` (the calibrated
Pandora particle flow). The sim half stays in `colliderml/acts-arrow`. See the full
architecture, recipe, and gotchas in [`../../docs/TWO_CONTAINER_PIPELINE.md`](../../docs/TWO_CONTAINER_PIPELINE.md).

## Build (on the login node — the steps need internet)

```bash
podman-hpc build --jobs 4 -t colliderml/reco:<date> .
```

FROM `ghcr.io/key4hep/key4hep-sim-reco-ubuntu24:reco-image-amd64` (no cvmfs: ROOT 6.36 +eve,
Gaudi 40, k4FWCore 1.3, podio 1.4.1, edm4hep 0.99.2, DD4hep 1.32.1, prebuilt
KalTest/DDKalTest/LCIO). Each `steps/*.sh` is COPYed in its own layer right before its RUN;
they're heavily commented with *why* each fix exists. Build order and rationale:

| Step | Why |
|------|-----|
| `key4hep-shim.sh` | native activation + source-built k4Reco prefix + cvmfs-style merged-include CPATH |
| `build_k4reco.sh` | k4Reco **v0.3.0** GaudiTrkUtils from source (base's spack k4reco is wrong API + no CMake export) |
| `build_pandora.sh` | PandoraSDK / LCContent / k4GaudiPandora / k4DetectorPerformance at pinned org-fork refs |
| `build_k4odd.sh` | k4ODD plugins + calibrated options |
| `patch_k4fwcore.sh` | guard `EventLoopMgr(Warnings=False)` — Gaudi 40 dropped it, kills every k4run |
| `build_odd.sh` | the calibrated `azaborow/addLayeredCalo_MuonCoil` ODD (kept last to preserve the Pandora cache) |

If the sim/reco images go ENOEXEC, the podman store is full → `podman-hpc system prune -f`.
