# ColliderML Production

## Repo boundary

User-facing code (the `colliderml` pip package, benchmark tasks, HuggingFace Spaces, and documentation) lives in the **public repo**: [`OpenDataDetector/ColliderML`](https://github.com/OpenDataDetector/ColliderML).

**This repo** (`colliderml-production`) is the home for:
- Pipeline scripts (`simulation/`, `postprocessing/`, `scripts/`)
- FastAPI backend service (`backend/`)
- Container image workflow (`Dockerfile`, `.devcontainer/`)
- SFAPI runner and sbatch templates
- Admin dashboard (`spaces/colliderml-admin/`)
- Development and production configs (`configs_development/`, `configs_production/`)

The public library's `colliderml.simulate` module auto-clones this repo at a pinned ref (`production_ref` in its `colliderml/simulate/pipeline_config.yaml`) for local Docker/Podman simulations.

---

# Docker Pipeline Guide

The full chain (generation, Geant4, ACTS tracking and vertexing, Pandora particle
flow, parquet tables) runs in two containers on one machine. The guide is
[`docs/DOCKER_PIPELINE.md`](docs/DOCKER_PIPELINE.md): stages, images, options,
measured timings and the podman-hpc notes.

```bash
# Full chain, 10 ttbar events without pile-up
./scripts/cli/run_pipeline_docker.sh --channel ttbar --events 10

# Pythia-only channel
./scripts/cli/run_pipeline_docker.sh --channel higgs_portal

# A single stage
./scripts/cli/run_docker.sh simulation/pythia_gen.py \
    configs_development/docker_test/higgs_portal/pythia_config.yaml
```

| File | Purpose |
|------|---------|
| `scripts/cli/run_pipeline_docker.sh` | all stages of a channel |
| `scripts/cli/run_docker.sh` | one stage; picks the sim or reco image |
| `scripts/cli/docker_images.env` | the two image names |
| `scripts/cli/docker_stage_env.sh` | environment inside the container (`sim` or `reco`) |
| `scripts/cli/setup_container_env.sh` | sim-image environment, shared with production |
| `scripts/cli/stage_timer.py` | wall time and peak memory per stage |
| `configs_development/docker_test/<channel>/` | stage configs (container paths only) |

## ACTS API Notes (container version)

The ACTS build in this container uses the main branch (version 999.999.999). Key differences from released versions:

- `RootParticleWriter` / `RootMeasurementWriter` → `acts.examples.root` module
- `PodioReader` → `acts.examples.edm4hep` module  
- `addCounter(geoIds, min, max)` instead of `addCounter(geoIds, count)`
