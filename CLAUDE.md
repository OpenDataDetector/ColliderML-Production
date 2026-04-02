# ColliderML Production - Docker Pipeline Guide

## Quick Start

```bash
# Run the full Higgs portal pipeline
./scripts/cli/run_pipeline_docker.sh --channel higgs_portal

# Run the full ttbar pipeline
./scripts/cli/run_pipeline_docker.sh --channel ttbar

# Run a single stage
./scripts/cli/run_docker.sh simulation/pythia_gen.py \
    configs_development/docker_test/higgs_portal/pythia_config.yaml
```

## Docker Container

**Image:** `ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0`

Contains: MadGraph 5.3.9, Pythia 8.313, DD4hep (ddsim), Geant4 11.3, ROOT 6.38, HepMC3 3.3, ACTS (main).

### Key scripts

| Script | Purpose |
|--------|---------|
| `scripts/cli/run_pipeline_docker.sh` | Run a complete pipeline (all stages) |
| `scripts/cli/run_docker.sh` | Run a single stage in Docker |
| `scripts/cli/setup_container_env.sh` | Environment setup inside the container (sourced automatically) |

### Environment details

- **Python:** System python3.12 is used. ACTS bindings (.cpython-313) are symlinked under `/tmp/acts` so `import acts` works.
- **ODD:** OpenDataDetector v4.0.4 from CERN GitLab (`gitlab.cern.ch/acts/OpenDataDetector`). Cloned on first run and cached in `.cache/odd-v4/`. The factory library (`libOpenDataDetector.so`) is built automatically.
- **Geant4 data:** ~2 GB of physics datasets, auto-downloaded and cached in `.cache/g4data/`.
- **Pip packages:** `pyarrow`, `uproot`, `pandas`, `awkward`, `h5py`, `tqdm`, `pyhepmc`, `psutil` — installed to `.cache/pip/` on first run.
- **No network inside container:** The `run_docker.sh` / `run_pipeline_docker.sh` scripts clone ODD on the host before launching Docker. Inside the container, `setup_container_env.sh` handles building ODD and downloading G4 data.

### .cache directory

Persistent cache at `<repo>/.cache/` (gitignored). Contains:
```
.cache/
  odd-v4/              # ODD v4.0.4 source (XML geometry, data)
  odd-v4-install/      # Built libOpenDataDetector.so
  g4data/              # Geant4 physics datasets (~2 GB)
  pip/                 # Python packages for postprocessing
```

## Pipeline Stages

### Higgs Portal (Pythia-only generation)

Configs: `configs_development/docker_test/higgs_portal/`

| # | Stage | Config | Script | Time (10 evts) |
|---|-------|--------|--------|----------------|
| 1 | Pythia generation | `pythia_config.yaml` | `simulation/pythia_gen.py` | ~8s |
| 2 | DDSim simulation | `simulation_config.yaml` | `simulation/ddsim_run.py` | ~10 min |
| 3 | Digi + Reco | `digitization_config.yaml` | `simulation/digi_and_reco.py` | ~11s |
| 4 | Parquet conversion | `convert_all.yaml` | `postprocessing/convert_all.py` | ~5s |

### ttbar (MadGraph + Pythia)

Configs: `configs_development/docker_test/ttbar/`

| # | Stage | Config | Script | Time (est.) |
|---|-------|--------|--------|-------------|
| 1 | MadGraph init | `madgraph_init_config.yaml` | `simulation/madgraph_init.py` | ~5-20 min |
| 2 | MadGraph gen | `madgraph_generation_config.yaml` | `simulation/madgraph_gen.py` | ~1-2 min |
| 3 | Pythia generation | `pythia_config.yaml` | `simulation/pythia_gen.py` | ~10s |
| 4 | DDSim simulation | `simulation_config.yaml` | `simulation/ddsim_run.py` | ~10 min |
| 5 | Digi + Reco | `digitization_config.yaml` | `simulation/digi_and_reco.py` | ~11s |
| 6 | Parquet conversion | `convert_all.yaml` | `postprocessing/convert_all.py` | ~5s |

## Expected Output

```
output/runs/0/
  merged_events.hepmc3       # Pythia output (merged signal + pileup)
  events.hepmc3              # MadGraph output (ttbar only)
  edm4hep.root               # DDSim output (~220 MB for 10 events)
  measurements.root          # Digitized tracker hits
  particles.root             # Truth particles
  tracksummary_ambi.root     # Reconstructed tracks
```

## ACTS API Notes (container version)

The ACTS build in this container uses the main branch (version 999.999.999). Key differences from released versions:

- `RootParticleWriter` / `RootMeasurementWriter` → `acts.examples.root` module
- `PodioReader` → `acts.examples.edm4hep` module  
- `addCounter(geoIds, min, max)` instead of `addCounter(geoIds, count)`

## Troubleshooting

- **"ODD_PATH not set":** The setup script handles this. If manual: `export ODD_PATH=<repo>/.cache/odd-v4`
- **"libOpenDataDetector.so not found":** Rebuild: `cd .cache && cmake odd-v4 -DCMAKE_INSTALL_PREFIX=odd-v4-install && make install`
- **"No module named 'acts'":** The setup script symlinks ACTS Python bindings. Check `/tmp/acts` exists.
- **Geant4 data missing:** Run `download_geant4_datasets.sh` inside the container, or populate `.cache/g4data/`.
- **"No module named 'pyedm4hep'":** Add `pyedm4hep` to the pip install list in `setup_container_env.sh`.

## Container Image Fix Checklist

The current container (`ghcr.io/opendatadetector/sw:0.2.2`) has issues that need to be fixed
in the next image build. These are documented here for reference.

### 1. Missing `bc` command (blocks MadGraph shower)

MadGraph checks for `bc` to calculate shower parameters. Without it, MadGraph falls
back to `noshower` mode and only produces LHE files (no HepMC output).

**Fix:** Add to Dockerfile or spack environment:
```dockerfile
RUN apt-get update && apt-get install -y bc
```

### 2. mg5amc_py8_interface incompatible with Pythia 8.3+ (blocks ttbar NLO+PS)

The container ships Pythia 8.313, but the `mg5amc_py8_interface` (C++ driver MadGraph
uses to steer Pythia8 showering) only works with Pythia 8.2.x. This is a
[known upstream issue](https://gitlab.com/Pythia8/releases/-/issues/24).

**Impact:** ttbar NLO events cannot be showered inside MadGraph. Only LHE output is
produced, which the downstream Pythia merge stage cannot read (expects HepMC).

**Fix options (choose one):**
1. **Pre-install the interface with `--pythia8_makefile` flag** — compiles the interface
   to use dynamic HepMC2 linking instead of static. Requires HepMC2 (`libHepMC.so`)
   at runtime. The container already has HepMC2 with both static and dynamic libs.
   ```bash
   cd /path/to/MG5aMC_PY8_interface
   python compile.py /pythia8/prefix --pythia8_makefile
   ```
2. **Add Pythia 8.2.x alongside 8.313** — Install a second Pythia (e.g., 8.245) via
   spack for MadGraph use. Set `pythia8_path` in `mg5_configuration.txt` to point to it.
3. **Port the interface to Pythia 8.3 API** — The Pythia team has low priority on this,
   but it's the cleanest long-term fix.

**Workaround references:**
- [MG5 Launchpad: HepMC2 static linking](https://answers.launchpad.net/mg5amcnlo/+question/693533)
- [Pythia GitLab: MG5-PY8 interface issue](https://gitlab.com/Pythia8/releases/-/issues/24)
- [MG5 Launchpad: Pythia8 path config](https://answers.launchpad.net/mg5amcnlo/+question/709602)

### 3. Missing `pyedm4hep` pip package (blocks parquet conversion)

The `convert_all.py` postprocessing script requires `pyedm4hep` which is a pip package.

**Fix:** Add to the pip install in the container or in `setup_container_env.sh`:
```bash
pip install pyedm4hep
```
Already added to `setup_container_env.sh` pip install list.

### 4. ACTS Python bindings path issue

ACTS installs Python files in `<prefix>/python/` but they must be importable as
`import acts`. The setup script works around this by symlinking `python/` → `/tmp/acts`.

**Fix:** The ACTS spack package should install to `<prefix>/lib/pythonX.Y/site-packages/acts/`
instead of `<prefix>/python/`. Or add a `acts.pth` file.
