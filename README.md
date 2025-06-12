# ColliderML Development Repository

## Quickstart

``` bash
git clone https://github.com/OpenDataDetector/ColliderML
cd ColliderML
conda create -n collider-env python=3.10
conda activate collider-env
pip install -e .
```
We assume that you have access to `cvmfs` and therefore also LCG views.

Before running the pipeline, you need to point to the correct environment setup directories, in `scripts/cli/env_setup.yaml`. These commands will be called by `run_stage.py`, so make sure `software_dir` points to the path that contains this repository, as well as the other paths for cvmfs, lcg views, dd4hep, acts, etc.


Then run the pipeline as follows:
```bash
python scripts/cli/run_stage.py configs_production/full_pileup_pilot/ttbar/madgraph_config.yaml
```

You will be prompted to be on the right branch (in this case `git checkout -b campaign/full_pileup_pilot/dataset/ttbar`), since every run is git committed. This stage generates N hard-scatter ttbar events, and splits them across `events_per_file` files (if splitting is enabled).

```bash
python scripts/cli/run_stage.py configs_production/full_pileup_pilot/ttbar/pythia_config.yaml
```

This stage runs pythia generation of pileup, using the ACTS examples framework.

```bash
python scripts/cli/run_stage.py configs_production/full_pileup_pilot/ttbar/simulation_config.yaml
```

This stage runs DDSim, using the DD4HEP framework.

```bash
python scripts/cli/run_stage.py configs_production/full_pileup_pilot/ttbar/digitization_config.yaml
```

This stage runs digitization, using the ACTS examples framework.


## Batch mode

Simply changing `execution_mode` to `distributed_slurm` in the config file will run the pipeline in batch mode.
---

## Development Repository

### 1. Simulation Scripts (`scripts/simulation/`)
All scripts needed for simulation are stored in the `scripts/simulation` directory. They follow the pipeline:

1. **Pythia parton generation** (`pythia_gen.py`): Generates signal and pileup events using Pythia8
2. **Event merging and smearing** (`merge_and_smear.py`): Combines signal and pileup, applies vertex smearing
3. **Detector simulation** (`ddsim_run.py`): Simulates detector response using DD4hep
4. **Digitization and reconstruction** (`digi_and_reco.py`): Performs hit digitization, calo digitization (COMING SOON) and track reconstruction

Since this is a simple sequence, we avoid complex workflow management tools like Snakemake, and instead run the scripts as follows:

**Interactive mode**
```
# For example, to run the Pythia generation script
python scripts/simulation/pythia_gen.py --config configs_production/ttbar/generation_test.yaml
```

**Batch mode**
```
python batch/job_submission.py configs_production/ttbar/generation_test.yaml
```

Observe that the same config file can be used for interactive and batch modes. Batch mode configs simply have more options set, such as `job_config` options.

### 2. Configuration Files
- `configs_development/`: Development and testing configurations
  - `testing_and_validation/`: Configs for testing pipeline components
  - `parallel_tests/`: Configs for performance scaling studies
- `configs_production/`: Production-ready configurations for dataset generation

### 3. Batch Processing (`batch/`)
- Scripts for running on HPC systems
- SLURM job submission templates
- Parallel execution configurations

### 4. Analysis Tools (`scripts/analysis/`)
- Tools for analyzing simulation outputs
- Performance measurement utilities
- Data quality validation scripts

### 5. Utilities (`scripts/simulation/utils/`)
- Common utilities used across scripts:
  - Configuration handling
  - Logging setup
  - Performance monitoring

### 6. Development Tools (`notebooks/`)
- Jupyter notebooks for:
  - Pipeline development and testing
  - Output visualization and validation
  - Performance analysis

### 7. Installation
- `setup.py`: Package installation configuration
- Dependencies and environment setup

## Storage Spaces Overview

### 1. Simulation Space
- **Purpose**: Permanent, immutable storage of raw simulation outputs, including all truth information.
- **Key Features**:
  - Each dataset is stamped with its configuration and the git commit used to generate it.
  - Files here should never be deleted or modified.

### 2. Processing Space
- **Purpose**: Flexible, mutable space for intermediate computations, testing, and derivations.
- **Key Features**:
  - Subdivided into processing stages for modular workflows.
  - Temporary data can be modified or deleted as needed.

### 3. Staging Space
- **Purpose**: Semi-stable storage for final validation and quality checks before data is made public.
- **Key Features**:
  - Reviewed for completeness and correctness.
  - Serves as the last step before public release.

### 4. Public Space
- **Purpose**: Final, user-accessible, and append-only storage of curated datasets.
- **Key Features**:
  - Files are immutable once added.
  - Organized for ease of navigation and reproducibility.

---

## Directory Structure for Public and Simulation Spaces

The **Public** and **Simulation** spaces follow the same directory structure, organized hierarchically for clarity and ease of access. The structure is as follows:

### Top-Level Directories

1. **SingleParticle**
   - Contains datasets focused on isolated particle studies.
   - Subdirectories for each particle type:
     - `single_muon`
     - `single_pion`
     - `single_electron`
     - Other particles as required.

2. **Tracker**
   - Dedicated to tracker subsystem simulations with two configurations:
     - `pileup200`: Regular detector configuration.
     - `pileup400`: Maximized detector configuration.
   - Within each pileup configuration:
     - Subdirectories organized by channels, such as:
       - `ttbar`
       - `zplusjet`
       - `zprime`
       - `SUSY`
       - `Higgs`
       - Additional channels as needed.

3. **FullDetector**
   - Comprehensive datasets simulating the entire detector.
   - Subdirectories organized by physics processes:
     - `ttbar`
     - `zjets`
     - `higgs`
     - Other physics-driven categories as required.

4. **Challenge**
   - Special datasets simulating unique or anomalous scenarios.
   - Subdirectories for each challenge type:
     - `misalignment` (detector misalignments)
     - `miscalibration` (e.g., head modules)
     - `anomalies` (data with rare or unusual features)
     - Additional challenges as they arise.

---

## File Naming and Metadata

### File Naming Convention
All files across directories adhere to a structured naming scheme to ensure traceability and reproducibility:
1. **Storage Space**: `simulation`, `processing`, `staging`, or `public`
2. **Dataset**: `single_particle`, `pileup_10`, `pileup_200`
3. **Channel/Process**: Particle type, physics process, or challenge type, e.g. `ttbar`, `susy`
4. **Version**: `v1`, `v2`, etc.
5. **Object Type**: `truth`, `reco`, `measurement`
6. **Object**: `particles`, `tracks`, `caloclusters`, etc.

Then the file name again follows this convention, to absolutely ensure that files are traceable. It finally includes the event range.

Example:  
`public/pileup-10/ttbar/v1/reco/tracks/pileup-10.ttbar.v1.reco.tracks.events0-999.h5`

### Metadata
Each dataset includes accompanying metadata files containing:
- Simulation configuration details.
- Git commit hash for reproducibility.
- Timestamp of dataset generation.

---

## Principles for Dataset Management

1. **Consistency**: The directory structure is mirrored across all storage spaces, with additional subdirectories in the **Processing** and **Staging** spaces for intermediate steps.
2. **Immutability**: Data in the **Simulation** and **Public** spaces is never modified or deleted once created.
3. **Reproducibility**: Every dataset is linked to its generation configuration and version control state via metadata.
4. **Scalability**: The structure supports adding new datasets and channels without disrupting existing organization.

---
