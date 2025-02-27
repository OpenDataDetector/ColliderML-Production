# ColliderML Development Repository

## Quickstart

``` bash
git clone https://github.com/OpenDataDetector/ColliderML
cd ColliderML
pip install -e .
```

Then run the pipeline as follows:

``` bash
python scripts/simulation/pythia_gen.py --config configs_development/testing_and_validation/quickstart/generation_test.yaml # Wait a minute or two
python scripts/simulation/merge_and_smear.py --config configs_development/testing_and_validation/quickstart/merge_smear_test.yaml # Wait a few seconds
python scripts/simulation/ddsim_run.py --config configs_development/testing_and_validation/quickstart/simulation_test.yaml # Wait up to 10 minutes
python scripts/simulation/digi_and_reco.py --config configs_development/testing_and_validation/quickstart/digitization_test.yaml # Wait a minute or two
```

Two low-pileup events will be generated, merged, smeared, simulated, digitized and reconstructed in the `outputs` subdirectory. You can also use these configs in batch, as detailed below.

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
- **Channel/Process**: Particle type, physics process, or challenge type.
- **Parameters**: Key configuration values (e.g., `theta`, `phi`, pileup level).
- **Run Metadata**: Timestamp and git commit hash.

Example:  
`FullDetector/ttbar_theta0.1_phi0.2_githash1234.hdf5`

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
