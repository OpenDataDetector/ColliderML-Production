# Dataset Structure for ColliderML

The ColliderML dataset is organized into four distinct spaces: **Simulation**, **Processing**, **Staging**, and **Public**. This document explains the storage structure, focusing primarily on the **Public** and **Simulation** spaces, which share a consistent directory layout. 

---

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
     - `mu` (single muon)
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

This structure ensures clarity, reproducibility, and ease of access, enabling efficient use of the ColliderML datasets for both development and research purposes.

