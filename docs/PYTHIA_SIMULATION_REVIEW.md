# ColliderML Pythia & Simulation Architecture Review

## Executive Summary

The ColliderML repository implements a comprehensive, production-ready event generation and detector simulation pipeline for high-energy physics. The system is designed to generate Beyond Standard Model (BSM) samples through multiple pathways (MadGraph, Pythia8, particle gun) and simulate them through the full detector chain using ACTS/DD4hep. The architecture emphasizes reproducibility, scalability, and seamless integration between generation and simulation stages.

**Key Strengths:**
- Modular, well-documented workflow with clear separation of concerns
- Flexible configuration system supporting both interactive and batch execution modes
- Strong reproducibility guarantees via git tracking and configuration snapshots
- Support for multiple physics processes and generation tools
- Integrated ACTS-based event merging for pileup handling

---

## 1. Architecture Overview

### 1.1 Pipeline Stages

The full simulation pipeline consists of 8 stages, organized into **generation** and **postprocessing** phases:

```
Generation Phase:
├── madgraph_init           → Generate and compile MadGraph process (matrix elements)
├── madgraph_generation     → Parallel HepMC event generation from compiled process
├── pythia_generation       → Pythia8 hard scatter and/or pileup generation
├── particlegun_generation  → Particle gun events for single-particle studies
└── merge_smear             → Event merging with ACTS HepMC3Reader + vertex smearing

Simulation Phase:
├── simulation              → DD4hep detector simulation (ddsim)
├── digitization            → Hit digitization and track reconstruction
├── calo_digitization       → Calorimeter digitization (in development)

Postprocessing Phase:
├── build_tracks            → Extract tracking detector hits
├── build_tracker_hits      → Tracker hit conversion
├── build_particles         → Particle-level reconstruction
├── build_manifest          → Dataset metadata generation
└── convert_all             → Complete conversion pipeline
```

### 1.2 Execution Modes

The system supports **three SLURM execution modes**:

| Mode | Purpose | Parallelization |
|------|---------|---|
| **interactive** | Development/testing on login nodes | None (serial) |
| **monolithic_slurm** | Single large job | Within single node |
| **distributed_slurm** | Multiple independent jobs | One job per SLURM task (array job) |
| **multi_node_slurm** | Single job spanning multiple nodes | Task farming across nodes |

---

## 2. Pythia8 Generation System (`scripts/simulation/pythia_gen.py`)

### 2.1 Architecture

The Pythia8 system implements a **three-phase workflow**:

1. **Hard Scatter Generation** (optional)
   - Generates signal events using Pythia8 with user-defined hard process
   - Output: `events_signal.hepmc3`
   - Respects `hard_process` and `pythia_settings` configuration

2. **Pileup Generation** (optional)
   - Generates individual pileup events for later merging
   - Uses Poisson sampling for event multiplicity
   - Output: `events_pileup.hepmc3`
   - Supports both deterministic and stochastic multiplicity (`poisson_sample`)

3. **Event Merging via ACTS** (optional)
   - Merges hard scatter and pileup using ACTS `HepMC3Reader`
   - Applies Gaussian vertex smearing during merge
   - Output: `merged_events.hepmc3`

### 2.2 Configuration Parameters

```yaml
# Generation parameters
events: 1000              # Events per run
seed: 42                  # Random seed (integer or string pattern)
pileup: 200               # Pileup multiplicity (events per hard scatter)

# Hard scatter definition
hard_process: "HardQCD:all"  # Pythia8 process (e.g., "Top:pair", "Higgs:gg2bbH")
pythia_settings:             # Additional Pythia8 settings
  - "PhaseSpace:pTHatMin = 10"
  - "PhaseSpace:pTHatMax = 1000"

# Vertex smearing (mm and ns)
vertex_sigma_xy: 0.0125  # Transverse smearing
vertex_sigma_z: 55.5     # Longitudinal smearing
vertex_sigma_t: 5.0      # Time smearing

# Pileup sampling
poisson_sample: false                  # Use Poisson sampling
poisson_buffer_sigma: 5.0              # Buffer for Poisson calculation
```

### 2.3 Workflow Determination

The script intelligently determines which phases to execute based on configuration:

```python
# Auto-detection logic:
should_generate_hard_scatter = bool(config.hard_process)
should_generate_pileup = config.pileup > 0
should_merge = both generated OR explicitly set
```

This allows flexible usage:
- **Pure Pythia**: hard_process + pileup → both generated and merged
- **MadGraph + Pythia**: hard_process=None, pileup > 0 → only pileup generated
- **Merge-only**: Existing files with `--merge` flag

### 2.4 Key Implementation Details

**Random Number Generation:**
```python
# Hard scatter: seed = config.seed
# Pileup: seed = config.seed + 1000
# Merge: seed = config.seed
```
Ensures reproducibility while giving different streams to signal/pileup.

**Vertex Smearing:**
- Applied during **merge phase only** (not during generation)
- Uses ACTS `GaussianVertexGenerator` with independent smearing per event
- Consistent across signal and pileup vertices

**File Auto-Detection:**
```python
# Hard scatter search order:
1. events_signal.hepmc3     (Pythia8)
2. events.hepmc3            (MadGraph)
3. events.hepmc             (MadGraph uncompressed)
4. events.hepmc.gz          (MadGraph compressed)

# Pileup search order:
1. events_pileup.hepmc3
```

---

## 3. MadGraph Generation System

### 3.1 Two-Stage Workflow

MadGraph uses a **two-job architecture** to minimize recomputation:

| Stage | Purpose | Cost | Parallelization |
|-------|---------|------|---|
| **madgraph_init** | Generate physics process, compile matrix elements | **HIGH** (1-4 hours) | Serial |
| **madgraph_generation** | Generate events from compiled process | **LOW** (minutes per job) | Embarrassingly parallel |

### 3.2 MadGraph Initialization (`scripts/simulation/madgraph_init.py`)

**Input Configuration:**
```yaml
dataset: "dihiggs"
version: "v1"
mg_base_path: "/path/to/MG5_aMC_v3_5_8"
generation_scratch_dir: "/scratch/mg5_temp"
mg_model: "sm-eft"          # Model (e.g., "sm", "mssm", "sm-eft")
mg_definitions: []          # Define blocks/parameters
mg_generate_command: "generate p p > h h > b b b b [QCD]"
```

**Process:**
1. Creates MG5 input script with model/process definitions
2. Runs `mg5_aMC` to generate process directory
3. Compiles matrix elements for all requested processes
4. Applies default card customizations (run_card.dat, shower_card.dat, pythia8_card.dat)
5. **Tarballs compiled process** for parallel distribution

**Output Structure:**
```
{dataset}/{version}/
├── madgraph_process.tgz          # Compiled process (tarball)
└── final_cards/
    ├── {process_name}_run_card.dat
    └── {process_name}_pythia8_card.dat
```

### 3.3 MadGraph Event Generation (`scripts/simulation/madgraph_gen.py`)

**Parallel Execution Model:**
- Each SLURM task gets unique job scratch directory
- Tarball extracted locally per task (safe concurrent access)
- Cards customized for run-specific parameters (events, seed)
- Events generated and split into runs

**Key Features:**

1. **Flexible Splitting:**
   ```yaml
   splitting_config:
     enable: true
     events_per_file: 64          # Split large outputs
     max_files_per_mg_run: 100    # Cap output size
     output_filename: "events.hepmc"
   ```
   - Prevents enormous single files
   - Creates run directories: `0/events.hepmc`, `1/events.hepmc`, etc.

2. **Card Customization Pattern:**
   - Base cards set during `madgraph_init`
   - Per-run customization: events, seed, run name
   - Supports multiple run modes:
     - **LO+MLM**: Uses `pythia8_card.dat` (Pythia8 shower + MLM JetMatching)
     - **NLO/FxFx**: Uses `shower_card.dat` (Herwig/Pythia8 shower)

3. **Multi-Node Coordination:**
   ```python
   # MadGraph run ID from SLURM_PROCID
   mg_run_id = int(args.output_subdir)
   
   # Staging directory (avoids collision with split output)
   staging_dir = runs/all/{mg_run_id}/
   
   # Split output directory
   split_base = runs/
   
   # Global run offset for proper indexing
   global_run_offset = mg_run_id * runs_per_job
   ```

**File Processing:**
- LHE files: Moved to staging directory
- HepMC files: Split into run directories (if enabled) or moved to staging
- Final cards: Copied to both run directory and central version directory

---

## 4. Event Merging System

### 4.1 ACTS-Based Merging (`scripts/simulation/merge_and_smear.py`)

The merging system reads separate signal and pileup HepMC files and combines them with vertex smearing.

**Implementation:**
```python
# Phase 1: Read signal and pileup files
signal_events = list(read_hepmc(signal_file))
pileup_events = list(read_hepmc(pileup_file))

# Phase 2: For each signal event, select N pileup events
for signal_event in signal_events:
    pileup_batch = select_pileup_events(pileup_events, n_pileup)
    
    # Phase 3: Merge and smear
    merged = merge_events(signal_event, pileup_batch, vertex_sigmas)
    write_merged(merged)
```

### 4.2 Vertex Smearing

Applied independently to all vertices in an event:

```python
def smear_vertex_position(event, vertex_sigmas):
    # Generate one smearing offset per event
    offset = [
        gaussian(0, sigma_xy),   # x
        gaussian(0, sigma_xy),   # y
        gaussian(0, sigma_z),    # z
        gaussian(0, sigma_t)     # time
    ]
    
    # Apply same offset to all vertices in event
    for vertex in event.vertices:
        vertex.position += offset
```

**Rationale:** Same vertex offset for all vertices in an event mimics detector resolution effects uniformly.

---

## 5. Configuration System

### 5.1 Three-Level Configuration Architecture

**Level 1: Environment Setup** (`scripts/cli/env_setup.yaml`)
```yaml
env_variables:
  ATLAS_LOCAL_ROOT_BASE: "/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase"
  LCG_VIEW: "LCG_107 x86_64-el9-gcc13-opt"
  DD4HEP_SETUP_SCRIPT: "/path/to/dd4hep/setup.sh"
  ACTS_SETUP_SCRIPT: "/path/to/acts/setup.sh"

config_defaults:
  common:
    output_base_dir: "/eos/user/.../simulation"
  madgraph:
    mg_base_path: "/path/to/MG5_aMC"
    generation_scratch_dir: "/scratch/mg5"
```

**Level 2: Stage Configuration** (e.g., `pythia_config.yaml`)
```yaml
campaign: "full_pileup_pilot"
dataset: "ttbar"
version: "v5"
stage: "pythia_generation"

job_config:
  n_runs: 128
  runs_per_node: 128
  execution_mode: "distributed_slurm"

events: 32
pileup: 200
hard_process: "Top:pair"
```

**Level 3: CLI Overrides**
```bash
python pythia_gen.py --config config.yaml \
  --events 100 \
  --seed 12345 \
  --output /custom/path
```

### 5.2 Variable Substitution

The system supports **dynamic path substitution** from env_setup.yaml:

```yaml
# In config file:
mg_base_path: "{madgraph.mg_base_path}"
generation_scratch_dir: "{madgraph.generation_scratch_dir}"

# At runtime: values from env_setup.yaml are substituted
```

Processed by `cli_utils.load_and_process_config()` before script execution.

### 5.3 Seed Management

**Seed Generation Pattern** (`utils/config.py`):
```python
# Option 1: Numeric seed (used directly, constrained to [1, 900000000])
seed = 42  → 42

# Option 2: String pattern (hashed to deterministic value)
seed = "ttbar_run{output_subdir}"  → hash(...) % 900000000

# Option 3: Default (time-based)
seed = int(time.time()) % 900000000
```

Ensures:
- Reproducibility with numeric or pattern-based seeds
- Uniqueness across parallel jobs
- Compatibility with Pythia8 seed range

---

## 6. Batch Submission System

### 6.1 Job Submission Pipeline (`scripts/cli/run_stage.py`)

```
User Input
    ↓
Load & Process Config (env setup, variable substitution)
    ↓
Git Commit & Config Snapshot
    ↓
Determine Execution Mode
    ↓
├─→ interactive: Run directly on login node
├─→ monolithic_slurm: Single SLURM job
├─→ distributed_slurm: Array job (n_runs tasks)
└─→ multi_node_slurm: Task farm across nodes
    ↓
[Optional] Validation + Error Guardian
```

### 6.2 Command Building (`cli_utils.build_stage_command()`)

Constructs final execution command based on mode and stage:

```python
# Base command
python scripts/simulation/{stage_script} \
  --config /path/to/config.yaml \
  --output /output/dir \
  --output-subdir {run_id}

# Add seed (derived from output_subdir)
--seed {dataset}_{version}_run{run_id}

# Shifter container (if stage requires it)
shifter --image=registry.cern.ch/atlasadc/atlas-grid-almalinux9 -- \
  {env_setup_commands} && {python_command}
```

### 6.3 Shifter Container Integration

**Stages requiring Shifter:**
- pythia_generation
- merge_smear
- simulation (ddsim)
- digitization
- calo_digitization

**Stages running on host (no shifter):**
- madgraph_init
- madgraph_generation
- particlegun_generation
- postprocessing

Rationale: MadGraph/postprocessing need specific user environments; simulation stages benefit from standardized CVMFS/DD4hep setup.

### 6.4 Git Integration

**Reproducibility Guarantee:**
```python
# Before stage execution:
1. Check git status for uncommitted changes
2. Commit all changes with descriptive message
3. Log git hash to output directory
4. Save config snapshot (expanded_config.yaml)

# Per-version tracking:
{dataset}/{version}/
├── .git_commit_success    # Timestamp + commit hash
└── expanded_config.yaml   # Full resolved configuration
```

Ensures later reproducibility: can reconstruct exact conditions from git hash + config snapshot.

---

## 7. Data Directory Structure

### 7.1 Storage Spaces

```
{output_base_dir}/
├── {dataset}/
│   └── {version}/
│       ├── software_snapshots/current/
│       │   └── [Git snapshot files]
│       ├── expanded_config.yaml
│       ├── .git_commit_success
│       │
│       └── madgraph_process/           (madgraph_init output)
│           ├── madgraph_process.tgz
│           └── final_cards/
│
├── simulation/
│   ├── {dataset}/
│   │   └── {version}/
│       ├── runs/
│       │   ├── 0/
│       │   │   ├── all/
│       │   │   │   ├── 0/
│       │   │   │   │   ├── events.hepmc3
│       │   │   │   │   └── final_cards/
│       │   │   │   └── 1/...
│       │   │   ├── events.hepmc3
│       │   │   ├── merged_events.hepmc3
│       │   │   └── simulation_output/
│       │   └── 1/...
```

### 7.2 File Naming Convention

All output files follow the pattern:
```
{space}/{dataset}/{version}/{object_type}/{particles/clusters}/
  {dataset}.{version}.{object_type}.{particle_type}.events{N}-{M}.{format}
```

Example:
```
simulation/ttbar/v1/truth/particles/
  ttbar.v1.truth.particles.events0-999.hepmc3

processed/ttbar/v1/reco/tracks/
  ttbar.v1.reco.tracks.events0-999.parquet
```

---

## 8. Key Utilities and Helpers

### 8.1 Configuration (`scripts/simulation/utils/config.py`)

- `create_base_parser()`: Standard argument parser for all scripts
- `load_config()`: Merge YAML config with CLI overrides
- `hash_seed_string()`: Convert seed patterns to Pythia8-compatible integers

### 8.2 MadGraph Utilities (`scripts/simulation/utils/madgraph_utils.py`)

- `run_command()`: Execute subprocess with optional streaming/capture
- `customize_card_with_regex()`: Modify MadGraph card files
- `get_version_directory_path()`: Construct version-specific paths

### 8.3 Logging (`scripts/simulation/utils/app_logging.py`)

- `setup_logging()`: Consistent logging configuration
- `TimingRecorder`: Performance measurement and reporting

---

## 9. Physics Processes Supported

### 9.1 Pythia8-Based Processes

Can be specified via `hard_process` parameter:

```yaml
# Standard Model
"Top:pair"          # ttbar
"Higgs:gg2bbH"      # Higgs production
"HardQCD:all"       # QCD hard scattering
"WeakBoson:all"     # W/Z production

# BSM Examples (requires appropriate model)
"SUSY:gg2gluinogluino"
"ComposittnessLL:all"
```

### 9.2 MadGraph-Based Processes

Specified in config via `mg_generate_command`:

```yaml
# LO processes
"generate p p > t t~ [QCD]"
"generate p p > h h > b b b b [QCD]"

# NLO processes
"generate p p > t t~ @NLO"
"generate p p > v v @NLO"

# With matching/merging
"generate p p > j j [QCD]=[MLM]"  # LO+MLM
"generate p p > j j [QCD]=[FxFx]" # NLO+FxFx
```

Requires appropriate model (sm, mssm, etc.) specified in `mg_model`.

---

## 10. Current Implementation Status

### 10.1 Fully Implemented ✓

- [x] Pythia8 generation (hard scatter + pileup)
- [x] MadGraph process generation and event generation
- [x] ACTS-based event merging with vertex smearing
- [x] DD4hep simulation (ddsim)
- [x] Digitization and reconstruction
- [x] Batch job submission (distributed/monolithic/multi-node)
- [x] Configuration management and git tracking
- [x] Performance monitoring

### 10.2 In Development

- [ ] Calorimeter digitization (`calo_digitization.py`)
- [ ] Advanced guardian error policies
- [ ] Real-time validation during simulation

### 10.3 Not Yet Implemented

- Systematic uncertainty propagation
- Advanced event filtering/selection
- Detector alignment variation studies
- Cross-section calculation integration

---

## 11. Notable Features for BSM Studies

### 11.1 Flexible Process Definition

**Pythia8 Approach:**
```yaml
hard_process: "SUSY:gg2gluinogluino"
pythia_settings:
  - "SLHA:file = /path/to/susy.slha"
  - "Squark:mass = 2000"
  - "Gluino:mass = 2500"
```

**MadGraph Approach:**
```yaml
mg_model: "mssm"
mg_generate_command: "generate p p > go go > t1 t1* t1 t1*"
```

### 11.2 Pileup Flexibility

- **Fixed multiplicity**: `pileup: 200` (200 events per hard scatter)
- **Poisson sampling**: `poisson_sample: true` with `poisson_buffer_sigma: 5.0`
  - Generates extra pileup events for Poisson sampling
  - Prevents edge effects in pileup distribution

### 11.3 Vertex Smearing for Detector Effects

Three independent smearing parameters:
```yaml
vertex_sigma_xy: 0.0125  # Transverse (beamspot, ~125 μm)
vertex_sigma_z: 55.5     # Longitudinal (bunch length, ~55 mm)
vertex_sigma_t: 5.0      # Time (bunch separation effects, ~5 ns)
```

Can be tuned to match specific detector configurations.

### 11.4 Seed Management for Reproducibility

```bash
# Reproducible BSM study 1
python pythia_gen.py --config bsm_sample.yaml --seed 42

# Reproducible BSM study 2 (different random number stream)
python pythia_gen.py --config bsm_sample.yaml --seed 43

# Pattern-based (useful for farm jobs)
python pythia_gen.py --config bsm_sample.yaml --seed "susy_run_{SLURM_PROCID}"
```

---

## 12. Recommendations for BSM Simulations

### 12.1 Configuration Best Practices

```yaml
# Explicit process definition
campaign: "bsm_pilot"
dataset: "gluino_pair"
version: "v1"

# Clear physics process
stage: "pythia_generation"
hard_process: "SUSY:gg2gluinogluino"
pythia_settings:
  - "1000021:m0 = 2500"     # Gluino mass in GeV
  - "SLHA:file = /path/to/susy.slha"  # SUSY masses

# Reproducibility
seed: 42
events: 10000
pileup: 200

# Detector effects
vertex_sigma_xy: 0.0125
vertex_sigma_z: 55.5
vertex_sigma_t: 5.0

# Batch configuration
job_config:
  n_runs: 100
  runs_per_node: 1
  execution_mode: "distributed_slurm"
  time_limit: "01:00:00"
  qos: "regular"
```

### 12.2 Workflow Example: Dark Matter Pair Production

```bash
# Step 1: Configure process
cat > dm_pair.yaml << 'EOF'
campaign: "dm_studies"
dataset: "dm_pair"
version: "v1"
stage: "pythia_generation"

hard_process: "DarkMatter:pair"
events: 5000
pileup: 200

vertex_sigma_xy: 0.0125
vertex_sigma_z: 55.5
vertex_sigma_t: 5.0

job_config:
  n_runs: 50
  runs_per_node: 1
  execution_mode: "distributed_slurm"
EOF

# Step 2: Run generation
python scripts/cli/run_stage.py dm_pair.yaml

# Step 3: Simulate through detector
python scripts/cli/run_stage.py dm_pair_simulation.yaml

# Step 4: Reconstruct
python scripts/cli/run_stage.py dm_pair_digitization.yaml
```

### 12.3 Performance Considerations

**For large MadGraph processes (many Feynman diagrams):**
- Allocate more time for `madgraph_init` (may take 2-4 hours)
- Use reasonable `nb_core` in MG5 initialization
- Keep `events_per_file` moderate (64-128) for manageable file sizes

**For pileup-heavy samples:**
- Pre-generate pileup in separate step
- Reuse same pileup file across multiple signal samples
- Use Poisson sampling to reduce generated pileup overhead

**Memory usage:**
- Pythia8: ~1-2 GB per process
- DD4hep simulation: ~2-4 GB per process
- MadGraph: ~4-8 GB during initialization

---

## 13. Troubleshooting Guide

### Common Issues and Solutions

| Issue | Likely Cause | Solution |
|-------|---|---|
| Seed out of range | String seed too large | Use numeric seed or check hashing |
| Pileup generation fails | Insufficient temp space | Check `generation_scratch_dir` quota |
| ACTS merge crashes | Incompatible HepMC versions | Verify pyhepmc compatibility |
| MadGraph process timeout | Complex process, insufficient time | Increase `time_limit` in config |
| File not found errors | Path substitution failed | Check env_setup.yaml variables |

---

## 14. Code Organization Summary

```
scripts/
├── cli/                           # Job submission and stage orchestration
│   ├── run_stage.py              # Main entry point
│   ├── job_submission.py         # SLURM job handling
│   ├── multi_config_job.py       # Multi-stage pipelines
│   └── cli_utils.py              # Shared CLI utilities
│
├── simulation/                    # Event generation and detector simulation
│   ├── pythia_gen.py             # Pythia8 generation + ACTS merging
│   ├── madgraph_init.py          # Process compilation
│   ├── madgraph_gen.py           # Event generation
│   ├── merge_and_smear.py        # Alternative merging (legacy)
│   ├── ddsim_run.py              # DD4hep simulation
│   ├── digi_and_reco.py          # Digitization
│   ├── particlegun_gen.py        # Single particle generation
│   └── utils/
│       ├── config.py             # Configuration handling
│       ├── madgraph_utils.py     # MadGraph utilities
│       └── app_logging.py        # Logging infrastructure
│
└── postprocessing/               # Data conversion and reconstruction
    ├── convert_all.py            # Master conversion script
    ├── convert_particles.py      # Particle-level conversion
    ├── convert_tracks.py         # Track conversion
    └── build_manifest.py         # Metadata generation
```

---

## 15. References and Documentation

**Key Configuration Files:**
- Production configs: `configs_production/`
- Development configs: `configs_development/`
- Environment template: `scripts/cli/env_setup.yaml.template`

**Example Workflows:**
- `scripts/simulation/example_usage.md` - Pythia/MadGraph workflows
- `scripts/cli/README.md` - Job submission details

**Physics Documentation:**
- Pythia8: [pythia.org](https://pythia.org)
- MadGraph: [launchpad.net/madgraph5](https://launchpad.net/madgraph5)
- ACTS: [acts.readthedocs.io](https://acts.readthedocs.io)
- DD4hep: [dd4hep.cern.ch](https://dd4hep.cern.ch)

---

## 16. Future Development Opportunities

1. **Advanced BSM Support**
   - Automated grid scanning for parameter space exploration
   - Integration with constraint tools (HiggsBounds, CheckMate)

2. **Performance Optimization**
   - GPU acceleration for detector simulation
   - Cached pileup library for faster resampling

3. **Analysis Integration**
   - Direct HDF5/Parquet export from simulation
   - Analysis-ready selection filters

4. **Validation Framework**
   - Cross-section validation against analytical predictions
   - Detector acceptance/efficiency studies

---

**Document Version:** 1.0  
**Last Updated:** 2025-01-19  
**Repository:** ColliderML Development  
**Focus:** Pythia8/MadGraph-based BSM Simulation Architecture

