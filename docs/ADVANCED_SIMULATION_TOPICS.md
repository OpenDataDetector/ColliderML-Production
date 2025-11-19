# Advanced ColliderML Simulation Topics

This document provides deeper technical details on specialized simulation topics, focusing on capabilities useful for BSM physics studies.

## Table of Contents

1. [Event Merging Algorithm Details](#1-event-merging-algorithm-details)
2. [Seed Management & Reproducibility](#2-seed-management--reproducibility)
3. [Multi-Node Job Coordination](#3-multi-node-job-coordination)
4. [Splitting and Distribution Strategies](#4-splitting-and-distribution-strategies)
5. [Card Customization Patterns](#5-card-customization-patterns)
6. [Batch Job Architecture](#6-batch-job-architecture)
7. [Performance Analysis](#7-performance-analysis)
8. [Troubleshooting Complex Scenarios](#8-troubleshooting-complex-scenarios)

---

## 1. Event Merging Algorithm Details

### 1.1 ACTS HepMC3Reader Merging

The system supports **two merging approaches**:

#### Approach 1: ACTS-Native Merging (Current)
```python
# pythia_gen.py (lines 315-336)
inputs = [
    HepMC3Reader.Input.Fixed(hard_scatter_file, 1),
    HepMC3Reader.Input.Fixed(pileup_file, pileup_multiplicity)
]

s.addReader(
    HepMC3Reader(
        inputs=inputs,
        vertexGenerator=vtxGen,  # Gaussian vertex smearing
        numEvents=config.events,
    )
)
```

**Advantages:**
- Integrated with ACTS framework
- Consistent with downstream simulation
- Native Poisson sampling support

**Process:**
1. Read one hard-scatter event
2. Select N pileup events (N = fixed or Poisson sample)
3. Apply vertex smearing (same offset to all vertices in event)
4. Merge particle lists
5. Output single merged event

#### Approach 2: Standalone Merging (Legacy)
```python
# merge_and_smear.py (alternative implementation)
def merge_hepmc_files(signal_path, pileup_path, output_path, vertex_sigmas):
    """Pure pyhepmc-based merging without ACTS"""
```

Uses pyhepmc directly for merging without ACTS infrastructure.

### 1.2 Poisson Sampling Algorithm

**Motivation:** Realistic pileup distribution has Poisson-distributed multiplicity, not fixed.

**Implementation:**
```python
# pythia_gen.py (lines 196-210)

# Step 1: Calculate needed pileup events
mu = pileup_multiplicity  # Expected value
z_sigma = 5.0  # Buffer in standard deviations
expected = config.events * mu
total_pileup_events = ceil(expected + z_sigma * sqrt(expected))

# Step 2: During merge, ACTS samples Poisson
if poisson_sample:
    inputs.append(HepMC3Reader.Input.Poisson(pileup_file, float(mu)))
```

**Effect on Storage:**
```
Fixed multiplicity (mu=200):
- Need exactly: 1000 * 200 = 200,000 pileup events

Poisson sampling (mu=200, buffer=5σ):
- Need: 1000*200 + 5*sqrt(1000*200) ≈ 200,000 + 7,071 ≈ 207,071 events
- Excess provides buffer for Poisson fluctuations
```

### 1.3 Vertex Smearing Mechanics

**Single Offset per Event:**
```python
# merge_and_smear.py (lines 52-57)
offset = np.array([
    np.random.normal(0, vertex_sigmas['xy']),  # x displacement
    np.random.normal(0, vertex_sigmas['xy']),  # y displacement
    np.random.normal(0, vertex_sigmas['z']),   # z displacement
    np.random.normal(0, vertex_sigmas['t'])    # time offset
])

# Apply same offset to ALL vertices in event
for vertex in event.vertices:
    vertex.position += offset
```

**Interpretation:**
- Single offset ≈ global detector misalignment or beamspot uncertainty
- Per-vertex variation would model resolution effects (more expensive)
- Magnitude tunable to match detector specifications

**Typical Values** (Open Data Detector):
```yaml
vertex_sigma_xy: 0.0125  # 125 μm transverse (beamspot)
vertex_sigma_z: 55.5     # 55.5 mm longitudinal (bunch length)
vertex_sigma_t: 5.0      # 5 ns (bunch timing)
```

---

## 2. Seed Management & Reproducibility

### 2.1 Seed Propagation Across Stages

```
User provides: seed = 42 (or pattern)
        ↓
pythia_gen.py:
    hard_scatter:  seed = 42
    pileup:        seed = 42 + 1000 = 1042
    merge:         seed = 42
        ↓
ddsim_run.py:
    simulation:    seed = 42 (from config)
        ↓
digi_and_reco.py:
    reconstruction: seed = 42
```

**Code References:**
```python
# pythia_gen.py (lines 158, 214)
rnd_hard = RandomNumbers(seed=config.seed)
rnd_pileup = RandomNumbers(seed=(config.seed or int(time.time())) + 1000)

# pythia_gen.py (line 311)
rng = RandomNumbers(seed=config.seed or 42)
```

### 2.2 Seed String Pattern Resolution

**Pattern Format:** `{variable}_{variable}` or environment variable references

**Resolution Mechanism** (`utils/config.py:14-44`):
```python
def hash_seed_string(seed_str):
    """Convert seed patterns to deterministic integers"""
    
    # Case 1: Pure numeric string
    if seed_str.isdigit():
        return int(seed_str)
    
    # Case 2: String pattern (hash to integer)
    hash_bytes = hashlib.md5(seed_str.encode()).digest()[:4]
    seed = int.from_bytes(hash_bytes, 'big', signed=True)
    
    # Constrain to Pythia8 range [1, 900000000]
    seed = abs(seed) % 900000000
    return seed if seed != 0 else 1
```

**Examples:**
```bash
# Numeric seed
--seed 42 → 42

# String pattern
--seed "ttbar_run_0" → hash("ttbar_run_0") → 123456789

# Environment variable (evaluated at shell level)
--seed "$SLURM_JOB_ID:$SLURM_PROCID" → shell expands → hash result

# In distributed_slurm mode (automatic)
--seed "{dataset}_{version}_run$((SLURM_PROCID + offset))"
```

### 2.3 Reproducibility Guarantees

**Stored in Output Directory:**
```
{output_base_dir}/{dataset}/{version}/
├── expanded_config.yaml          # Full resolved config
├── .git_commit_success          # Git commit hash + timestamp
└── software_snapshots/current/  # Git tree snapshot
```

**Reproduction Procedure:**
```bash
# 1. Get commit hash from marker
COMMIT_HASH=$(grep "Git Hash:" .git_commit_success | awk '{print $NF}')

# 2. Restore code
git checkout $COMMIT_HASH

# 3. Load exact config
source expanded_config.yaml

# 4. Re-run with same seed
python pythia_gen.py --config expanded_config.yaml --seed <original_seed>
```

---

## 3. Multi-Node Job Coordination

### 3.1 Distributed vs Multi-Node Architectures

| Aspect | distributed_slurm | multi_node_slurm |
|--------|---|---|
| **Jobs** | N independent SLURM tasks | 1 SLURM job spanning N nodes |
| **Scheduling** | Kernel schedules separately | Kernel allocates contiguous allocation |
| **Load Balancing** | Independent task queues | Shared task queue with work stealing |
| **Communication** | Filesystem coordination | Optional inter-process (not used) |
| **Failure Handling** | Per-task retry | Entire job retry |

### 3.2 Directory Structure in Multi-Node Mode

For MadGraph multi-node generation with splitting:

```
output_base_dir/
└── {dataset}/
    └── {version}/
        ├── runs/                    # Global split output
        │   ├── 0/
        │   │   └── events.hepmc
        │   ├── 1/
        │   │   └── events.hepmc
        │   └── ...
        │
        └── runs/all/                # Per-job staging (temporary)
            ├── 0/
            │   ├── events.hepmc.gz
            │   └── final_cards/
            │       ├── run_card.dat
            │       └── pythia8_card.dat
            ├── 1/
            │   └── ...
```

**Key Design:**
- `runs/all/X/` - MG events and metadata (per job, may overlap)
- `runs/X/` - Final split output (global indexing, no overlap)

### 3.3 Global Run Offset Calculation

**Code Reference:** `madgraph_gen.py (lines 367-391)`

```python
def _calculate_split_config(mg_run_id, staging_output_dir, events_per_mg_run, 
                            split_events_per_file, max_files_per_mg_run, logger):
    """Calculate split output directory and global run offset."""
    
    is_multinode = mg_run_id is not None and mg_run_id >= 0
    
    if is_multinode:
        # Multi-node mode
        split_output_base_dir = staging_output_dir.parent.parent  # runs/
        
        # Runs per job
        if max_files_per_mg_run is not None:
            runs_per_mg_job = max_files_per_mg_run
        else:
            runs_per_mg_job = events_per_mg_run // split_events_per_file
        
        # Global offset prevents collision
        global_run_offset = mg_run_id * runs_per_mg_job
```

**Example Calculation:**
```
Config: events=35000, events_per_file=64, max_files_per_mg_run=100
mg_run_id=0: offset=0*100=0, outputs to runs/0-99/
mg_run_id=1: offset=1*100=100, outputs to runs/100-199/
mg_run_id=2: offset=2*100=200, outputs to runs/200-299/
...
mg_run_id=20: offset=20*100=2000, outputs to runs/2000-2099/
```

### 3.4 Coordinating with SLURM Array Jobs

**Array Job Setup:**
```bash
# Submit array job (21 tasks, 0-20)
sbatch --array=0-20 slurm_script.sh
```

**Task Environment Variables:**
```bash
SLURM_PROCID=0          # Task 0
SLURM_JOB_NODELIST     # Allocated nodes
SLURM_NTASKS           # 21 tasks total
```

**Task-to-Run-ID Mapping:**
```python
# In distributed_slurm mode
output_subdir = "$(({slurm_procid_offset} + SLURM_PROCID))"

# Example with offset=0:
# Task 0: output_subdir=0
# Task 1: output_subdir=1
# ...
# Task 20: output_subdir=20

# Detection in madgraph_gen.py (line 632)
if args.output_subdir.isdigit():
    mg_run_id = int(args.output_subdir)  # Detected!
    staging_output_dir = effective_output_dir.parent / "all" / args.output_subdir
```

---

## 4. Splitting and Distribution Strategies

### 4.1 HepMC File Splitting Algorithm

**Purpose:** Prevent monolithic output files, enable parallel postprocessing.

**Code Reference:** `madgraph_gen.py (lines 175-272)`

```python
def split_hepmc_file(input_hepmc_path, final_output_base_dir, 
                     events_per_file, output_filename="events.hepmc",
                     global_run_offset=0, max_files_per_mg_run=None):
    """Split large HepMC into smaller files in subdirectories"""
    
    files_created = []
    current_writer = None
    event_index = 0
    
    with hep.open(str(input_hepmc_path)) as f_in:
        for event in f_in:
            # Stop if capped
            if max_files_per_mg_run and event_index >= max_files_per_mg_run * events_per_file:
                break
            
            # New file every events_per_file events
            if event_index % events_per_file == 0:
                if current_writer:
                    current_writer.close()
                
                chunk_index = event_index // events_per_file
                global_run_index = global_run_offset + chunk_index
                split_dir = final_output_base_dir / str(global_run_index)
                split_dir.mkdir(parents=True, exist_ok=True)
                split_path = split_dir / output_filename
                current_writer = WriterAscii(str(split_path))
                files_created.append(split_path)
            
            # Write event (reset event number for each file)
            event.event_number = event_index % events_per_file
            current_writer.write_event(event)
            event_index += 1
```

**Key Features:**
- **Incremental writing**: Uses streaming pyhepmc reader (low memory)
- **Directory per run**: Each run gets `{run_index}/events.hepmc`
- **Event number reset**: Per-file event numbering for consistency
- **Partial chunk discard**: Removes incomplete final chunk

**Output Structure:**
```
split_output_base_dir/
├── 0/
│   └── events.hepmc       # Events 0-63
├── 1/
│   └── events.hepmc       # Events 64-127
├── 2/
│   └── events.hepmc       # Events 128-191
...
└── 99/
    └── events.hepmc       # Events 6336-6399
```

### 4.2 Capping Output Size

**Configuration:**
```yaml
splitting_config:
  max_files_per_mg_run: 100    # Cap at 100 split files per MadGraph job
```

**Use Case:** Prevent runaway large MadGraph runs from consuming excessive disk.

**Behavior:**
```python
max_events_to_process = max_files_per_mg_run * events_per_file
# With max_files=100, events_per_file=64:
# max_events_to_process = 100 * 64 = 6400 events

# If MadGraph generates 35000 events:
# - First 6400 events → written to runs/0-99/
# - Remaining 28600 events → discarded
```

---

## 5. Card Customization Patterns

### 5.1 MadGraph Card Hierarchy

**Three customization levels:**

| Level | Timing | Scope | Example |
|-------|--------|-------|---------|
| **1. Default** | `madgraph_init` | Process-wide | Matrix element settings |
| **2. Base** | `madgraph_init` | Physics-specific | Jet multiplicity, kinematics |
| **3. Run** | `madgraph_generation` | Per-job | Events, seed, run name |

**Hierarchy (later overrides earlier):**
```
Default → Base → Run
```

### 5.2 Card Customization by Run Mode

**LO+MLM (Loop+MLM JetMatching):**
```python
# madgraph_gen.py (lines 155-157)
if str(run_mode).lower() == 'lo_mlm':
    _customize_pythia8_card(cards_dir, config, logger, " for run_mode=lo_mlm")
    # Only modifies pythia8_card.dat (MLM settings live here)
```

**NLO/FxFx (Flexible+Fixed scale):**
```python
# madgraph_gen.py (lines 158-173)
else:
    if shower_card_path.exists():
        # Customize shower_card.dat (Herwig/Pythia8 shower settings)
        customize_card_with_regex(shower_card_path, final_shower_settings)
    else:
        # Fallback to pythia8_card.dat
        _customize_pythia8_card(cards_dir, config, logger, " for loop-induced/NLO fallback")
```

### 5.3 Regex-Based Card Modification

**Code Reference:** `madgraph_utils.py`

```python
def customize_card_with_regex(card_path, params_dict):
    """Modify card file using regex pattern matching"""
    
    with open(card_path, 'r') as f:
        content = f.read()
    
    for param_name, param_value in params_dict.items():
        # Build regex: match parameter name (case-insensitive) and value
        pattern = rf'^(\s*){re.escape(param_name)}(\s*)(=|\s)(.*)$'
        replacement = f'\\1{param_name}\\2 = {param_value}'
        
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.IGNORECASE)
    
    with open(card_path, 'w') as f:
        f.write(content)
```

**Example Modifications:**
```yaml
# run_card.dat
nevents: 35000
iseed: 42

# pythia8_card.dat (LO+MLM)
Main:numberOfEvents: 35000
Random:seed: 42
Dire:limitMass: 20.0
```

---

## 6. Batch Job Architecture

### 6.1 Command Construction Flowchart

```
Config File
    ↓
[cli_utils.load_and_process_config()]
    ↓
Environment Variables Substituted
    ↓
[cli_utils.get_env_setup_cmds()]
    ↓
Setup Commands + Environment
    ↓
[cli_utils.build_stage_command()]
    ↓
Python Command + Shifter Prefix (if needed)
    ↓
Final Shell Command Ready for Execution
```

### 6.2 Shifter Container Usage

**Example Command (simulation stage):**
```bash
shifter --image=registry.cern.ch/atlasadc/atlas-grid-almalinux9 -- \
  bash -c "
    source /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh && \
    source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_107 x86_64-el9-gcc13-opt && \
    source /path/to/dd4hep/setup.sh && \
    source /path/to/acts/setup.sh && \
    python scripts/simulation/ddsim_run.py --config config.yaml --output /output --output-subdir 0
  "
```

**No Shifter (madgraph_generation):**
```bash
source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_107 x86_64-el9-gcc13-opt && \
source /path/to/madgraph/env/bin/activate && \
python scripts/simulation/madgraph_gen.py --config config.yaml --output /output --output-subdir 0
```

### 6.3 Validation & Guardian Integration

**Optional Post-Stage Workflow:**
```python
# run_stage.py (lines 227-325)

# Phase 1: Execute stage
stage_exit_code = run_stage_script(...)

# Phase 2 (optional): Validate outputs
if validation_enabled:
    validation_result = run_validation(...)
    
    # Phase 3 (optional): Guardian decision
    decision = run_guardian(validation_result, ...)
    
    if decision['action'] == 'RETRY':
        # Retry with adjusted parameters
    elif decision['action'] == 'FAIL':
        sys.exit(decision['exit_code'])
```

---

## 7. Performance Analysis

### 7.1 Timing Instrumentation

**Code Reference:** `utils/app_logging.py`

```python
class TimingRecorder:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.timings = {}
    
    @contextmanager
    def record(self, label):
        """Context manager for timing code blocks"""
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            self.timings[label] = elapsed
    
    def write_report(self):
        """Write timing report to JSON"""
        report_path = self.output_dir / "timing_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.timings, f, indent=2)
```

**Usage in pythia_gen.py:**
```python
timer = TimingRecorder(output_dir)

with timer.record("Pythia8 + ACTS Workflow"):
    final_output = run_workflow(output_dir, config, logger)

timer.write_report()  # timing_report.json
```

### 7.2 Expected Performance Timings

**Pythia8 Generation (1000 hard scatter + 200k pileup events):**
- Hard scatter: ~10-30 seconds (depending on process)
- Pileup: ~30-60 seconds
- ACTS merge: ~15-30 seconds
- **Total: ~60-120 seconds per job**

**MadGraph Initialization (e.g., di-Higgs → 4b):**
- Process generation: 30-60 minutes
- Matrix element compilation: 30-90 minutes
- **Total: 1-4 hours (one-time)**

**MadGraph Event Generation (35k events):**
- Event generation: 5-15 minutes
- HepMC splitting: 2-5 minutes (if enabled)
- **Total: 10-20 minutes per job**

**DD4hep Simulation (1000 events, full detector):**
- Initialization: ~30 seconds
- Event loop: ~20-50 seconds
- Output writing: ~5-10 seconds
- **Total: ~60-90 seconds per job**

---

## 8. Troubleshooting Complex Scenarios

### 8.1 Seed Collision Detection

**Problem:** Multiple jobs producing identical events (seed collision).

**Diagnosis:**
```bash
# Extract seeds from output logs
grep "Final seed value:" logs/*.log

# Check for duplicates
grep "Final seed value:" logs/*.log | sort | uniq -d
```

**Solution:**
```bash
# Use environment-based seed pattern
--seed "$SLURM_JOB_ID:$SLURM_PROCID"

# Or explicit numeric series
--seed "$(($base_seed + $SLURM_PROCID))"
```

### 8.2 Pileup Generation Timeout

**Problem:** Pileup generation exceeds time limit with Poisson sampling.

**Cause:** Large buffer calculation with high μ and 5σ:
```
events_needed = 1000 * 300 + 5 * sqrt(1000 * 300)
              = 300,000 + 5 * 547.7
              ≈ 302,740 events  (2.7% overhead)

events_needed = 10000 * 300 + 5 * sqrt(10000 * 300)
              = 3,000,000 + 5 * 1732
              ≈ 3,008,660 events  (0.29% overhead)
```

Higher multiplicity actually reduces relative overhead.

**Solutions:**
1. Reduce σ buffer: `poisson_buffer_sigma: 3.0` (instead of 5.0)
2. Increase time limit: `time_limit: "02:00:00"`
3. Use fixed multiplicity: `poisson_sample: false`

### 8.3 MadGraph Process Compilation Failure

**Problem:** `madgraph_init` fails partway through.

**Debugging:**
```bash
# Check MG5 output for errors
tail -100 /path/to/mg5_init_log.txt

# Look for common issues:
grep -i "error\|fatal\|segfault" /path/to/mg5_init_log.txt
```

**Common Causes:**
- Model not found: Verify `mg_model` exists in MG5
- Card syntax error: Check `mg_definitions` YAML syntax
- Process generation timeout: May need longer time limit
- Disk space: Check scratch directory quota

**Recovery:**
```bash
# Clean previous attempt
rm -rf /scratch/mg5_init_*

# Re-run with verbose output
python madgraph_init.py --config config.yaml 2>&1 | tee debug.log
```

### 8.4 ACTS Merge Crashes

**Problem:** `merged_events.hepmc3` not created, ACTS merge fails.

**Common Causes:**
- Incompatible pyhepmc version
- Corrupted HepMC input files
- Memory exhaustion with large pileup

**Diagnostics:**
```bash
# Verify HepMC files
python -c "
import pyhepmc
with pyhepmc.open('events.hepmc3') as f:
    for i, event in enumerate(f):
        if i >= 10:
            break
        print(f'Event {i}: {len(list(event.particles))} particles')
"

# Check for memory issues in logs
grep -i "memory\|alloc\|segfault" logs/pythia_gen.log
```

**Solutions:**
1. Upgrade pyhepmc: `pip install --upgrade pyhepmc==2.14.0`
2. Reduce events per run: Lower `events` configuration
3. Increase memory allocation: Check SLURM config

---

## 9. Configuration Validation Patterns

### 9.1 Pre-flight Checks

**Recommended before submitting large jobs:**

```bash
#!/bin/bash
# validate_config.sh

CONFIG=$1

# 1. Check YAML syntax
python -c "import yaml; yaml.safe_load(open('$CONFIG'))" || exit 1

# 2. Verify required fields
for field in dataset version stage events seed; do
  if ! grep -q "^$field:" "$CONFIG"; then
    echo "ERROR: Missing required field '$field'"
    exit 1
  fi
done

# 3. Check paths exist (if absolute)
python << 'EOF'
import yaml
config = yaml.safe_load(open('$CONFIG'))

# Check output base dir accessible
import os
out_base = config.get('common', {}).get('output_base_dir')
if out_base and not os.access(out_base, os.W_OK):
    print(f"ERROR: Cannot write to {out_base}")
    exit(1)
EOF

echo "Configuration validation passed"
```

### 9.2 Post-Execution Verification

**Checks after stage completion:**

```python
def verify_stage_output(output_dir, stage, events):
    """Verify stage produced expected outputs"""
    
    checks = {
        'pythia_generation': {
            'files': ['events_signal.hepmc3', 'events_pileup.hepmc3', 'merged_events.hepmc3'],
            'min_size_mb': 50,
        },
        'madgraph_generation': {
            'files': ['0/events.hepmc', '1/events.hepmc'],
            'min_size_mb': 10,
        },
        'simulation': {
            'files': ['simulation_output.root'],
            'min_size_mb': 100,
        },
    }
    
    stage_checks = checks.get(stage, {})
    
    for required_file in stage_checks.get('files', []):
        fpath = Path(output_dir) / required_file
        if not fpath.exists():
            raise FileNotFoundError(f"Expected output not found: {required_file}")
        
        size_mb = fpath.stat().st_size / (1024**2)
        if size_mb < stage_checks.get('min_size_mb', 1):
            raise ValueError(f"Output file too small: {required_file} ({size_mb}MB)")
```

---

## 10. Advanced Configuration Recipes

### 10.1 High-Multiplicity Pileup Study

```yaml
campaign: "pileup_scan"
dataset: "ttbar"
version: "v1"

stage: "pythia_generation"

events: 1000
pileup: 1000              # Extreme pileup (1000 events per hard scatter)

hard_process: "Top:pair"

# Realistic vertex smearing scaled for high-pileup regime
vertex_sigma_xy: 0.025    # Double transverse for crowded detector
vertex_sigma_z: 100.0     # Larger z spread
vertex_sigma_t: 10.0      # Time spread

# Use Poisson for realistic multiplicity
poisson_sample: true
poisson_buffer_sigma: 3.0

job_config:
  n_runs: 10              # Fewer runs (fewer events needed overall)
  runs_per_node: 1
  time_limit: "01:00:00"  # Longer for pileup generation
  execution_mode: "distributed_slurm"
```

### 10.2 Cross-Section Study with Multiple Seeds

```yaml
campaign: "cross_section_study"
dataset: "higgs_gg2bb"
version: "v1"

stage: "madgraph_generation"

# NLO Higgs production
mg_model: "sm"
mg_generate_command: "generate p p > h > b b @NLO"
run_mode: "nlo_fxfx"

events: 50000          # More events for better statistics
seed: 42               # Will be overridden per job

job_config:
  n_runs: 20           # 20 independent seeds
  runs_per_node: 1
  execution_mode: "distributed_slurm"

# Per-job customization (events, seed)
splitting_config:
  enable: true
  events_per_file: 100
```

### 10.3 BSM Resonance Search Template

```yaml
campaign: "zprime_search"
dataset: "zprime_1p5tev"
version: "v1"

stage: "pythia_generation"

hard_process: "NewGaugeBoson:Zp2bbbar"  # Z' → bb

pythia_settings:
  - "Zprime:mass = 1500"               # 1.5 TeV resonance
  - "Zprime:width = 150"               # 10% relative width
  - "Zprime:coup2u = 0.1"              # Custom couplings
  - "Zprime:coup2d = 0.1"

events: 50000          # Higher statistics for rare process
pileup: 200

vertex_sigma_xy: 0.0125
vertex_sigma_z: 55.5
vertex_sigma_t: 5.0

job_config:
  n_runs: 50
  runs_per_node: 1
  time_limit: "01:30:00"
  execution_mode: "distributed_slurm"
```

---

**Advanced Topics Document Version:** 1.0  
**Last Updated:** 2025-01-19  
**Focus:** Technical Implementation Details for Complex Simulations

