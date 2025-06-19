# Pythia8 + ACTS Workflow Usage Guide

The refactored `pythia_gen.py` implements a clean two-phase workflow:

1. **Generation Phase** (optional): Generate hard scatter and/or pileup events
2. **Merging Phase** (optional): Merge events using ACTS HepMC3Reader

## Quick Start Examples

### 1. Generate Everything and Merge (Auto-mode)
```bash
# Auto-determines workflow from config
python pythia_gen.py --config my_config.yaml --output /path/to/output
```
This will:
- Generate hard scatter if `hard_process` is configured
- Generate pileup if `pileup > 0` is configured  
- Merge them if both are generated

### 2. Explicit Control
```bash
# Generate hard scatter only
python pythia_gen.py --config config.yaml --generate-hard-scatter --output /path/to/output

# Generate pileup only
python pythia_gen.py --config config.yaml --generate-pileup --output /path/to/output

# Generate both and merge
python pythia_gen.py --config config.yaml --generate-hard-scatter --generate-pileup --merge --output /path/to/output

# Just merge existing files
python pythia_gen.py --config config.yaml --merge --output /path/to/output
```

### 3. MadGraph + Pythia8 + ACTS Workflow
```bash
# Step 1: Generate signal with MadGraph
python madgraph_gen.py --config config.yaml --output /path/to/output

# Step 2: Generate pileup with Pythia8  
python pythia_gen.py --config config.yaml --generate-pileup --output /path/to/output

# Step 3: Merge with ACTS
python pythia_gen.py --config config.yaml --merge --output /path/to/output
```

### 4. Using the Standalone ACTS Merger
```bash
# Merge with auto-detection
python acts_merge.py --config config.yaml --output /path/to/output

# Merge with explicit file paths
python acts_merge.py --config config.yaml \
  --hard-scatter /path/to/signal.hepmc3 \
  --pileup /path/to/pileup.hepmc3 \
  --output /path/to/output
```

## Configuration Options

### In YAML Config
```yaml
# Basic settings
events: 1000
seed: 42
pileup: 200

# Hard scatter generation  
hard_process: "HardQCD:all"
pythia_settings:
  - "PhaseSpace:pTHatMin = 10"
  - "PhaseSpace:pTHatMax = 1000"

# Vertex smearing
vertex_sigma_xy: 0.1  # mm
vertex_sigma_z: 5.0   # mm
vertex_sigma_t: 0.0   # ns

# File paths (optional)
hard_scatter_file: "/path/to/existing/signal.hepmc3"
pileup_file: "/path/to/existing/pileup.hepmc3"
```

### Command Line Overrides
```bash
# Override vertex smearing
python pythia_gen.py --config config.yaml \
  --vertex-sigma-xy 0.05 \
  --vertex-sigma-z 2.0 \
  --generate-hard-scatter --generate-pileup --merge

# Override pileup multiplicity
python acts_merge.py --config config.yaml \
  --pileup-multiplicity 100
```

## File Naming Conventions

The workflow uses consistent file naming:

- **Hard scatter**: `events_signal.hepmc3` (from Pythia8) or `events.hepmc3`/`events.hepmc.gz` (from MadGraph)
- **Pileup**: `events_pileup.hepmc3`
- **Merged**: `merged_events.hepmc3`

## Auto-Detection Logic

The scripts automatically detect files in this order:

### Hard Scatter Files:
1. `events_signal.hepmc3` (Pythia8 generation)
2. `events.hepmc3` (MadGraph with splitting)
3. `events.hepmc.gz` (MadGraph without splitting)
4. Any single `.hepmc*` file (if only one exists)

### Pileup Files:
1. `events_pileup.hepmc3`

## Key Features

### Always ACTS Merging
- No more traditional Pythia8 merging
- Consistent vertex smearing via ACTS `GaussianVertexGenerator`
- Better integration with ACTS simulation chain

### Flexible Workflow
- Generate hard scatter: `--generate-hard-scatter`
- Generate pileup: `--generate-pileup`
- Merge events: `--merge`
- Any combination is valid

### Smart File Handling
- Auto-detection of existing files
- Config-based file path specification
- Command-line path overrides

### Efficient Resource Usage
- Different random seeds for signal vs pileup
- Proper memory management
- Timing reports for performance monitoring

## Typical Workflows

### Pure Pythia8 Workflow
```bash
python pythia_gen.py --config config.yaml --generate-hard-scatter --generate-pileup --merge --output output/
```

### MadGraph + Pythia8 Workflow  
```bash
# Generate signal
python madgraph_gen.py --config config.yaml --output output/signal/

# Generate pileup  
python pythia_gen.py --config config.yaml --generate-pileup --output output/pileup/

# Merge
python acts_merge.py --config config.yaml \
  --hard-scatter output/signal/events.hepmc.gz \
  --pileup output/pileup/events_pileup.hepmc3 \
  --output output/merged/
```

### Pileup-Only Generation
```bash
python pythia_gen.py --config config.yaml --generate-pileup --output output/
```

This creates individual pileup events that can be used later for merging with any signal file. 