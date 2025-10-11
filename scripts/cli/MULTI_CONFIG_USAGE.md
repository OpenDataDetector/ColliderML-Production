# Multi-Config SLURM Job Submission

## Overview

The ColliderML CLI now supports combining multiple stage configurations into a single large SLURM job. This is useful for taking advantage of bulk discounts on HPC systems like Perlmutter, which offers reduced charging for jobs using >256 nodes.

## Key Features

- **Combine multiple stages** into a single SLURM allocation
- **Parallel execution** - all stages run simultaneously (not sequentially)
- **Isolated outputs** - each stage writes to its own version directory
- **Automatic PROCID remapping** - each stage sees local PROCID values (0-based)
- **Backward compatible** - single-config usage unchanged

## Usage

### Basic Syntax

```bash
python run_stage.py config1.yaml config2.yaml config3.yaml --execution-mode multi_node_slurm [OPTIONS]
```

### Example

Combine pythia generation, simulation, and digitization into one 300-node job:

```bash
python run_stage.py \
    configs/pythia_config.yaml \
    configs/simulation_config.yaml \
    configs/digitization_config.yaml \
    --execution-mode multi_node_slurm \
    --dry-run
```

## Requirements and Constraints

### 1. Execution Mode

Multi-config jobs **only support** `multi_node_slurm` mode. This is enforced automatically.

### 2. Stage Compatibility

Only **simulation stages** that use shifter containers can be combined:
- `pythia_generation`
- `particlegun_generation`
- `merge_smear`
- `simulation`
- `digitization`

**You cannot combine:**
- Simulation stages with postprocessing stages
- Stages with different container images
- MadGraph stages with other simulation stages (different environment)

### 3. Container Consistency

All configs must specify the same `common.container` value. The system will validate this and error if containers differ.

## How It Works

### Resource Allocation

The system:
1. Calculates total nodes needed (sum across all stages)
2. Calculates total tasks needed (sum of all run counts)
3. Creates a single SLURM job with these combined resources

Example:
- Stage 1: 100 runs × 4 runs/node = 25 nodes
- Stage 2: 150 runs × 2 runs/node = 75 nodes  
- Stage 3: 200 runs × 2 runs/node = 100 nodes
- **Total: 200 nodes, 450 tasks**

### Parallel Execution

Each stage gets its own `srun` command running in the background:

```bash
# Stage 0: pythia_generation (PROCID 0-99)
srun --ntasks=100 shifter bash -c "..." &

# Stage 1: simulation (PROCID 100-249)
srun --ntasks=150 shifter bash -c "..." &

# Stage 2: digitization (PROCID 250-449)
srun --ntasks=200 shifter bash -c "..." &

wait  # All stages must complete
```

### PROCID Remapping

SLURM assigns global `SLURM_PROCID` values (0, 1, 2, ..., N-1). Each stage remaps these to local values:

- **Global PROCID 0-99** → Stage 0 sees `STAGE_PROCID` 0-99
- **Global PROCID 100-249** → Stage 1 sees `STAGE_PROCID` 0-149
- **Global PROCID 250-449** → Stage 2 sees `STAGE_PROCID` 0-199

This ensures each stage writes to the correct run directories (0, 1, 2, ...) within its own version directory.

## Configuration

Each config file works exactly as before:

```yaml
# config1.yaml
campaign: "full_pileup_pilot"
dataset: "ttbar"
version: "v1"
stage: "pythia_generation"

job_config:
  n_runs: 100
  runs_per_node: 4
  time_limit: "02:00:00"
  qos: "regular"

# ... stage-specific settings ...
```

The combined job will use:
- **Time limit**: Maximum across all configs
- **QOS**: From first config (warning if different)
- **Account**: From first config (must be same, enforced)
- **Container**: Must be identical (enforced)

## Output Structure

Each stage writes to its own directory:

```
output_base_dir/
├── campaign1/
│   └── dataset1/
│       └── version1/
│           ├── runs/
│           │   ├── 0/
│           │   ├── 1/
│           │   └── ...
│           ├── logs/
│           │   └── stage_pythia_generation/
│           └── configs/
│               └── config1.yaml
├── campaign2/
│   └── dataset2/
│       └── version2/
│           ├── runs/
│           │   ├── 0/
│           │   ├── 1/
│           │   └── ...
│           └── ...
└── ...
```

## Validation Jobs

Validation jobs are submitted **separately for each stage** after the combined job completes. Each validation job depends on the combined job ID using SLURM's `afterany` dependency.

## Git Commit and Config Snapshots

- Git commit is performed **once** (using first config)
- Each config is saved to its respective `version_dir/configs/` directory
- Git hash is recorded in the first config's version directory

## Dry Run

Test your multi-config setup without submitting:

```bash
python run_stage.py config1.yaml config2.yaml --execution-mode multi_node_slurm --dry-run
```

This generates batch scripts in `version_dir/dry_run_combined/` for inspection.

## Error Handling

The system validates:
- All stages are simulation stages (shifter-compatible)
- All configs use the same container
- Execution mode is `multi_node_slurm` when multiple configs provided

Common errors:

```
# Mixing incompatible stages
ERROR: Stage 'build_tracks' is not in SHIFTER_STAGES

# Different containers
ERROR: All configs must use the same container. Found: {...}

# Wrong execution mode
ERROR: Multi-config jobs only support multi_node_slurm mode
```

## Advanced Options

All standard `run_stage.py` options work:

```bash
python run_stage.py config1.yaml config2.yaml \
    --execution-mode multi_node_slurm \
    --dry-run \
    --force-commit \
    --allow-master
```

**Note**: `--run-range` and `--run-list` are **not supported** in multi-config mode (they would need to be specified per-config, which is not currently implemented).

## Examples

### Two-stage job (200 nodes)

```bash
python run_stage.py \
    configs/pythia_150nodes.yaml \
    configs/simulation_50nodes.yaml \
    --execution-mode multi_node_slurm
```

### Three-stage job (300+ nodes)

```bash
python run_stage.py \
    configs/pythia_100nodes.yaml \
    configs/simulation_100nodes.yaml \
    configs/digitization_100nodes.yaml \
    --execution-mode multi_node_slurm \
    --dry-run  # Test first!
```

## Backward Compatibility

Single-config usage is **completely unchanged**:

```bash
# This still works exactly as before
python run_stage.py config.yaml --execution-mode interactive
python run_stage.py config.yaml --execution-mode distributed_slurm
python run_stage.py config.yaml --execution-mode multi_node_slurm
```

## Implementation Details

Multi-config logic is isolated in `multi_config_job.py`:
- `MultiConfigJobSubmitter` class handles combined job submission
- Uses individual `JobSubmitter` instances internally for directory setup and validation
- No changes to existing `JobSubmitter` class (maintains backward compatibility)
- Minimal changes to `run_stage.py` (just routing logic)

## Troubleshooting

### Import errors
```bash
cd scripts/cli
python3 -c "import multi_config_job; print('OK')"
```

### Check generated script
```bash
cat version_dir/dry_run_combined/job_combined_multiconfig.sh
```

### Verify PROCID ranges
Look for log messages:
```
Combined resources: 200 nodes, 450 tasks
PROCID ranges: [(0, 100), (100, 250), (250, 450)]
```

## Performance Considerations

- **Node efficiency**: Ensure `runs_per_node` values fill nodes efficiently
- **Load balancing**: SLURM distributes tasks; stages run in parallel
- **Time limits**: Set conservatively; combined job fails if any stage times out
- **Cost savings**: >256 nodes gets discount on Perlmutter (check your HPC's policy)

