# ColliderML Dataset Upload to HuggingFace

Tools for uploading the unified ColliderML dataset to HuggingFace Hub.

## Overview

The ColliderML dataset is uploaded to HuggingFace as a **single unified dataset** with multiple configurations (subsets) for different physics processes, pileup conditions, and object types.

**Unified Dataset:** `OpenDataDetector/ColliderML-Release-1`

**Configuration naming:** `{dataset}_{pileup_label}_{object_type}`

**Examples:**
- `ttbar_pu0_particles` - ttbar with no pileup, truth particles
- `ttbar_pu0_tracker_hits` - ttbar with no pileup, tracker hits
- `ttbar_pu200_particles` - ttbar with 200 pileup, truth particles
- `ggf_pu0_tracks` - gluon fusion with no pileup, reconstructed tracks

## Quick Start

### 1. Install Requirements

```bash
pip install datasets huggingface_hub
```

### 2. Authenticate with HuggingFace

```bash
# Option 1: Interactive login
huggingface-cli login

# Option 2: Set token as environment variable
export HF_TOKEN=your_token_here
```

Get your token from: https://huggingface.co/settings/tokens

### 3. Configure Upload

Edit `unified_dataset_config.yaml` to:
- Enable/disable specific datasets
- Add new campaigns or physics processes
- Adjust upload settings

### 4. Upload Dataset

```bash
# Dry run first (no upload, just shows what would happen)
python scripts/dataset/upload_to_hf_unified.py scripts/dataset/unified_dataset_config.yaml --dry-run

# Upload all enabled configs
python scripts/dataset/upload_to_hf_unified.py scripts/dataset/unified_dataset_config.yaml

# Upload specific configs only
python scripts/dataset/upload_to_hf_unified.py scripts/dataset/unified_dataset_config.yaml --configs ttbar_pu0_particles

# Resume from specific config (if interrupted)
python scripts/dataset/upload_to_hf_unified.py scripts/dataset/unified_dataset_config.yaml --start-from ggf_pu0_particles
```

## Upload Script: `upload_to_hf_unified.py`

### How It Works

The script uses the `datasets` library's `push_to_hub()` method to upload parquet files directly to HuggingFace:

1. Loads parquet files from local directory
2. Creates a HuggingFace Dataset object
3. Pushes to hub with a specific config name
4. Automatically generates/updates README.md with YAML metadata

Each config is uploaded **independently** - you can add new configs without affecting existing ones.

### Key Features

- ✅ **Progressive uploads** - Add configs one at a time
- ✅ **Skip existing configs** - Avoid re-uploading (default behavior)
- ✅ **Automatic sharding** - Control file sizes with `max_shard_size`
- ✅ **Automatic README** - YAML metadata generated automatically
- ✅ **Dry run mode** - Test before uploading
- ✅ **Resume capability** - Continue from where you left off
- ✅ **Custom cache directory** - Uses project filesystem to avoid home quota issues

### Command-Line Options

```bash
python upload_to_hf_unified.py CONFIG_FILE [OPTIONS]

Options:
  --dry-run              Show what would be uploaded without uploading
  --configs CONFIG ...   Upload only these specific configs
  --start-from CONFIG    Start from this config (skip earlier ones)
  --skip-existing        Skip configs that already exist (default)
  --no-skip-existing     Re-upload even if config exists
  --verbose              Enable detailed logging
```

## Configuration File: `unified_dataset_config.yaml`

### Structure

```yaml
huggingface:
  repo_id: "OpenDataDetector/ColliderML-Release-1"
  license: "cc-by-4.0"
  private: false

data:
  base_dir: "/global/cfs/cdirs/m4958/data/ColliderML/simulation"
  version: "v1"
  format_subdir: "parquet"

object_types:
  - particles
  - tracker_hits
  - calo_hits
  - tracks

campaigns:
  - campaign_name: "hard_scatter"
    pileup_label: "pu0"
    datasets:
      - name: "ttbar"
        enabled: true
      - name: "ggf"
        enabled: true
```

### Available Campaigns

1. **hard_scatter** (pu0) - Hard scatter events with no pileup
   - ttbar, ggf, dihiggs, susy_rpv

2. **full_pileup** (pu200) - Realistic HL-LHC pileup (~200 interactions)
   - ttbar, ggf, dihiggs, diphoton, jets, susy_rpv, zee, zmumu

3. **single_particle_pilot** (single) - Particle gun studies
   - single_muon_1GeV, single_muon_10GeV, single_muon_100GeV, single_electron, etc.

### Enabling Datasets

To enable a dataset for upload, set `enabled: true`:

```yaml
- name: "ttbar"
  enabled: true  # Will be uploaded
- name: "ggf"
  enabled: false  # Will be skipped
```

## Usage Examples

### Example 1: Test with Dry Run

```bash
python scripts/dataset/upload_to_hf_unified.py \
    scripts/dataset/unified_dataset_config.yaml \
    --dry-run
```

### Example 2: Upload Single Config for Testing

```bash
python scripts/dataset/upload_to_hf_unified.py \
    scripts/dataset/unified_dataset_config.yaml \
    --configs ttbar_pu0_particles
```

### Example 3: Upload All ttbar Configs

```bash
python scripts/dataset/upload_to_hf_unified.py \
    scripts/dataset/unified_dataset_config.yaml \
    --configs ttbar_pu0_particles ttbar_pu0_tracker_hits ttbar_pu0_calo_hits ttbar_pu0_tracks
```

### Example 4: Resume Interrupted Upload

If upload was interrupted, skip already-uploaded configs:

```bash
python scripts/dataset/upload_to_hf_unified.py \
    scripts/dataset/unified_dataset_config.yaml \
    --skip-existing  # This is the default
```

Or start from a specific config:

```bash
python scripts/dataset/upload_to_hf_unified.py \
    scripts/dataset/unified_dataset_config.yaml \
    --start-from ggf_pu0_particles
```

## Directory Structure

The script expects parquet files organized as:

```
{base_dir}/{campaign_name}/{dataset_name}/{version}/{format_subdir}/
├── truth/
│   └── particles/
│       └── *.parquet
└── reco/
    ├── tracker_hits/
    │   └── *.parquet
    ├── calo_hits/
    │   └── *.parquet
    └── tracks/
        └── *.parquet
```

**Example:**
```
/global/cfs/cdirs/m4958/data/ColliderML/simulation/hard_scatter/ttbar/v1/parquet/
├── truth/particles/*.parquet
└── reco/tracker_hits/*.parquet
```

## Loading Data (User Perspective)

Once uploaded, users can access the dataset like this:

```python
from datasets import load_dataset, get_dataset_config_names

# List all available configs
configs = get_dataset_config_names("OpenDataDetector/ColliderML-Release-1")
print(f"Available configs: {configs}")

# Load a specific config
ds = load_dataset("OpenDataDetector/ColliderML-Release-1", "ttbar_pu0_particles")

# Load first 100 events only
ds = load_dataset(
    "OpenDataDetector/ColliderML-Release-1",
    "ttbar_pu0_particles",
    split="train[:100]"
)

# Load specific columns only (efficient!)
ds = load_dataset(
    "OpenDataDetector/ColliderML-Release-1",
    "ttbar_pu0_particles",
    split="train[:100]",
    columns=["event_id", "px", "py", "pz", "energy"]
)
```

## Troubleshooting

### Authentication Errors

```bash
# Make sure you're logged in
huggingface-cli login

# Or check if token is set
echo $HF_TOKEN
```

### Files Not Found

Check that:
- `base_dir` in config points to the correct location
- Parquet files exist in expected structure: `{campaign}/{dataset}/v1/parquet/truth/particles/` or `reco/{object_type}/`
- Dataset is enabled in the config file

### Upload Failures

- Use `--verbose` flag for detailed error messages
- Try uploading one config at a time with `--configs`
- Check HuggingFace service status
- Verify sufficient disk space and network connectivity

### Disk Quota Exceeded

The script automatically uses a cache directory on the project filesystem (`/global/cfs/cdirs/m4958/data/ColliderML/.hf_cache`) to avoid home directory quota issues on NERSC. If you still encounter quota issues:

```bash
# Clear the HuggingFace cache
rm -rf /global/cfs/cdirs/m4958/data/ColliderML/.hf_cache/*

# Or manually set cache location before running
export HF_HOME=/global/cfs/cdirs/m4958/data/ColliderML/.hf_cache
```

### Config Already Exists

By default, existing configs are skipped. To force re-upload:

```bash
python upload_to_hf_unified.py unified_dataset_config.yaml --no-skip-existing
```

## Support

For issues or questions:
- Email: daniel.thomas.murnane@cern.ch
- GitHub: https://github.com/OpenDataDetector/ColliderML
- HuggingFace Discussions: https://huggingface.co/datasets/OpenDataDetector/ColliderML-Release-1/discussions
