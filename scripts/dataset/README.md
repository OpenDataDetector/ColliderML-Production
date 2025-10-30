# Dataset Upload Scripts

Scripts for managing and uploading ColliderML datasets to HuggingFace Hub.

## upload_to_huggingface.py

Upload ColliderML parquet datasets to HuggingFace Hub with automatic validation and README generation.

### Features

- ✅ Validates that parquet files exist in expected locations
- ✅ Checks and fixes file permissions (world-readable)
- ✅ Populates README template with dataset information
- ✅ Saves README to local data directory
- ✅ Uploads same README to HuggingFace Hub
- ✅ Extracts parquet schemas automatically
- ✅ Template-based approach ensures consistency

### Usage

```bash
# Basic usage (upload to HuggingFace)
python scripts/dataset/upload_to_huggingface.py configs_production/hard_scatter/ttbar/huggingface_config.yaml

# Dry run (generate README but don't upload)
python scripts/dataset/upload_to_huggingface.py configs_production/hard_scatter/ttbar/huggingface_config.yaml --dry-run

# Save generated README to additional location
python scripts/dataset/upload_to_huggingface.py configs_production/hard_scatter/ttbar/huggingface_config.yaml --output /tmp/README.md

# Automatically fix file permissions
python scripts/dataset/upload_to_huggingface.py configs_production/hard_scatter/ttbar/huggingface_config.yaml --fix-permissions

# Use custom template
python scripts/dataset/upload_to_huggingface.py configs_production/hard_scatter/ttbar/huggingface_config.yaml --template /path/to/custom_template.md
```

### README Template

The script uses `scripts/dataset/README_template.md` as the template for generating the dataset card. This template:
- Uses Jinja2 syntax for dynamic values
- Contains all dataset documentation (structure, fields, usage examples)
- Is populated with dataset-specific information from the config
- Results in identical README for local data directory and HuggingFace

### Configuration File

The script requires a YAML configuration file. See `configs_production/hard_scatter/ttbar/huggingface_config.yaml` for an example.

Required fields:
- `campaign`: Campaign name (e.g., "hard_scatter")
- `dataset`: Dataset name (e.g., "ttbar")
- `version`: Version string (e.g., "v1")
- `data.base_dir`: Local path to parquet files
- `data.public_url_base`: Public NERSC portal URL
- `huggingface.repo_id`: HuggingFace repository ID
- `objects`: List of object types to include

### Workflow

1. **Generate parquet files** using `convert_all.py`

2. **Move to public location** (if needed):
   ```bash
   rsync -avz /path/to/output/ /global/cfs/cdirs/m4958/data/ColliderML/public/campaign/dataset/version/parquet/
   ```

3. **Set permissions**:
   ```bash
   chmod -R a+rX /global/cfs/cdirs/m4958/data/ColliderML/public/campaign/dataset/version/
   ```

4. **Create/update HuggingFace config**:
   ```bash
   cp configs_production/hard_scatter/ttbar/huggingface_config.yaml configs_production/campaign/dataset/huggingface_config.yaml
   # Edit the config file
   ```

5. **Test with dry run**:
   ```bash
   python scripts/dataset/upload_to_huggingface.py configs_production/campaign/dataset/huggingface_config.yaml --dry-run
   ```

6. **Upload to HuggingFace**:
   ```bash
   python scripts/dataset/upload_to_huggingface.py configs_production/campaign/dataset/huggingface_config.yaml
   ```

   This will:
   - Generate README from template
   - Save to `{base_dir}/README.md` (e.g., `/global/cfs/.../parquet/README.md`)
   - Upload same file to HuggingFace Hub

### Directory Structure Expected

The script expects parquet files to be organized as:

```
{base_dir}/
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

### Troubleshooting

**Files not found**:
- Check that `data.base_dir` in config points to correct location
- Verify subdirectory structure matches expected layout

**Permission errors**:
- Run with `--fix-permissions` flag
- Or manually: `chmod -R a+r /path/to/parquet/files`

**Upload fails**:
- Ensure you're authenticated with HuggingFace: `huggingface-cli login`
- Check that repo exists on HuggingFace Hub
- Verify you have write access to the repository

**Files not accessible via URL**:
- Verify files are in `/global/cfs/cdirs/m4958/data/ColliderML/public/`
- Check permissions: `ls -la /path/to/files`
- Test URL manually: `curl -I https://portal.nersc.gov/cfs/m4958/...`

### HuggingFace Authentication

Before uploading, authenticate with HuggingFace:

```bash
pip install huggingface_hub
huggingface-cli login
```

Enter your HuggingFace token when prompted (get it from https://huggingface.co/settings/tokens).
