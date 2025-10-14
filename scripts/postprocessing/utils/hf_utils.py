"""
HuggingFace utilities for dataset upload and management.

This module provides functions for:
- Generating dataset cards (README.md)
- Managing file inventories
- Merging configurations for progressive updates
- Parsing event ranges from filenames
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


def parse_event_range_from_filename(filename: str) -> Optional[Tuple[int, int]]:
    """
    Parse event range from ColliderML filename convention.
    
    Args:
        filename: Filename like "dataset.object.events0-999.parquet"
        
    Returns:
        Tuple of (start_event, end_event) or None if not found
        
    Example:
        >>> parse_event_range_from_filename("ttbar.particles.events0-999.parquet")
        (0, 999)
    """
    match = re.search(r'events(\d+)-(\d+)', filename)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None


def build_file_inventory(
    source_base: str,
    configs: List[Dict]
) -> Dict[str, Dict]:
    """
    Build inventory of files to upload with metadata.
    
    Args:
        source_base: Base directory containing Parquet files
        configs: List of config dicts with name and local_path
        
    Returns:
        Dictionary mapping config_name -> split_name -> list of file info dicts
        
    Example output:
        {
            "particles": {
                "ttbar": [
                    {
                        "filename": "events0-999.parquet",
                        "path": "/path/to/file.parquet",
                        "size_mb": 1.5,
                        "event_range": (0, 999),
                        "num_events": 1000
                    },
                    ...
                ]
            }
        }
    """
    inventory = defaultdict(lambda: defaultdict(list))
    
    for config in configs:
        config_name = config['name']
        split_name = config['split']
        local_path = Path(source_base) / config['local_path']
        
        if not local_path.exists():
            logger.warning(f"Path does not exist: {local_path}")
            continue
            
        # Find all Parquet files
        parquet_files = sorted(local_path.glob("*.parquet"))
        
        for file_path in parquet_files:
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB
            event_range = parse_event_range_from_filename(file_path.name)
            
            file_info = {
                'filename': file_path.name,
                'path': str(file_path),
                'size_mb': file_size,
                'event_range': event_range,
                'num_events': (event_range[1] - event_range[0] + 1) if event_range else None
            }
            
            inventory[config_name][split_name].append(file_info)
    
    return dict(inventory)


def compute_inventory_statistics(inventory: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Compute summary statistics from file inventory.
    
    Args:
        inventory: File inventory from build_file_inventory()
        
    Returns:
        Statistics dict with counts and sizes per config and split
    """
    stats = {}
    
    for config_name, splits in inventory.items():
        config_stats = {
            'total_files': 0,
            'total_size_mb': 0,
            'total_events': 0,
            'splits': {}
        }
        
        for split_name, files in splits.items():
            split_files = len(files)
            split_size = sum(f['size_mb'] for f in files)
            split_events = sum(f['num_events'] for f in files if f['num_events'])
            
            config_stats['splits'][split_name] = {
                'num_files': split_files,
                'size_mb': split_size,
                'num_events': split_events
            }
            
            config_stats['total_files'] += split_files
            config_stats['total_size_mb'] += split_size
            config_stats['total_events'] += split_events
        
        stats[config_name] = config_stats
    
    return stats


def merge_hf_configs(
    existing_configs: List[Dict],
    new_configs: List[Dict]
) -> List[Dict]:
    """
    Merge new configs with existing HuggingFace configs for progressive updates.
    
    This function supports:
    - Adding new configs
    - Adding new splits to existing configs
    - Updating data_files patterns for existing config+split combinations
    - Preserving configs not mentioned in new_configs
    
    Args:
        existing_configs: Existing configs from HF dataset card
        new_configs: New configs to add/update
        
    Returns:
        Merged list of configs
        
    Example:
        existing = [
            {'config_name': 'particles', 'data_files': {'ttbar': 'url1'}}
        ]
        new = [
            {'config_name': 'particles', 'data_files': {'ggf': 'url2'}},
            {'config_name': 'tracks', 'data_files': {'ttbar': 'url3'}}
        ]
        result = [
            {'config_name': 'particles', 'data_files': {'ttbar': 'url1', 'ggf': 'url2'}},
            {'config_name': 'tracks', 'data_files': {'ttbar': 'url3'}}
        ]
    """
    # Build dict for easier merging
    merged = {}
    
    # Add existing configs
    for config in existing_configs:
        config_name = config['config_name']
        merged[config_name] = {
            'config_name': config_name,
            'data_files': config.get('data_files', {}).copy()
        }
    
    # Merge new configs
    for config in new_configs:
        config_name = config['config_name']
        if config_name not in merged:
            merged[config_name] = {
                'config_name': config_name,
                'data_files': {}
            }
        
        # Merge data_files (splits)
        new_data_files = config.get('data_files', {})
        merged[config_name]['data_files'].update(new_data_files)
    
    return list(merged.values())


def generate_dataset_card(
    repo_id: str,
    base_url: str,
    inventory: Dict[str, Dict],
    configs: List[Dict],
    template_path: Optional[str] = None,
    dataset_description: Optional[str] = None,
    existing_configs: Optional[List[Dict]] = None
) -> str:
    """
    Generate HuggingFace dataset card (README.md) with YAML front matter.
    
    Args:
        repo_id: HuggingFace repository ID
        base_url: Base NERSC URL for data files
        inventory: File inventory with metadata
        configs: List of config dicts from upload config
        template_path: Optional path to custom README template
        dataset_description: Optional dataset description
        existing_configs: Optional existing configs to merge with
        
    Returns:
        Complete README.md content with YAML front matter
    """
    # If template provided, use it
    if template_path and os.path.exists(template_path):
        with open(template_path, 'r') as f:
            return f.read()
    
    # Otherwise, auto-generate
    stats = compute_inventory_statistics(inventory)
    
    # Build configs for YAML front matter
    hf_configs = []
    for config in configs:
        config_name = config['name']
        split_name = config['split']
        nersc_subpath = config['nersc_subpath']
        
        # Build data_files URL with glob pattern
        data_url = f"{base_url}/{nersc_subpath}/*.parquet"
        
        # Check if config already exists
        existing_config = None
        if existing_configs:
            existing_config = next(
                (c for c in existing_configs if c['config_name'] == config_name),
                None
            )
        
        if existing_config:
            # Update existing config with new split
            existing_config['data_files'][split_name] = data_url
            hf_configs.append(existing_config)
        else:
            # Create new config
            hf_configs.append({
                'config_name': config_name,
                'data_files': {split_name: data_url}
            })
    
    # Deduplicate configs by name
    final_configs = {}
    for config in hf_configs:
        name = config['config_name']
        if name in final_configs:
            # Merge data_files
            final_configs[name]['data_files'].update(config['data_files'])
        else:
            final_configs[name] = config
    
    hf_configs = list(final_configs.values())
    
    # Generate YAML front matter
    yaml_configs = []
    for config in hf_configs:
        config_str = f"- config_name: {config['config_name']}\n"
        config_str += "  data_files:\n"
        for split, url in config['data_files'].items():
            config_str += f"    {split}: \"{url}\"\n"
        yaml_configs.append(config_str)
    
    yaml_front_matter = f"""---
license: cc-by-4.0
task_categories:
- other
tags:
- physics
- particle-physics
- high-energy-physics
- simulation
- collider-ml
pretty_name: {repo_id.split('/')[-1]}
size_categories:
- n<1M
configs:
{''.join(yaml_configs)}---

"""
    
    # Generate dataset description
    if not dataset_description:
        dataset_description = f"# {repo_id.split('/')[-1]}\n\nParticle physics simulation data from ColliderML, hosted on NERSC and streamed via HuggingFace.\n"
    
    # Generate statistics table
    stats_table = "\n## Dataset Statistics\n\n"
    stats_table += "| Configuration | Split | Files | Events | Size (MB) |\n"
    stats_table += "|---------------|-------|-------|--------|----------|\n"
    
    for config_name, config_stats in stats.items():
        for split_name, split_stats in config_stats['splits'].items():
            stats_table += f"| {config_name} | {split_name} | "
            stats_table += f"{split_stats['num_files']} | "
            stats_table += f"{split_stats['num_events']:,} | "
            stats_table += f"{split_stats['size_mb']:.1f} |\n"
    
    # Generate quick start
    quick_start = "\n## Quick Start\n\n"
    quick_start += "```python\n"
    quick_start += "from datasets import load_dataset\n\n"
    quick_start += f'# Load dataset (streaming mode)\n'
    quick_start += f'dataset = load_dataset(\n'
    quick_start += f'    "{repo_id}",\n'
    
    if hf_configs:
        first_config = hf_configs[0]['config_name']
        first_split = list(hf_configs[0]['data_files'].keys())[0]
        quick_start += f'    "{first_config}",\n'
        quick_start += f'    split="{first_split}",\n'
    
    quick_start += '    streaming=True\n'
    quick_start += ')\n\n'
    quick_start += '# Process events\n'
    quick_start += 'for event in dataset:\n'
    quick_start += '    print(f"Event {event[\'event_id\']}:")\n'
    quick_start += '    break\n'
    quick_start += '```\n'
    
    # Generate configurations section
    configs_section = "\n## Configurations\n\n"
    for config in hf_configs:
        configs_section += f"### `{config['config_name']}`\n\n"
        configs_section += f"**Splits:** {', '.join(config['data_files'].keys())}\n\n"
    
    # Data source and citation
    footer = f"""
## Data Source

Data is hosted on NERSC Perlmutter and streamed directly via HTTPS:
- **Base URL**: {base_url}
- **Format**: Parquet files with event-based naming
- **No download required**: Data streams on-demand

## Citation

```bibtex
@dataset{{colliderml2025,
  title={{ColliderML: High-Energy Physics Simulations for Machine Learning}},
  author={{ColliderML Collaboration}},
  year={{2025}},
  publisher={{NERSC}},
  url={{https://portal.nersc.gov/project/m4958/}}
}}
```

## Acknowledgments

Data generated using the ColliderML pipeline on NERSC Perlmutter infrastructure.
"""
    
    # Combine all sections
    readme = yaml_front_matter + dataset_description + stats_table + configs_section + quick_start + footer
    
    return readme


