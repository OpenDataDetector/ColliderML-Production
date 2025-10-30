#!/usr/bin/env python3
"""
Upload ColliderML datasets to HuggingFace Hub.

This script:
1. Validates that parquet files exist and are world-readable
2. Populates a README template with dataset information
3. Saves the README to the local data directory
4. Uploads the same README to HuggingFace Hub
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pyarrow.parquet as pq
from huggingface_hub import HfApi
from jinja2 import Template


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def find_parquet_files(data_dir: Path, object_types: List[str]) -> Dict[str, List[Path]]:
    """Find all parquet files for each object type."""
    files = {}

    for obj_type in object_types:
        if obj_type == "particles":
            subdir = data_dir / "truth" / "particles"
        else:
            subdir = data_dir / "reco" / obj_type

        if not subdir.exists():
            raise FileNotFoundError(f"Directory not found: {subdir}")

        parquet_files = sorted(subdir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {subdir}")

        files[obj_type] = parquet_files
        print(f"Found {len(parquet_files)} files for {obj_type}")

    return files


def check_file_permissions(files: Dict[str, List[Path]]) -> bool:
    """Check if all files are world-readable."""
    all_readable = True

    for obj_type, file_list in files.items():
        for file_path in file_list:
            mode = file_path.stat().st_mode
            world_readable = bool(mode & 0o004)

            if not world_readable:
                print(f"WARNING: File not world-readable: {file_path}")
                all_readable = False

    return all_readable


def fix_file_permissions(files: Dict[str, List[Path]]) -> None:
    """Make all files and parent directories world-readable."""
    import stat

    # Fix directory permissions first
    dirs_to_fix = set()
    for obj_type, file_list in files.items():
        for file_path in file_list:
            # Add all parent directories up to data_dir
            for parent in file_path.parents:
                dirs_to_fix.add(parent)

    for dir_path in sorted(dirs_to_fix):
        try:
            current_mode = dir_path.stat().st_mode
            new_mode = current_mode | stat.S_IROTH | stat.S_IXOTH
            dir_path.chmod(new_mode)
        except Exception as e:
            print(f"Warning: Could not fix permissions for {dir_path}: {e}")

    # Fix file permissions
    for obj_type, file_list in files.items():
        for file_path in file_list:
            current_mode = file_path.stat().st_mode
            new_mode = current_mode | stat.S_IROTH
            file_path.chmod(new_mode)

    print("✅ Fixed file and directory permissions")


def get_parquet_schema(file_path: Path) -> Dict[str, str]:
    """Extract schema from a parquet file."""
    table = pq.read_table(file_path)
    schema = {}

    for field in table.schema:
        schema[field.name] = str(field.type)

    return schema


def estimate_total_events(files: Dict[str, List[Path]]) -> int:
    """Estimate total number of events from file names."""
    total = 0

    # Use first object type to count events
    first_obj = list(files.keys())[0]
    file_list = files[first_obj]

    for file_path in file_list:
        # Parse event range from filename
        # Format: campaign.dataset.version.category.object.eventsXXXX-YYYY.parquet
        filename = file_path.stem
        parts = filename.split('.')

        # Find events range
        for part in parts:
            if part.startswith('events'):
                event_range = part.replace('events', '')
                if '-' in event_range:
                    start, end = event_range.split('-')
                    # Add 1 because range is inclusive
                    total += int(end) - int(start) + 1

    return total


def generate_file_urls(files: Dict[str, List[Path]], base_url: str, data_dir: Path) -> Dict[str, List[str]]:
    """Generate public URLs for all files."""
    urls = {}

    for obj_type, file_list in files.items():
        urls[obj_type] = []
        for file_path in file_list:
            # Get relative path from data_dir
            rel_path = file_path.relative_to(data_dir)
            url = f"{base_url}/{rel_path}"
            urls[obj_type].append(url)

    return urls


def populate_readme_template(
    template_path: Path,
    config: Dict[str, Any],
    urls: Dict[str, List[str]],
    schemas: Dict[str, Dict[str, str]],
    total_events: int
) -> str:
    """Populate README template with dataset information."""

    # Load template
    with open(template_path, 'r') as f:
        template_text = f.read()

    template = Template(template_text)

    # Get first file as example
    first_obj = list(urls.keys())[0]
    file_example = urls[first_obj][0].split('/')[-1] if urls[first_obj] else "example.parquet"

    # Get license name
    license_map = {
        'cc-by-4.0': 'Creative Commons Attribution 4.0 International (CC BY 4.0)',
        'mit': 'MIT License',
        'apache-2.0': 'Apache License 2.0'
    }
    license_name = license_map.get(config['huggingface']['license'], config['huggingface']['license'])

    # Prepare template variables
    template_vars = {
        # Basic metadata
        'campaign': config['campaign'],
        'dataset': config['dataset'],
        'version': config['version'],
        'pileup': config.get('pileup', 0),
        'year': config.get('year', 2024),
        'date': config.get('date', 'October 2024'),

        # HuggingFace metadata
        'repo_id': config['huggingface']['repo_id'],
        'pretty_name': config['huggingface']['pretty_name'],
        'process_description': config['huggingface']['process_description'],
        'process_description_long': config['huggingface'].get('process_description_long', config['huggingface']['process_description']),
        'license': config['huggingface']['license'],
        'license_name': license_name,
        'tags': config['huggingface']['tags'],
        'size_category': config['huggingface']['size_category'],
        'contact': config['huggingface'].get('contact', 'danieltm@lbl.gov'),

        # Data info
        'total_events': total_events,
        'num_configs': len(urls),
        'public_url_base': config['data']['public_url_base'],
        'data_files': urls,
        'schemas': schemas,
        'file_example': file_example,

        # Optional metadata
        'curation_notes': config.get('metadata', {}).get('notes', ''),
        'related_datasets': config.get('related_datasets', []),
        'related_status': config.get('related_status', 'coming soon'),
    }

    # Populate template
    readme = template.render(**template_vars)

    return readme


def save_local_readme(readme_content: str, data_dir: Path) -> Path:
    """Save README to local data directory."""
    readme_path = data_dir / "README.md"

    readme_path.write_text(readme_content)
    print(f"✅ Saved local README: {readme_path}")

    return readme_path


def upload_to_huggingface(readme_path: Path, repo_id: str, commit_message: str = None) -> None:
    """Upload README to HuggingFace."""

    if commit_message is None:
        commit_message = "Update dataset card"

    api = HfApi()
    print(f"\nUploading to HuggingFace: {repo_id}")

    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message
    )

    print(f"✅ Successfully uploaded to HuggingFace!")
    print(f"Dataset URL: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload ColliderML dataset to HuggingFace Hub"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--fix-permissions",
        action="store_true",
        help="Automatically fix file permissions if not world-readable"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate README but don't upload to HuggingFace"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save generated README to custom path (in addition to data directory)"
    )
    parser.add_argument(
        "--template",
        type=str,
        help="Path to custom README template (default: scripts/dataset/README_template.md)"
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Validate configuration
    required_keys = ['campaign', 'dataset', 'version', 'data', 'huggingface', 'objects']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key in config: {key}")

    # Find parquet files
    data_dir = Path(config['data']['base_dir'])
    print(f"\nSearching for parquet files in: {data_dir}")

    files = find_parquet_files(data_dir, config['objects'])

    # Check permissions
    print("\nChecking file permissions...")
    all_readable = check_file_permissions(files)

    if not all_readable:
        if args.fix_permissions:
            print("\nFixing file permissions...")
            fix_file_permissions(files)
        else:
            print("\nWARNING: Some files are not world-readable!")
            print("Run with --fix-permissions to fix automatically, or run:")
            print(f"  chmod -R a+rX {data_dir}")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                sys.exit(1)
    else:
        print("✅ All files are world-readable")

    # Generate file URLs
    print("\nGenerating file URLs...")
    urls = generate_file_urls(files, config['data']['public_url_base'], data_dir)

    # Get schema from first file of each type
    print("\nExtracting schemas...")
    schemas = {}
    for obj_type, file_list in files.items():
        schemas[obj_type] = get_parquet_schema(file_list[0])

    # Estimate total events
    total_events = estimate_total_events(files)
    print(f"\nEstimated total events: {total_events}")

    # Find template
    if args.template:
        template_path = Path(args.template)
    else:
        # Default template location
        script_dir = Path(__file__).parent
        template_path = script_dir / "README_template.md"

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    print(f"\nUsing template: {template_path}")

    # Populate template
    print("Populating README template...")
    readme_content = populate_readme_template(
        template_path, config, urls, schemas, total_events
    )

    # Save local README
    print("\nSaving README to data directory...")
    local_readme_path = save_local_readme(readme_content, data_dir)

    # Save to custom output location if specified
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(readme_content)
        print(f"✅ Also saved to: {output_path}")

    # Upload to HuggingFace
    if args.dry_run:
        print("\n" + "="*70)
        print("DRY RUN - README generated but not uploaded")
        print("="*70)
        print(f"README saved to: {local_readme_path}")
        print(f"\nTo upload, run without --dry-run")

    else:
        # Upload to HuggingFace
        commit_msg = f"Update {config['campaign']}/{config['dataset']}/{config['version']} dataset card"
        upload_to_huggingface(local_readme_path, config['huggingface']['repo_id'], commit_msg)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
