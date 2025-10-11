#!/usr/bin/env python3
"""
Script to normalize run directories by removing empty directories and renumbering
the remaining ones to be continuous starting from 0.

Usage:
    python normalise_runs.py <runs_directory> [--dry-run]
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


def is_directory_empty(directory):
    """
    Check if a directory is empty (ignoring hidden files).
    
    Args:
        directory: Path to directory to check
        
    Returns:
        bool: True if directory is empty, False otherwise
    """
    try:
        # Check if directory has any non-hidden files
        contents = [f for f in os.listdir(directory) if not f.startswith('.')]
        return len(contents) == 0
    except Exception as e:
        print(f"Warning: Could not check directory {directory}: {e}")
        return True


def get_run_directories(runs_path):
    """
    Get all run directories that are numeric and sort them.
    
    Args:
        runs_path: Path to the runs directory
        
    Returns:
        list: Sorted list of (run_number, directory_path, is_empty) tuples
    """
    runs = []
    
    for item in os.listdir(runs_path):
        item_path = os.path.join(runs_path, item)
        
        # Only consider directories with numeric names
        if os.path.isdir(item_path) and item.isdigit():
            run_num = int(item)
            is_empty = is_directory_empty(item_path)
            runs.append((run_num, item_path, is_empty))
    
    # Sort by run number
    runs.sort(key=lambda x: x[0])
    
    return runs


def normalise_runs(runs_path, dry_run=False):
    """
    Normalize run directories by removing empty ones and renumbering.
    
    Args:
        runs_path: Path to the runs directory
        dry_run: If True, only print what would be done without actually doing it
        
    Returns:
        tuple: (num_removed, num_renamed, non_empty_runs)
    """
    runs_path = os.path.abspath(runs_path)
    
    if not os.path.isdir(runs_path):
        raise ValueError(f"Directory does not exist: {runs_path}")
    
    print(f"Analyzing runs in: {runs_path}")
    print("-" * 80)
    
    # Get all run directories
    all_runs = get_run_directories(runs_path)
    
    # Separate empty and non-empty
    empty_runs = [r for r in all_runs if r[2]]
    non_empty_runs = [r for r in all_runs if not r[2]]
    
    print(f"Total run directories: {len(all_runs)}")
    print(f"Non-empty directories: {len(non_empty_runs)}")
    print(f"Empty directories: {len(empty_runs)}")
    print()
    
    if empty_runs:
        print(f"Empty directories to be removed (showing first 20):")
        for run_num, path, _ in empty_runs[:20]:
            print(f"  - {run_num}")
        if len(empty_runs) > 20:
            print(f"  ... and {len(empty_runs) - 20} more")
        print()
    
    # Check which directories need renaming
    rename_map = []
    for new_idx, (old_num, old_path, _) in enumerate(non_empty_runs):
        if new_idx != old_num:
            rename_map.append((old_num, new_idx, old_path))
    
    if rename_map:
        print(f"Directories to be renumbered: {len(rename_map)}")
        print(f"Showing first 20 renames:")
        for old_num, new_num, _ in rename_map[:20]:
            print(f"  {old_num} -> {new_num}")
        if len(rename_map) > 20:
            print(f"  ... and {len(rename_map) - 20} more")
        print()
    else:
        print("No directories need renumbering.")
        print()
    
    if dry_run:
        print("DRY RUN MODE - No changes made")
        return len(empty_runs), len(rename_map), non_empty_runs
    
    # Perform the actual operations
    print("=" * 80)
    print("Executing changes...")
    print("=" * 80)
    
    # First, rename all directories to temporary names to avoid conflicts
    temp_renames = []
    for old_num, new_num, old_path in rename_map:
        temp_name = f"temp_rename_{old_num}"
        temp_path = os.path.join(runs_path, temp_name)
        temp_renames.append((old_path, temp_path, new_num))
    
    # Step 1: Rename to temporary names
    print("Step 1: Renaming to temporary names...")
    for old_path, temp_path, new_num in temp_renames:
        try:
            os.rename(old_path, temp_path)
            print(f"  Renamed {os.path.basename(old_path)} -> {os.path.basename(temp_path)}")
        except Exception as e:
            print(f"  ERROR renaming {old_path} to {temp_path}: {e}")
            raise
    
    # Step 2: Rename to final names
    print("\nStep 2: Renaming to final names...")
    for old_path, temp_path, new_num in temp_renames:
        final_path = os.path.join(runs_path, str(new_num))
        try:
            os.rename(temp_path, final_path)
            print(f"  Renamed {os.path.basename(temp_path)} -> {new_num}")
        except Exception as e:
            print(f"  ERROR renaming {temp_path} to {final_path}: {e}")
            raise
    
    # Step 3: Remove empty directories
    print("\nStep 3: Removing empty directories...")
    for run_num, path, _ in empty_runs:
        try:
            os.rmdir(path)
            print(f"  Removed empty directory: {run_num}")
        except Exception as e:
            print(f"  Warning: Could not remove {path}: {e}")
    
    print("\n" + "=" * 80)
    print("Normalization complete!")
    print(f"  - Removed {len(empty_runs)} empty directories")
    print(f"  - Renumbered {len(rename_map)} directories")
    print(f"  - Final continuous range: 0 to {len(non_empty_runs) - 1}")
    print("=" * 80)
    
    return len(empty_runs), len(rename_map), non_empty_runs


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Normalize run directories by removing empty ones and renumbering"
    )
    parser.add_argument(
        "runs_directory",
        help="Path to the runs directory to normalize"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually making changes"
    )
    
    args = parser.parse_args()
    
    try:
        num_removed, num_renamed, non_empty_runs = normalise_runs(
            args.runs_directory,
            dry_run=args.dry_run
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

