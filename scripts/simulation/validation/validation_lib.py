#!/usr/bin/env python3
"""
Validation Library for Pipeline Stage Outputs

This library provides file size-based validation for pipeline stage outputs.
Based on the successful pattern from validate_pythia_generation.py.

Usage:
    from validation_lib import validate_stage
    
    result = validate_stage(
        runs_dir="/path/to/runs",
        stage="simulation",
        validation_rules=rules_dict
    )
    
    print(f"Status: {result['status']}")
    print(f"Failure rate: {result['failure_rate']:.1%}")
"""

import argparse
import logging
from pathlib import Path
import sys
import yaml
import statistics
from glob import glob
from typing import Dict, List, Tuple, Optional
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_validation_rules(rules_path: Path) -> dict:
    """
    Load validation rules from YAML file.
    
    Args:
        rules_path: Path to validation_rules.yaml
        
    Returns:
        Dictionary of validation rules
    """
    with open(rules_path, 'r') as f:
        rules = yaml.safe_load(f)
    return rules


def get_run_directories(runs_dir: Path) -> List[Path]:
    """
    Get all numeric run directories, sorted by run number.
    
    Args:
        runs_dir: Path to runs directory
        
    Returns:
        Sorted list of run directory paths
    """
    if not runs_dir.is_dir():
        logger.error(f"Runs directory does not exist: {runs_dir}")
        return []
    
    # Only numeric-named subdirectories
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    run_dirs = sorted(run_dirs, key=lambda p: int(p.name))
    
    return run_dirs


def check_file_pattern(run_dir: Path, pattern: str, check_type: str = "size") -> Tuple[bool, Optional[int], str]:
    """
    Check if files matching pattern exist in run directory and get size.
    
    Args:
        run_dir: Path to run directory
        pattern: Glob pattern to match (e.g., "*.hepmc3")
        check_type: "exists" or "size"
        
    Returns:
        Tuple of (files_found, total_size_bytes, issue_description)
        - files_found: True if at least one matching file found
        - total_size_bytes: Total size of all matching files (None if check_type="exists")
        - issue_description: Empty string if OK, otherwise describes the issue
    """
    # Find matching files
    matching_files = list(run_dir.glob(pattern))
    
    if not matching_files:
        return False, None, f"No files matching pattern '{pattern}'"
    
    if check_type == "exists":
        # Just check existence
        return True, None, ""
    
    # check_type == "size"
    total_size = 0
    for file_path in matching_files:
        try:
            total_size += file_path.stat().st_size
        except Exception as e:
            return False, None, f"Failed to get size of {file_path.name}: {e}"
    
    return True, total_size, ""


def calculate_size_statistics(sizes: List[int]) -> Dict[str, float]:
    """
    Calculate statistics for file sizes.
    
    Args:
        sizes: List of file sizes in bytes
        
    Returns:
        Dictionary with median, min, max in MB
    """
    if not sizes:
        return {}
    
    sizes_mb = [s / (1024**2) for s in sizes]
    
    return {
        "median_size_mb": statistics.median(sizes_mb),
        "min_size_mb": min(sizes_mb),
        "max_size_mb": max(sizes_mb),
        "mean_size_mb": statistics.mean(sizes_mb),
        "count": len(sizes)
    }


def validate_stage(
    runs_dir: Path,
    stage: str,
    validation_rules: dict,
    dry_run: bool = False,
    run_ids: list = None
) -> dict:
    """
    Validate outputs for a pipeline stage.
    
    Args:
        runs_dir: Path to runs directory (or version directory for stages with custom output_location)
        stage: Stage name (must match key in validation_rules)
        validation_rules: Validation rules dictionary
        dry_run: If True, log actions but don't modify anything
        run_ids: Optional list of specific run IDs to validate
        
    Returns:
        Dictionary with validation results
    """
    logger.info(f"=" * 80)
    logger.info(f"Validating stage: {stage}")
    logger.info(f"Base directory: {runs_dir}")
    logger.info(f"=" * 80)
    
    # Get stage rules
    if stage not in validation_rules.get('stages', {}):
        logger.error(f"No validation rules found for stage: {stage}")
        return {
            "stage": stage,
            "status": "CONFIGURATION_ERROR",
            "error": f"No rules defined for stage '{stage}'"
        }
    
    stage_rules = validation_rules['stages'][stage]
    file_patterns = stage_rules.get('file_patterns', [])
    output_location = stage_rules.get('output_location')  # Optional: alternative output directory
    
    if not file_patterns:
        logger.warning(f"No file patterns defined for stage {stage}")
    
    # Determine search directories based on output_location
    if output_location:
        # Custom output location (e.g., parquet/): treat the output dir as a single "run"
        search_dir = runs_dir.parent / output_location if runs_dir.name == "runs" else runs_dir / output_location
        logger.info(f"Using custom output location: {search_dir}")
        
        if not search_dir.exists():
            logger.error(f"Output directory does not exist: {search_dir}")
            return {
                "stage": stage,
                "status": "COMPLETE_FAILURE",
                "total_runs": 0,
                "successful_runs": 0,
                "failed_runs": 0,
                "failure_rate": 1.0,
                "error": f"Output directory not found: {search_dir}"
            }
        
        # Create a pseudo "run directory" list with just the output dir
        run_dirs = [search_dir]
        logger.info(f"Validating aggregated outputs in {output_location}/")
    else:
        # Standard per-run validation
        all_run_dirs = get_run_directories(Path(runs_dir))
    
    # Filter to specific run IDs if provided
    if run_ids is not None:
        run_ids_set = set(run_ids)
        run_dirs = [d for d in all_run_dirs if int(d.name) in run_ids_set]
        logger.info(f"Filtered to {len(run_dirs)} runs (from {len(all_run_dirs)} total) based on run_ids filter")
    else:
        run_dirs = all_run_dirs
    
    if not run_dirs:
        logger.error(f"No run directories found in {runs_dir}")
        return {
            "stage": stage,
            "status": "COMPLETE_FAILURE",
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "failure_rate": 1.0,
            "failed_run_ids": [],
            "failure_reasons": {},
            "error": "No run directories found"
        }
    
    logger.info(f"Found {len(run_dirs)} run directories to validate")
    
    # Validate each pattern across all runs
    failed_runs = set()  # Use set to avoid duplicates
    failure_reasons = {}  # run_id -> list of reasons
    pattern_statistics = {}
    
    for pattern_config in file_patterns:
        pattern = pattern_config['pattern']
        check_type = pattern_config.get('check_type', 'size')
        min_size_mb = pattern_config.get('min_size_mb', 0)
        median_threshold_pct = pattern_config.get('median_threshold_pct', 0.8)
        required = pattern_config.get('required', True)
        
        logger.info(f"\nChecking pattern: {pattern}")
        
        # Collect sizes and check each run
        sizes = []
        run_sizes = {}  # run_id -> size
        run_issues = {}  # run_id -> issue description
        
        for run_dir in run_dirs:
            run_id = run_dir.name
            files_found, size, issue = check_file_pattern(run_dir, pattern, check_type)
            
            if not files_found:
                if required:
                    logger.warning(f"  Run {run_id}: {issue}")
                    run_issues[run_id] = issue
                    failed_runs.add(run_id)
                    if run_id not in failure_reasons:
                        failure_reasons[run_id] = []
                    failure_reasons[run_id].append(f"{pattern}: {issue}")
                continue
            
            if check_type == "size" and size is not None:
                # Check absolute minimum
                size_mb = size / (1024**2)
                if size_mb < min_size_mb:
                    issue = f"{pattern}: size {size_mb:.2f} MB < minimum {min_size_mb} MB"
                    logger.warning(f"  Run {run_id}: {issue}")
                    run_issues[run_id] = issue
                    failed_runs.add(run_id)
                    if run_id not in failure_reasons:
                        failure_reasons[run_id] = []
                    failure_reasons[run_id].append(issue)
                    continue
                
                sizes.append(size)
                run_sizes[run_id] = size
        
        # Calculate statistics and check median threshold
        if sizes and check_type == "size":
            stats = calculate_size_statistics(sizes)
            pattern_statistics[pattern] = stats
            
            median_size = statistics.median(sizes)
            threshold_size = median_size * median_threshold_pct
            threshold_mb = threshold_size / (1024**2)
            
            logger.info(f"  Pattern statistics:")
            logger.info(f"    Median size: {stats['median_size_mb']:.2f} MB")
            logger.info(f"    Threshold ({median_threshold_pct*100:.0f}% of median): {threshold_mb:.2f} MB")
            logger.info(f"    Range: {stats['min_size_mb']:.2f} - {stats['max_size_mb']:.2f} MB")
            
            # Check for size outliers
            for run_id, size in run_sizes.items():
                if size < threshold_size:
                    size_mb = size / (1024**2)
                    issue = f"{pattern}: size {size_mb:.2f} MB < threshold {threshold_mb:.2f} MB"
                    logger.warning(f"  Run {run_id}: {issue}")
                    failed_runs.add(run_id)
                    if run_id not in failure_reasons:
                        failure_reasons[run_id] = []
                    failure_reasons[run_id].append(issue)
    
    # Determine overall status
    total_runs = len(run_dirs)
    failed_runs_count = len(failed_runs)
    successful_runs_count = total_runs - failed_runs_count
    failure_rate = failed_runs_count / total_runs if total_runs > 0 else 0.0
    
    if failed_runs_count == 0:
        status = "SUCCESS"
    elif failed_runs_count == total_runs:
        status = "COMPLETE_FAILURE"
    else:
        status = "PARTIAL_FAILURE"
    
    # Format failure reasons (list to string)
    formatted_failure_reasons = {
        run_id: "; ".join(reasons)
        for run_id, reasons in failure_reasons.items()
    }
    
    # Create command-ready format for re-running failed runs
    failed_run_ids_sorted = sorted(list(failed_runs), key=int)
    rerun_command = ""
    if failed_run_ids_sorted:
        rerun_command = f"--run-list {' '.join(map(str, failed_run_ids_sorted))}"
    
    result = {
        "stage": stage,
        "status": status,
        "total_runs": total_runs,
        "successful_runs": successful_runs_count,
        "failed_runs": failed_runs_count,
        "failure_rate": failure_rate,
        "failed_run_ids": failed_run_ids_sorted,
        "rerun_command": rerun_command,
        "failure_reasons": formatted_failure_reasons,
        "statistics": pattern_statistics
    }
    
    # Log summary
    logger.info(f"\n" + "=" * 80)
    logger.info(f"VALIDATION SUMMARY - {stage}")
    logger.info(f"=" * 80)
    logger.info(f"Status: {status}")
    logger.info(f"Total runs: {total_runs}")
    logger.info(f"Successful: {successful_runs_count}")
    logger.info(f"Failed: {failed_runs_count}")
    logger.info(f"Failure rate: {failure_rate:.1%}")
    
    if failed_runs:
        logger.info(f"\nFailed runs: {', '.join(sorted(list(failed_runs), key=int)[:20])}")
        if len(failed_runs) > 20:
            logger.info(f"  ... and {len(failed_runs) - 20} more")
        
        # Show rerun command
        logger.info(f"\nTo re-run failed runs only:")
        logger.info(f"  {rerun_command}")
    
    return result


def main():
    """Main entry point for standalone usage."""
    parser = argparse.ArgumentParser(
        description="Validate pipeline stage outputs using file size-based approach"
    )
    parser.add_argument(
        "--runs-dir",
        required=True,
        help="Directory containing run subdirectories"
    )
    parser.add_argument(
        "--stage",
        required=True,
        help="Pipeline stage name (must match validation_rules.yaml)"
    )
    parser.add_argument(
        "--rules",
        default=None,
        help="Path to validation_rules.yaml (default: look in same directory as script)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save JSON validation report"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode (no modifications)"
    )
    
    args = parser.parse_args()
    
    # Load validation rules
    if args.rules:
        rules_path = Path(args.rules)
    else:
        # Look in same directory as script
        script_dir = Path(__file__).parent
        rules_path = script_dir / "validation_rules.yaml"
    
    if not rules_path.exists():
        logger.error(f"Validation rules file not found: {rules_path}")
        sys.exit(2)
    
    logger.info(f"Loading validation rules from: {rules_path}")
    validation_rules = load_validation_rules(rules_path)
    
    # Run validation
    result = validate_stage(
        runs_dir=Path(args.runs_dir),
        stage=args.stage,
        validation_rules=validation_rules,
        dry_run=args.dry_run
    )
    
    # Save JSON report if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Validation report saved to: {output_path}")
    
    # Exit with appropriate code
    if result['status'] == 'SUCCESS':
        logger.info("✓ Validation PASSED")
        sys.exit(0)
    else:
        logger.error(f"✗ Validation FAILED: {result['status']}")
        sys.exit(1)


if __name__ == "__main__":
    main()

