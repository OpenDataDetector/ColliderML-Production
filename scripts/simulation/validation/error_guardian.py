#!/usr/bin/env python3
"""
Error Guardian - Decision Logic and Recovery Actions

Makes decisions based on validation results and executes recovery actions.
Integrates with SLURM requeue mechanism for automatic retry.

Usage:
    from error_guardian import make_decision
    
    decision = make_decision(
        validation_result=result,
        runs_dir=runs_directory,
        guardian_policy=policy_dict,
        retry_count=int(os.environ.get('SLURM_RESTART_COUNT', '0')),
        max_retries=3
    )
    
    sys.exit(decision['exit_code'])
"""

import argparse
import logging
from pathlib import Path
import sys
import yaml
import subprocess
import os
import json
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_guardian_policy(policy_path: Path) -> dict:
    """
    Load guardian policy from YAML file.
    
    Args:
        policy_path: Path to guardian_policy.yaml
        
    Returns:
        Dictionary of policy configuration
    """
    with open(policy_path, 'r') as f:
        policy = yaml.safe_load(f)
    return policy


def classify_failure_severity(failure_rate: float, policy: dict) -> str:
    """
    Classify failure severity based on failure rate and policy thresholds.
    
    Args:
        failure_rate: Fraction of failed runs (0.0-1.0)
        policy: Guardian policy dictionary
        
    Returns:
        Severity level: "none", "minor", "moderate", "critical"
    """
    failure_pct = failure_rate * 100
    
    thresholds = policy.get('thresholds', {})
    minor_threshold = thresholds.get('minor_failure_pct', 2.0)
    moderate_threshold = thresholds.get('moderate_failure_pct', 10.0)
    
    if failure_rate == 0.0:
        return "none"
    elif failure_pct < minor_threshold:
        return "minor"
    elif failure_pct < moderate_threshold:
        return "moderate"
    else:
        return "critical"


def lookup_failure_reason(validation_result: dict, run_id) -> str:
    """
    Look up a failure reason using int, string, or chunk-prefixed keys.

    Args:
        validation_result: Validation result dictionary.
        run_id: Failed run or chunk identifier.

    Returns:
        Human-readable failure reason.
    """
    failure_reasons = validation_result.get('failure_reasons', {})
    return (
        failure_reasons.get(run_id)
        or failure_reasons.get(str(run_id))
        or failure_reasons.get(f"chunk_{run_id}")
        or "Unknown"
    )


def remove_failed_runs(runs_dir: Path, failed_run_ids: List[str], dry_run: bool = False) -> Tuple[int, List[str]]:
    """
    Remove failed run directories.
    
    Args:
        runs_dir: Path to runs directory
        failed_run_ids: List of run IDs to remove
        dry_run: If True, don't actually delete
        
    Returns:
        Tuple of (number_removed, list_of_errors)
    """
    import shutil
    
    removed_count = 0
    errors = []
    
    logger.info(f"Removing {len(failed_run_ids)} failed run directories...")
    
    for run_id in failed_run_ids:
        run_path = runs_dir / run_id
        
        if not run_path.exists():
            logger.warning(f"  Run directory does not exist: {run_id}")
            continue
        
        if dry_run:
            logger.info(f"  [DRY RUN] Would remove: {run_path}")
            removed_count += 1
        else:
            try:
                shutil.rmtree(run_path)
                logger.info(f"  Removed: {run_path}")
                removed_count += 1
            except Exception as e:
                error_msg = f"Failed to remove {run_path}: {e}"
                logger.error(f"  {error_msg}")
                errors.append(error_msg)
    
    return removed_count, errors


def normalize_runs(runs_dir: Path, dry_run: bool = False) -> Tuple[bool, str]:
    """
    Call normalise_runs.py to renumber remaining run directories.
    
    Args:
        runs_dir: Path to runs directory
        dry_run: If True, call with --dry-run flag
        
    Returns:
        Tuple of (success, output_message)
    """
    # Find normalise_runs.py
    normalize_script = Path("/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software/colliderml_dev/scripts/postprocessing/normalise_runs.py")
    
    if not normalize_script.exists():
        error_msg = f"normalise_runs.py not found at {normalize_script}"
        logger.error(error_msg)
        return False, error_msg
    
    # Build command
    cmd = [sys.executable, str(normalize_script), str(runs_dir)]
    if dry_run:
        cmd.append("--dry-run")
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info("Normalization completed successfully")
        logger.debug(f"Output:\n{result.stdout}")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        error_msg = f"Normalization failed: {e.stderr}"
        logger.error(error_msg)
        return False, error_msg


def generate_failure_report(
    validation_result: dict,
    decision: dict,
    runs_dir: Path,
    output_path: Path = None
) -> str:
    """
    Generate detailed failure report.
    
    Args:
        validation_result: Validation result dictionary
        decision: Guardian decision dictionary
        runs_dir: Path to runs directory
        output_path: Optional path to save report
        
    Returns:
        Report text
    """
    report_lines = [
        "=" * 80,
        f"ERROR GUARDIAN FAILURE REPORT",
        "=" * 80,
        f"Stage: {validation_result['stage']}",
        f"Runs directory: {runs_dir}",
        f"Status: {validation_result['status']}",
        f"Total runs: {validation_result['total_runs']}",
        f"Successful runs: {validation_result['successful_runs']}",
        f"Failed runs: {validation_result['failed_runs']}",
        f"Failure rate: {validation_result['failure_rate']:.1%}",
        "",
        f"Failure severity: {decision['severity']}",
        f"Guardian action: {decision['action']}",
        f"Reason: {decision['reason']}",
        "",
    ]
    
    if validation_result.get('failed_run_ids'):
        report_lines.append("Failed run IDs:")
        failed_ids = validation_result['failed_run_ids']
        # Show first 50
        for run_id in failed_ids[:50]:
            reason = lookup_failure_reason(validation_result, run_id)
            report_lines.append(f"  - Run {run_id}: {reason}")
        if len(failed_ids) > 50:
            report_lines.append(f"  ... and {len(failed_ids) - 50} more")
        report_lines.append("")
    
    if validation_result.get('statistics'):
        report_lines.append("File size statistics:")
        for pattern, stats in validation_result['statistics'].items():
            report_lines.append(f"  {pattern}:")
            report_lines.append(f"    Median: {stats.get('median_size_mb', 0):.2f} MB")
            report_lines.append(f"    Range: {stats.get('min_size_mb', 0):.2f} - {stats.get('max_size_mb', 0):.2f} MB")
        report_lines.append("")
    
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report_text)
        logger.info(f"Failure report saved to: {output_path}")
    
    return report_text


def make_decision(
    validation_result: dict,
    runs_dir: Path,
    guardian_policy: dict,
    retry_count: int = 0,
    max_retries: int = 3,
    dry_run: bool = False
) -> dict:
    """
    Make decision based on validation results and execute recovery actions.
    
    Args:
        validation_result: Result from validate_stage()
        runs_dir: Path to runs directory
        guardian_policy: Guardian policy dictionary
        retry_count: Current retry count (from $SLURM_RESTART_COUNT)
        max_retries: Maximum retries allowed
        dry_run: If True, don't execute actions
        
    Returns:
        Decision dictionary:
        {
            "action": str (CONTINUE, FAIL, REQUEUE),
            "reason": str,
            "exit_code": int (0, 1, or 99),
            "severity": str,
            "retry_count": int,
            "max_retries": int,
            "actions_taken": list[str]
        }
    """
    logger.info("=" * 80)
    logger.info("ERROR GUARDIAN DECISION ENGINE")
    logger.info("=" * 80)
    # Check for configuration errors
    if validation_result['status'] == 'CONFIGURATION_ERROR':
        logger.error(f"Configuration error: {validation_result.get('error', 'Unknown error')}")
        return {
            "action": "FAIL",
            "reason": validation_result.get('error', 'Configuration error'),
            "exit_code": 1,
            "severity": "configuration_error",
            "retry_count": retry_count,
            "max_retries": max_retries,
            "actions_taken": []
        }

    logger.info(f"Validation status: {validation_result['status']}")
    logger.info(f"Failure rate: {validation_result['failure_rate']:.1%}")
    logger.info(f"Failed runs: {validation_result['failed_runs']}/{validation_result['total_runs']}")
    logger.info(f"Retry count: {retry_count}/{max_retries}")
    
    # Classify failure severity
    severity = classify_failure_severity(validation_result['failure_rate'], guardian_policy)
    logger.info(f"Failure severity: {severity}")
    
    actions_taken = []
    
    # Handle success (no failures)
    if severity == "none":
        logger.info("✓ No failures detected - continuing")
        return {
            "action": "CONTINUE",
            "reason": "All runs successful",
            "exit_code": 0,
            "severity": severity,
            "retry_count": retry_count,
            "max_retries": max_retries,
            "actions_taken": []
        }
    
    # Handle minor failures (<2%)
    if severity == "minor":
        logger.info("Minor failures detected - attempting auto-recovery")
        
        policy_actions = guardian_policy['actions']['minor_failure']
        if validation_result.get('stage') == 'convert_all':
            logger.info("Chunk-based convert_all failures detected - skipping run directory cleanup")
            return {
                "action": "CONTINUE",
                "reason": f"Minor failures ({validation_result['failure_rate']:.1%}) detected for chunk-based outputs",
                "exit_code": 0,
                "severity": severity,
                "retry_count": retry_count,
                "max_retries": max_retries,
                "actions_taken": ["Skipped run directory cleanup for convert_all chunk failures"],
            }
        
        # Remove failed runs
        if policy_actions.get('auto_remove_failed', True):
            removed, errors = remove_failed_runs(
                runs_dir,
                validation_result['failed_run_ids'],
                dry_run=dry_run
            )
            actions_taken.append(f"Removed {removed} failed run directories")
            
            if errors:
                logger.warning(f"Errors during removal: {errors}")
        
        # Normalize runs
        if policy_actions.get('auto_normalize', True):
            success, output = normalize_runs(runs_dir, dry_run=dry_run)
            if success:
                actions_taken.append("Renumbered remaining runs")
            else:
                logger.error(f"Normalization failed: {output}")
        
        logger.info(f"✓ Minor failures recovered - continuing")
        logger.info(f"Actions taken: {actions_taken}")
        
        return {
            "action": "CONTINUE",
            "reason": f"Minor failures ({validation_result['failure_rate']:.1%}) auto-recovered",
            "exit_code": 0,
            "severity": severity,
            "retry_count": retry_count,
            "max_retries": max_retries,
            "actions_taken": actions_taken
        }
    
    # Handle moderate failures (2-10%)
    if severity == "moderate":
        logger.warning("Moderate failures detected - human intervention required")
        
        # Generate failure report
        report_path = runs_dir.parent / f"failure_report_{validation_result['stage']}_moderate.txt"
        generate_failure_report(validation_result, {
            "severity": severity,
            "action": "FAIL",
            "reason": "Moderate failure rate requires human review"
        }, runs_dir, report_path)
        
        return {
            "action": "FAIL",
            "reason": f"Moderate failure rate ({validation_result['failure_rate']:.1%}) requires human intervention",
            "exit_code": 1,
            "severity": severity,
            "retry_count": retry_count,
            "max_retries": max_retries,
            "actions_taken": [f"Generated failure report: {report_path}"]
        }
    
    # Handle critical failures (>10% or complete failure)
    if severity == "critical" or validation_result['status'] == "COMPLETE_FAILURE":
        logger.error("Critical failure detected")
        
        # Check if we can retry
        if retry_count < max_retries:
            logger.warning(f"↻ Requeuing job (attempt {retry_count+1}/{max_retries})")
            
            # Generate failure report
            report_path = runs_dir.parent / f"failure_report_{validation_result['stage']}_attempt_{retry_count}.txt"
            generate_failure_report(validation_result, {
                "severity": severity,
                "action": "REQUEUE",
                "reason": f"Critical failure - retry attempt {retry_count+1}/{max_retries}"
            }, runs_dir, report_path)
            
            return {
                "action": "REQUEUE",
                "reason": f"Critical failure ({validation_result['failure_rate']:.1%}) - retrying",
                "exit_code": 99,  # SLURM requeue code
                "severity": severity,
                "retry_count": retry_count,
                "max_retries": max_retries,
                "actions_taken": [f"Generated failure report: {report_path}"]
            }
        else:
            logger.error(f"✗ Max retries ({max_retries}) exhausted - giving up")
            
            # Generate final failure report
            report_path = runs_dir.parent / f"failure_report_{validation_result['stage']}_FINAL.txt"
            generate_failure_report(validation_result, {
                "severity": severity,
                "action": "FAIL",
                "reason": f"Critical failure - max retries ({max_retries}) exhausted"
            }, runs_dir, report_path)
            
            return {
                "action": "FAIL",
                "reason": f"Critical failure ({validation_result['failure_rate']:.1%}) - max retries exhausted",
                "exit_code": 1,
                "severity": severity,
                "retry_count": retry_count,
                "max_retries": max_retries,
                "actions_taken": [f"Generated failure report: {report_path}"]
            }


def main():
    """Main entry point for standalone usage."""
    parser = argparse.ArgumentParser(
        description="Error guardian decision engine"
    )
    parser.add_argument(
        "--validation-result",
        required=True,
        help="Path to JSON validation result file"
    )
    parser.add_argument(
        "--runs-dir",
        required=True,
        help="Directory containing run subdirectories"
    )
    parser.add_argument(
        "--policy",
        default=None,
        help="Path to guardian_policy.yaml"
    )
    parser.add_argument(
        "--retry-count",
        type=int,
        default=0,
        help="Current retry count (default: from $SLURM_RESTART_COUNT)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries (default: 3)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode (no modifications)"
    )
    
    args = parser.parse_args()
    
    # Use SLURM_RESTART_COUNT if available
    if args.retry_count == 0 and 'SLURM_RESTART_COUNT' in os.environ:
        args.retry_count = int(os.environ['SLURM_RESTART_COUNT'])
    
    # Load validation result
    with open(args.validation_result, 'r') as f:
        validation_result = json.load(f)
    
    # Load guardian policy
    if args.policy:
        policy_path = Path(args.policy)
    else:
        # Look in ../configs relative to script
        script_dir = Path(__file__).parent
        policy_path = script_dir.parent / "configs" / "guardian_policy.yaml"
    
    if not policy_path.exists():
        logger.error(f"Guardian policy file not found: {policy_path}")
        sys.exit(2)
    
    logger.info(f"Loading guardian policy from: {policy_path}")
    guardian_policy = load_guardian_policy(policy_path)
    
    # Make decision
    decision = make_decision(
        validation_result=validation_result,
        runs_dir=Path(args.runs_dir),
        guardian_policy=guardian_policy,
        retry_count=args.retry_count,
        max_retries=args.max_retries,
        dry_run=args.dry_run
    )
    
    # Log decision
    logger.info("=" * 80)
    logger.info(f"Guardian decision: {decision['action']}")
    logger.info(f"Reason: {decision['reason']}")
    logger.info(f"Exit code: {decision['exit_code']}")
    if decision['actions_taken']:
        logger.info(f"Actions taken:")
        for action in decision['actions_taken']:
            logger.info(f"  - {action}")
    logger.info("=" * 80)
    
    # Exit with appropriate code
    sys.exit(decision['exit_code'])


if __name__ == "__main__":
    main()

