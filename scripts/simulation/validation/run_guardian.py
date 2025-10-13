#!/usr/bin/env python3
"""
Run error guardian decision logic.

This script loads a validation report, applies guardian policies,
and exits with an appropriate code for SLURM action.
"""
import argparse
import sys
import json
import os
from pathlib import Path
import logging

# Add validation lib to path
sys.path.insert(0, str(Path(__file__).parent))
from error_guardian import make_decision, load_guardian_policy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Run error guardian decision logic'
    )
    parser.add_argument(
        '--report',
        required=True,
        type=Path,
        help='Path to validation report JSON file'
    )
    parser.add_argument(
        '--runs-dir',
        required=True,
        type=Path,
        help='Path to runs directory'
    )
    parser.add_argument(
        '--policy',
        type=Path,
        default=Path(__file__).parent / 'guardian_policy.yaml',
        help='Path to guardian policy YAML file'
    )
    parser.add_argument(
        '--retry-count',
        type=int,
        default=None,
        help='Retry count (defaults to SLURM_RESTART_COUNT env var)'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='Maximum number of retries allowed'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run mode - show decision but don\'t execute actions'
    )
    
    args = parser.parse_args()
    
    # Load validation report
    logger.info(f"Loading validation report from: {args.report}")
    try:
        with open(args.report, 'r') as f:
            validation_result = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load validation report: {e}")
        sys.exit(1)
    
    # Load guardian policy
    logger.info(f"Loading guardian policy from: {args.policy}")
    try:
        guardian_policy = load_guardian_policy(args.policy)
    except Exception as e:
        logger.error(f"Failed to load guardian policy: {e}")
        sys.exit(1)
    
    # Get retry count
    retry_count = args.retry_count
    if retry_count is None:
        retry_count = int(os.environ.get('SLURM_RESTART_COUNT', '0'))
    
    # Override max_retries from policy if present
    max_retries = guardian_policy.get('retry_policy', {}).get('max_retries', args.max_retries)
    
    logger.info(f"Retry count: {retry_count}/{max_retries}")
    
    # Make decision
    try:
        decision = make_decision(
            validation_result=validation_result,
            runs_dir=args.runs_dir,
            guardian_policy=guardian_policy,
            retry_count=retry_count,
            max_retries=max_retries,
            dry_run=args.dry_run
        )
    except Exception as e:
        logger.error(f"Guardian decision failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print decision
    logger.info("=" * 80)
    logger.info(f"Guardian Action: {decision.get('action', 'UNKNOWN')}")
    logger.info(f"Severity: {decision.get('severity', 'unknown')}")
    logger.info(f"Reason: {decision.get('reason', 'No reason provided')}")
    logger.info(f"Exit Code: {decision.get('exit_code', 1)}")
    
    if decision.get('actions_taken'):
        logger.info("Actions taken:")
        for action in decision['actions_taken']:
            logger.info(f"  - {action}")
    
    if args.dry_run:
        logger.info("DRY RUN - No actions executed")
    
    logger.info("=" * 80)
    
    # If guardian decides to requeue, trigger SLURM requeue before exiting
    if decision['exit_code'] == 99 and not args.dry_run:
        job_id = os.environ.get('SLURM_JOB_ID')
        if job_id:
            logger.info(f"Triggering SLURM requeue for job {job_id}")
            try:
                import subprocess
                subprocess.run(['scontrol', 'requeue', job_id], check=True)
                logger.info(f"✓ Job {job_id} requeued successfully")
                # Exit with 0 since requeue was successful
                sys.exit(0)
            except subprocess.CalledProcessError as e:
                logger.error(f"✗ Failed to requeue job {job_id}: {e}")
                logger.error("Falling back to exit code 99")
                sys.exit(99)
            except FileNotFoundError:
                logger.error("scontrol command not found - are we in a SLURM environment?")
                logger.error("Falling back to exit code 99")
                sys.exit(99)
        else:
            logger.warning("SLURM_JOB_ID not set - cannot requeue (not in SLURM job?)")
            logger.warning("Exiting with code 99 anyway")
            sys.exit(99)
    
    # Exit with guardian's decision code
    sys.exit(decision['exit_code'])


if __name__ == '__main__':
    main()

