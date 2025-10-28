#!/usr/bin/env python3
"""
Run validation for a pipeline stage.

This script validates outputs from a pipeline stage by checking file sizes
and other criteria defined in validation_rules.yaml.
"""
import argparse
import sys
import json
from pathlib import Path
import logging

# Add validation lib to path
sys.path.insert(0, str(Path(__file__).parent))
from validation_lib import validate_stage, load_validation_rules

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Validate pipeline stage outputs'
    )
    parser.add_argument(
        '--stage',
        required=True,
        help='Pipeline stage name (e.g., digitization, simulation)'
    )
    parser.add_argument(
        '--runs-dir',
        required=True,
        type=Path,
        help='Path to runs directory to validate'
    )
    parser.add_argument(
        '--rules',
        type=Path,
        default=Path(__file__).parent / 'validation_rules.yaml',
        help='Path to validation rules YAML file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Path to save validation report JSON (optional)'
    )
    parser.add_argument(
        '--run-ids',
        type=int,
        nargs='+',
        help='Specific run IDs to validate (optional, defaults to all runs in directory)'
    )
    parser.add_argument(
        '--run-range',
        type=int,
        nargs=2,
        metavar=('START', 'END'),
        help='Range of run IDs to validate: START (inclusive) to END (exclusive)'
    )
    
    args = parser.parse_args()
    
    # Check that run-ids and run-range are not both specified
    if args.run_ids and args.run_range:
        logger.error("Cannot specify both --run-ids and --run-range")
        sys.exit(1)
    
    # Load validation rules
    logger.info(f"Loading validation rules from: {args.rules}")
    try:
        validation_rules = load_validation_rules(args.rules)
    except Exception as e:
        logger.error(f"Failed to load validation rules: {e}")
        sys.exit(1)
    
    # Determine which runs to validate
    run_ids_to_validate = None
    if args.run_ids:
        run_ids_to_validate = args.run_ids
        logger.info(f"Validating specific run IDs: {run_ids_to_validate}")
    elif args.run_range:
        start, end = args.run_range
        run_ids_to_validate = list(range(start, end))
        logger.info(f"Validating run range: {start} to {end-1} ({len(run_ids_to_validate)} runs)")
    else:
        logger.info("Validating all runs in directory")
    
    # Run validation
    logger.info(f"Validating stage '{args.stage}' in: {args.runs_dir}")
    try:
        result = validate_stage(
            runs_dir=args.runs_dir,
            stage=args.stage,
            validation_rules=validation_rules,
            run_ids=run_ids_to_validate
        )
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print summary
    logger.info(f"Validation status: {result.get('status', 'UNKNOWN')}")
    logger.info(f"Total runs: {result.get('total_runs', 0)}")
    logger.info(f"Successful: {result.get('successful_runs', 0)}")
    logger.info(f"Failed: {result.get('failed_runs', 0)}")
    if result.get('failed_runs', 0) > 0:
        logger.info(f"Failure rate: {result.get('failure_rate', 0):.1f}%")
    
    # Save report if output path provided
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Validation report saved to: {args.output}")
    else:
        # Print to stdout for pipeline consumption
        print(json.dumps(result, indent=2))
    
    # Exit with success
    sys.exit(0)


if __name__ == '__main__':
    main()

