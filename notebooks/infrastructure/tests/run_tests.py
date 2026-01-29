#!/usr/bin/env python3
"""
Command-line runner for ColliderML data consistency tests.

Usage:
    python run_tests.py --config /path/to/config.yaml
    python run_tests.py --config /path/to/config.yaml --run-id 5 --event 10
    python run_tests.py --base-path /path/to/data --run-id 0 --run-size 64 --chunk-size 100
    python run_tests.py --list
"""

import argparse
import sys
import time
from pathlib import Path

# Set up path so we can import tests as a package
TESTS_DIR = Path(__file__).parent
PARENT_DIR = TESTS_DIR.parent

# Add parent directory to path and import tests as package
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from tests.test_base import DataLoader, TestStatus, print_test_results
from tests.test_particles import ParticleTests
from tests.test_tracker_hits import TrackerHitTests
from tests.test_tracks import TrackTests
from tests.test_calorimeter import CalorimeterTests
from tests.test_hepmc import HepMCValidationTests
from tests.test_cross_object import CrossObjectTests


# All available test suites
TEST_SUITES = {
    "particles": ParticleTests,
    "tracker_hits": TrackerHitTests,
    "tracks": TrackTests,
    "calorimeter": CalorimeterTests,
    "hepmc": HepMCValidationTests,
    "cross_object": CrossObjectTests,
}


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def list_tests():
    """Print all available test suites and their tests."""
    print("\n" + "=" * 80)
    print("Available Test Suites")
    print("=" * 80)
    
    for suite_name, suite_class in TEST_SUITES.items():
        suite = suite_class()
        print(f"\n📦 {suite_name}: {suite.name}")
        print(f"   {suite.description}")
        for test in suite.tests:
            print(f"   • {test.name}")
    print()


def run_suite(suite_name: str, loader: DataLoader, local_event: int, verbose: bool = True):
    """Run a single test suite and return results."""
    if suite_name not in TEST_SUITES:
        print(f"❌ Unknown suite: {suite_name}")
        print(f"   Available: {', '.join(TEST_SUITES.keys())}")
        return None
    
    suite = TEST_SUITES[suite_name]()
    
    if verbose:
        print(f"\n🧪 Running: {suite.name}")
        print(f"   {suite.description}")
        print("-" * 60)
    
    results = suite.run_all(loader, local_event=local_event)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run ColliderML data consistency tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests using a config file
  python run_tests.py --config /path/to/config.yaml

  # Override config values from command line
  python run_tests.py --config /path/to/config.yaml --run-id 5 --event 10

  # Run without config (all required args must be specified)
  python run_tests.py --base-path /path/to/data --run-id 0 --run-size 64 --chunk-size 100

  # Run only specific test suites
  python run_tests.py --config /path/to/config.yaml --suite particles --suite tracks

  # Get JSON output (for CI/automation)
  python run_tests.py --config /path/to/config.yaml --json

  # List all available tests
  python run_tests.py --list
        """
    )
    
    parser.add_argument("--config", "-c", type=str,
                        help="Path to YAML config file")
    parser.add_argument("--base-path", "-b", type=str,
                        help="Base path to ColliderML data directory (required if no config)")
    parser.add_argument("--run-id", "-r", type=int,
                        help="Run ID to test")
    parser.add_argument("--run-size", type=int,
                        help="Number of events per run (required if no config)")
    parser.add_argument("--chunk-size", type=int,
                        help="Parquet chunk size (required if no config)")
    parser.add_argument("--event", "-e", type=int, default=0,
                        help="Local event index to test (default: 0)")
    parser.add_argument("--suite", "-s", type=str, action="append",
                        choices=list(TEST_SUITES.keys()),
                        help="Test suite(s) to run (default: all)")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List all available tests and exit")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--json", "-j", action="store_true",
                        help="Output results as JSON")
    
    args = parser.parse_args()
    
    if args.list:
        list_tests()
        return 0
    
    # Load config if provided
    config = {}
    if args.config:
        config = load_config(args.config)
        print(f"📄 Loaded config: {args.config}")
    
    # Merge config with command-line args (CLI takes precedence)
    base_path = args.base_path or config.get("base_path")
    run_id = args.run_id if args.run_id is not None else config.get("run_id")
    run_size = args.run_size or config.get("run_size")
    chunk_size = args.chunk_size or config.get("chunk_size")
    event = args.event if args.event != 0 else config.get("event", 0)
    suites = args.suite or config.get("suites")
    
    # Validate required arguments
    missing = []
    if not base_path:
        missing.append("--base-path")
    if run_id is None:
        missing.append("--run-id")
    if not run_size:
        missing.append("--run-size")
    if not chunk_size:
        missing.append("--chunk-size")
    
    if missing:
        print(f"❌ Missing required arguments: {', '.join(missing)}")
        print("   Provide them via --config or command-line arguments.")
        parser.print_help()
        return 1
    
    # Create data loader
    print(f"\n📂 Data path: {base_path}")
    print(f"🔢 Run ID: {run_id}, Event: {event}")
    print(f"📊 Run size: {run_size}, Chunk size: {chunk_size}")
    
    loader = DataLoader(
        base_path=base_path,
        run_id=run_id,
        run_size=run_size,
        chunk_size=chunk_size,
    )
    
    # Determine which suites to run
    suites_to_run = suites if suites else list(TEST_SUITES.keys())
    
    # Run tests
    all_results = []
    total_start = time.time()
    json_output = args.json
    verbose = args.verbose
    
    for suite_name in suites_to_run:
        results = run_suite(suite_name, loader, event, verbose=not json_output)
        if results:
            all_results.extend(results)
            if not json_output:
                print_test_results(results, suite_name)
    
    total_time = time.time() - total_start
    
    # Summary
    if json_output:
        import json
        output = {
            "base_path": base_path,
            "run_id": run_id,
            "event": event,
            "total_time_s": total_time,
            "results": [r.to_dict() for r in all_results],
        }
        print(json.dumps(output, indent=2))
    else:
        passed = sum(1 for r in all_results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in all_results if r.status == TestStatus.FAILED)
        skipped = sum(1 for r in all_results if r.status == TestStatus.SKIPPED)
        errors = sum(1 for r in all_results if r.status == TestStatus.ERROR)
        
        print("\n" + "=" * 80)
        print("📊 OVERALL SUMMARY")
        print("=" * 80)
        print(f"   ✅ Passed:  {passed}")
        print(f"   ❌ Failed:  {failed}")
        print(f"   ⏭️  Skipped: {skipped}")
        print(f"   💥 Errors:  {errors}")
        print(f"   ⏱️  Total time: {total_time:.2f}s")
        print("=" * 80)
        
        # Return non-zero exit code if any tests failed
        if failed > 0 or errors > 0:
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
