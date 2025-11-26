"""
Main test runner for ColliderML data consistency tests.

Provides functions to run all test suites and generate comprehensive reports.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime
import json

from .test_base import (
    TestResult,
    TestStatus,
    TestSuite,
    DataLoader,
    print_test_results,
)
from .test_particles import ParticleTests
from .test_tracker_hits import TrackerHitTests
from .test_tracks import TrackTests
from .test_calorimeter import CalorimeterTests
from .test_hepmc import HepMCValidationTests
from .test_cross_object import CrossObjectTests


def get_all_test_suites() -> List[TestSuite]:
    """Get all available test suites."""
    return [
        ParticleTests(),
        TrackerHitTests(),
        TrackTests(),
        CalorimeterTests(),
        HepMCValidationTests(),
        CrossObjectTests(),
    ]


def run_all_tests(
    base_path: str,
    run_id: int,
    local_event: int = 0,
    run_size: int = 128,
    chunk_size: int = 100,
    suites: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run all consistency tests for a specific event.
    
    Args:
        base_path: Base path to ColliderML data directory
        run_id: Run ID to test
        local_event: Local event index within the run (default 0)
        run_size: Number of events per run (default 128)
        chunk_size: Number of events per parquet chunk (default 100)
        suites: Optional list of suite names to run (default: all)
        verbose: Whether to print results (default True)
    
    Returns:
        Dictionary with test results and summary statistics
    """
    # Initialize data loader
    loader = DataLoader(base_path, run_id, run_size=run_size, chunk_size=chunk_size)
    
    # Get test suites
    all_suites = get_all_test_suites()
    
    if suites is not None:
        suite_map = {s.name: s for s in all_suites}
        all_suites = [suite_map[name] for name in suites if name in suite_map]
    
    # Run all tests
    all_results = {}
    total_passed = 0
    total_failed = 0
    total_skipped = 0
    total_errors = 0
    
    for suite in all_suites:
        if verbose:
            print(f"\nRunning {suite.name}...")
        
        results = suite.run_all(loader, local_event=local_event)
        all_results[suite.name] = results
        
        if verbose:
            passed, failed, skipped, errors = print_test_results(results, suite.name)
        else:
            passed = sum(1 for r in results if r.status == TestStatus.PASSED)
            failed = sum(1 for r in results if r.status == TestStatus.FAILED)
            skipped = sum(1 for r in results if r.status == TestStatus.SKIPPED)
            errors = sum(1 for r in results if r.status == TestStatus.ERROR)
        
        total_passed += passed
        total_failed += failed
        total_skipped += skipped
        total_errors += errors
    
    # Generate summary
    summary = {
        "run_id": run_id,
        "local_event": local_event,
        "global_event": run_id * run_size + local_event,
        "base_path": base_path,
        "timestamp": datetime.now().isoformat(),
        "total_tests": total_passed + total_failed + total_skipped + total_errors,
        "passed": total_passed,
        "failed": total_failed,
        "skipped": total_skipped,
        "errors": total_errors,
        "pass_rate": total_passed / (total_passed + total_failed) if (total_passed + total_failed) > 0 else 0,
    }
    
    if verbose:
        print("\n" + "=" * 80)
        print("OVERALL SUMMARY")
        print("=" * 80)
        print(f"Total tests: {summary['total_tests']}")
        print(f"  ✅ Passed: {total_passed}")
        print(f"  ❌ Failed: {total_failed}")
        print(f"  ⏭️  Skipped: {total_skipped}")
        print(f"  💥 Errors: {total_errors}")
        print(f"Pass rate: {summary['pass_rate']*100:.1f}%")
        print("=" * 80)
    
    return {
        "results": all_results,
        "summary": summary,
    }


def run_tests_multiple_events(
    base_path: str,
    run_id: int,
    event_range: range = range(0, 10),
    run_size: int = 128,
    chunk_size: int = 100,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run all tests across multiple events and return summary DataFrame.
    
    Args:
        base_path: Base path to ColliderML data directory
        run_id: Run ID to test
        event_range: Range of local events to test
        run_size: Number of events per run (default 128)
        chunk_size: Number of events per parquet chunk (default 100)
        verbose: Whether to print detailed results
    
    Returns:
        DataFrame with per-event test summary
    """
    results_list = []
    
    for local_event in event_range:
        try:
            result = run_all_tests(
                base_path=base_path,
                run_id=run_id,
                local_event=local_event,
                run_size=run_size,
                chunk_size=chunk_size,
                verbose=verbose,
            )
            results_list.append(result['summary'])
        except Exception as e:
            results_list.append({
                "run_id": run_id,
                "local_event": local_event,
                "global_event": run_id * run_size + local_event,
                "error": str(e),
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": 1,
            })
    
    return pd.DataFrame(results_list)


def generate_test_report(
    results: Dict[str, Any],
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a detailed test report.
    
    Args:
        results: Results dictionary from run_all_tests
        output_path: Optional path to save JSON report
    
    Returns:
        Formatted report string
    """
    report_lines = []
    
    summary = results['summary']
    
    report_lines.append("=" * 80)
    report_lines.append("COLLIDERML DATA CONSISTENCY TEST REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {summary['timestamp']}")
    report_lines.append(f"Run ID: {summary['run_id']}")
    report_lines.append(f"Event: {summary['local_event']} (global: {summary['global_event']})")
    report_lines.append(f"Data path: {summary['base_path']}")
    report_lines.append("")
    
    report_lines.append("SUMMARY")
    report_lines.append("-" * 40)
    report_lines.append(f"Total tests: {summary['total_tests']}")
    report_lines.append(f"  Passed: {summary['passed']}")
    report_lines.append(f"  Failed: {summary['failed']}")
    report_lines.append(f"  Skipped: {summary['skipped']}")
    report_lines.append(f"  Errors: {summary['errors']}")
    report_lines.append(f"Pass rate: {summary['pass_rate']*100:.1f}%")
    report_lines.append("")
    
    # Detailed results per suite
    for suite_name, suite_results in results['results'].items():
        report_lines.append(f"\n{suite_name}")
        report_lines.append("-" * len(suite_name))
        
        for result in suite_results:
            status_emoji = {
                TestStatus.PASSED: "✅",
                TestStatus.FAILED: "❌",
                TestStatus.SKIPPED: "⏭️",
                TestStatus.ERROR: "💥",
            }
            report_lines.append(f"  {status_emoji[result.status]} {result.name}: {result.message}")
            
            if result.status in (TestStatus.FAILED, TestStatus.ERROR) and result.details:
                for key, value in result.details.items():
                    report_lines.append(f"      {key}: {value}")
    
    report_lines.append("\n" + "=" * 80)
    
    report = "\n".join(report_lines)
    
    if output_path:
        # Save JSON report
        json_results = {
            "summary": summary,
            "results": {
                suite_name: [r.to_dict() for r in suite_results]
                for suite_name, suite_results in results['results'].items()
            }
        }
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
    
    return report


def get_failed_tests(results: Dict[str, Any]) -> List[TestResult]:
    """Get list of all failed tests from results."""
    failed = []
    for suite_results in results['results'].values():
        for result in suite_results:
            if result.status == TestStatus.FAILED:
                failed.append(result)
    return failed


def get_test_summary_df(results: Dict[str, Any]) -> pd.DataFrame:
    """Convert test results to a summary DataFrame."""
    rows = []
    for suite_name, suite_results in results['results'].items():
        for result in suite_results:
            rows.append({
                'suite': suite_name,
                'test': result.name,
                'status': result.status.value,
                'message': result.message,
                'duration_ms': result.duration_ms,
            })
    return pd.DataFrame(rows)
