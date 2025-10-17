#!/usr/bin/env python3
"""
Validate HepMC split files by comparing sizes and event counts.
Checks that split files (run i and run i+N) match the original.
"""

import argparse
from pathlib import Path
from tqdm import tqdm


def get_file_size(filepath):
    """Get file size in bytes, return 0 if doesn't exist."""
    if filepath.exists():
        return filepath.stat().st_size
    return 0


def count_events_fast(filepath):
    """Count events in HepMC file by counting 'E ' lines."""
    if not filepath.exists():
        return 0
    
    count = 0
    with open(filepath, 'rb') as f:
        for line in f:
            if line.startswith(b'E '):
                count += 1
    return count


def validate_single_run(runs_dir, run_num, offset, check_events=False):
    """
    Validate a single run split.
    Returns: dict with validation results
    """
    original_file = runs_dir / str(run_num) / "merged_events_original.hepmc3"
    first_split = runs_dir / str(run_num) / "merged_events.hepmc3"
    second_split = runs_dir / str(run_num + offset) / "merged_events.hepmc3"
    
    result = {
        'run': run_num,
        'original_exists': original_file.exists(),
        'first_split_exists': first_split.exists(),
        'second_split_exists': second_split.exists(),
    }
    
    if not result['original_exists']:
        result['status'] = 'missing_original'
        return result
    
    if not result['first_split_exists'] or not result['second_split_exists']:
        result['status'] = 'missing_splits'
        return result
    
    # Check file sizes
    original_size = get_file_size(original_file)
    first_size = get_file_size(first_split)
    second_size = get_file_size(second_split)
    combined_size = first_size + second_size
    
    result['original_size'] = original_size
    result['first_size'] = first_size
    result['second_size'] = second_size
    result['combined_size'] = combined_size
    
    # Size difference (split files will be slightly larger due to duplicated headers)
    size_diff = combined_size - original_size
    result['size_diff'] = size_diff
    result['size_diff_pct'] = (size_diff / original_size * 100) if original_size > 0 else 0
    
    # Check events if requested
    if check_events:
        original_events = count_events_fast(original_file)
        first_events = count_events_fast(first_split)
        second_events = count_events_fast(second_split)
        
        result['original_events'] = original_events
        result['first_events'] = first_events
        result['second_events'] = second_events
        result['combined_events'] = first_events + second_events
        result['events_match'] = (original_events == first_events + second_events)
    
    # Determine status
    # Size should be slightly larger due to duplicated header (typically < 1%)
    if check_events:
        if result['events_match']:
            result['status'] = 'valid'
        else:
            result['status'] = 'event_mismatch'
    else:
        # Without event check, use size heuristic (combined should be within 2% of original)
        if abs(result['size_diff_pct']) < 2.0:
            result['status'] = 'likely_valid'
        else:
            result['status'] = 'size_suspicious'
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Validate HepMC split files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick validation (size check only)
  %(prog)s /path/to/runs -N 10000
  
  # Full validation (count events - slower)
  %(prog)s /path/to/runs -N 10000 --check-events
  
  # Validate specific range
  %(prog)s /path/to/runs -N 10000 --min-run 0 --max-run 99
        """
    )
    parser.add_argument('runs_dir', type=str,
                        help='Directory containing numbered run subdirectories')
    parser.add_argument('-N', '--offset', type=int, required=True,
                        help='Offset used for new run directories')
    parser.add_argument('--min-run', type=int, default=0,
                        help='Minimum run number to validate')
    parser.add_argument('--max-run', type=int, default=None,
                        help='Maximum run number to validate')
    parser.add_argument('--check-events', action='store_true',
                        help='Count and verify events (slower but thorough)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show details for each run')
    
    args = parser.parse_args()
    
    runs_dir = Path(args.runs_dir)
    
    if not runs_dir.exists():
        print(f"Error: Directory '{runs_dir}' not found")
        return 1
    
    # Find all run directories with originals
    print(f"Scanning for original files in {runs_dir}...")
    run_nums = []
    for d in runs_dir.iterdir():
        if d.is_dir() and d.name.isdigit():
            run_num = int(d.name)
            if run_num >= args.min_run:
                if args.max_run is None or run_num <= args.max_run:
                    original = d / "merged_events_original.hepmc3"
                    if original.exists():
                        run_nums.append(run_num)
    
    run_nums.sort()
    
    if not run_nums:
        print("No runs with original files found!")
        return 1
    
    print(f"Found {len(run_nums)} runs to validate (runs {run_nums[0]} to {run_nums[-1]})")
    print(f"Offset: +{args.offset}")
    if args.check_events:
        print("Mode: Full validation (counting events)")
    else:
        print("Mode: Quick validation (size check only)")
    print()
    
    # Validate all runs
    results = []
    for run_num in tqdm(run_nums, desc="Validating runs", unit="run"):
        result = validate_single_run(runs_dir, run_num, args.offset, args.check_events)
        results.append(result)
        
        if args.verbose:
            status = result['status']
            if status == 'valid' or status == 'likely_valid':
                print(f"  Run {run_num}: ✓ {status}")
            else:
                print(f"  Run {run_num}: ✗ {status}")
    
    # Summarize results
    valid_count = sum(1 for r in results if r['status'] in ['valid', 'likely_valid'])
    error_count = len(results) - valid_count
    
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total runs checked: {len(results)}")
    print(f"Valid: {valid_count}")
    print(f"Errors/Issues: {error_count}")
    
    # Show statistics
    if results:
        valid_results = [r for r in results if 'size_diff' in r]
        if valid_results:
            avg_size_diff = sum(r['size_diff'] for r in valid_results) / len(valid_results)
            avg_size_diff_pct = sum(r['size_diff_pct'] for r in valid_results) / len(valid_results)
            print(f"\nAverage size difference: {avg_size_diff:,.0f} bytes ({avg_size_diff_pct:.2f}%)")
            print(f"  (Positive difference is expected due to duplicated headers)")
    
    if args.check_events:
        event_results = [r for r in results if 'events_match' in r]
        if event_results:
            event_matches = sum(1 for r in event_results if r['events_match'])
            print(f"\nEvent count matches: {event_matches}/{len(event_results)}")
    
    # Show errors
    errors = [r for r in results if r['status'] not in ['valid', 'likely_valid']]
    if errors:
        print(f"\n{'='*70}")
        print("ISSUES FOUND:")
        print(f"{'='*70}")
        for r in errors:
            print(f"  Run {r['run']}: {r['status']}")
            if r['status'] == 'missing_original':
                print(f"    - Original file not found")
            elif r['status'] == 'missing_splits':
                print(f"    - First split exists: {r['first_split_exists']}")
                print(f"    - Second split exists: {r['second_split_exists']}")
            elif r['status'] == 'event_mismatch':
                print(f"    - Original events: {r['original_events']}")
                print(f"    - Combined events: {r['combined_events']}")
            elif r['status'] == 'size_suspicious':
                print(f"    - Size difference: {r['size_diff_pct']:.2f}%")
    
    print(f"{'='*70}\n")
    
    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    exit(main())
