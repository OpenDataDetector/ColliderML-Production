#!/usr/bin/env python3
"""
Batch convert TEfficiency objects in ROOT files to TGraphAsymmErrors + TTrees.

This script finds all ROOT files matching a pattern and converts efficiency
histograms to a format that uproot can read. Uses parallel processing for speed.

Usage:
    # Convert all performance_finding_ckf.root files in a dataset
    python batch_convert_efficiency_graphs.py /path/to/dataset --pattern "performance_finding_ckf.root"
    
    # Dry run to see what would be processed
    python batch_convert_efficiency_graphs.py /path/to/dataset --dry-run
    
    # Specify custom efficiency objects to convert
    python batch_convert_efficiency_graphs.py /path/to/dataset --keys "trackeff_vs_pT,trackeff_vs_eta,trackeff_vs_phi"
    
    # Control parallelism
    python batch_convert_efficiency_graphs.py /path/to/dataset --workers 8
"""

import sys
import argparse
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count
import traceback
from tqdm import tqdm


def find_root_files(base_dir, pattern="performance_finding_ckf.root"):
    """
    Find all ROOT files matching the pattern under base_dir.
    
    Args:
        base_dir: Base directory to search
        pattern: Filename pattern to match
        
    Returns:
        list: List of Path objects for matching files
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        raise ValueError(f"Directory does not exist: {base_dir}")
    
    # Use rglob for recursive search
    root_files = list(base_path.rglob(pattern))
    return sorted(root_files)


def convert_single_file(args):
    """
    Convert a single ROOT file using the C++ macro.
    
    Args:
        args: Tuple of (input_path, output_path, keys_csv, script_path, verbose)
        
    Returns:
        dict: Result dictionary with status and info
    """
    input_path, output_path, keys_csv, script_path, verbose = args
    
    result = {
        'input': str(input_path),
        'output': str(output_path),
        'success': False,
        'message': ''
    }
    
    try:
        # Check if output already exists
        if output_path.exists():
            result['message'] = 'Output already exists (skipped)'
            result['success'] = True
            return result
        
        # Build the ROOT command
        # Format: root -l -b -q 'script.C+("input","output","keys")'
        root_cmd = (
            f'root -l -b -q \'{script_path}+('
            f'"{input_path}",'
            f'"{output_path}",'
            f'"{keys_csv}"'
            f')\''
        )
        
        # Execute ROOT command
        process = subprocess.run(
            root_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if process.returncode != 0:
            result['message'] = f'ROOT command failed: {process.stderr[:200]}'
            return result
        
        # Check if output was created
        if not output_path.exists():
            result['message'] = 'Output file was not created'
            return result
        
        result['success'] = True
        result['message'] = 'Converted successfully'
        
        if verbose:
            print(f"✓ {input_path.name} -> {output_path.name}")
        
    except subprocess.TimeoutExpired:
        result['message'] = 'Conversion timed out (>5min)'
    except Exception as e:
        result['message'] = f'Exception: {str(e)}'
        if verbose:
            traceback.print_exc()
    
    return result


def batch_convert(base_dir, pattern="performance_finding_ckf.root", 
                 output_suffix="_graphs", keys="trackeff_vs_pT,trackeff_vs_eta",
                 workers=None, dry_run=False, verbose=False, skip_existing=True):
    """
    Batch convert ROOT files containing TEfficiency objects.
    
    Args:
        base_dir: Base directory to search for ROOT files
        pattern: Filename pattern to match
        output_suffix: Suffix to add before .root extension
        keys: Comma-separated list of efficiency object names to convert
        workers: Number of parallel workers (None = auto)
        dry_run: If True, only print what would be done
        verbose: Print detailed progress
        skip_existing: Skip files where output already exists
        
    Returns:
        dict: Summary statistics
    """
    print(f"Searching for ROOT files in: {base_dir}")
    print(f"Pattern: {pattern}")
    print("-" * 80)
    
    # Find all matching ROOT files
    root_files = find_root_files(base_dir, pattern)
    print(f"Found {len(root_files)} ROOT files matching pattern")
    
    if len(root_files) == 0:
        print("No files found to process.")
        return {'total': 0, 'success': 0, 'failed': 0, 'skipped': 0}
    
    # Determine the script path (should be in same directory as this script)
    script_dir = Path(__file__).parent
    converter_script = script_dir / "convert_eff_to_graphs.C"
    
    if not converter_script.exists():
        raise FileNotFoundError(f"Converter script not found: {converter_script}")
    
    print(f"Using converter script: {converter_script}")
    print(f"Converting efficiency objects: {keys}")
    print("-" * 80)
    
    # Prepare conversion tasks
    tasks = []
    for input_path in root_files:
        # Create output path: same directory, with suffix added
        output_name = input_path.stem + output_suffix + input_path.suffix
        output_path = input_path.parent / output_name
        
        tasks.append((input_path, output_path, keys, converter_script, verbose))
    
    if dry_run:
        print("\n=== DRY RUN MODE ===")
        print("Would process the following files:\n")
        for input_path, output_path, _, _, _ in tasks[:10]:
            print(f"  {input_path}")
            print(f"  -> {output_path}\n")
        if len(tasks) > 10:
            print(f"  ... and {len(tasks) - 10} more files")
        print(f"\nTotal files to process: {len(tasks)}")
        return {'total': len(tasks), 'success': 0, 'failed': 0, 'skipped': 0}
    
    # Determine number of workers
    if workers is None:
        workers = min(cpu_count(), 16)  # Cap at 16 to avoid overwhelming the system
    
    print(f"Processing {len(tasks)} files with {workers} workers...")
    print("-" * 80)
    
    # Process files in parallel
    results = []
    with Pool(processes=workers) as pool:
        # Use tqdm for progress bar
        for result in tqdm(pool.imap_unordered(convert_single_file, tasks), 
                          total=len(tasks), 
                          desc="Converting",
                          unit="file"):
            results.append(result)
    
    # Summarize results
    stats = {
        'total': len(results),
        'success': sum(1 for r in results if r['success']),
        'failed': sum(1 for r in results if not r['success'] and 'skipped' not in r['message'].lower()),
        'skipped': sum(1 for r in results if 'skipped' in r['message'].lower())
    }
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files:      {stats['total']}")
    print(f"Successful:       {stats['success']}")
    print(f"Failed:           {stats['failed']}")
    print(f"Skipped:          {stats['skipped']}")
    
    # Show failures if any
    failures = [r for r in results if not r['success'] and 'skipped' not in r['message'].lower()]
    if failures:
        print("\n" + "-" * 80)
        print("FAILURES:")
        print("-" * 80)
        for failure in failures[:20]:  # Show first 20 failures
            print(f"\n{failure['input']}")
            print(f"  Error: {failure['message']}")
        if len(failures) > 20:
            print(f"\n... and {len(failures) - 20} more failures")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert TEfficiency objects in ROOT files to TGraphAsymmErrors + TTrees",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all performance_finding_ckf.root files
  %(prog)s /path/to/dataset
  
  # Convert with custom pattern
  %(prog)s /path/to/dataset --pattern "performance_*.root"
  
  # Dry run to preview
  %(prog)s /path/to/dataset --dry-run
  
  # Custom efficiency keys
  %(prog)s /path/to/dataset --keys "trackeff_vs_pT,trackeff_vs_eta,trackeff_vs_phi"
  
  # Use 4 parallel workers
  %(prog)s /path/to/dataset --workers 4
        """
    )
    
    parser.add_argument(
        'base_dir',
        help='Base directory to search for ROOT files'
    )
    
    parser.add_argument(
        '--pattern',
        default='performance_finding_ckf.root',
        help='Filename pattern to match (default: performance_finding_ckf.root)'
    )
    
    parser.add_argument(
        '--output-suffix',
        default='_graphs',
        help='Suffix to add to output filename before .root (default: _graphs)'
    )
    
    parser.add_argument(
        '--keys',
        default='trackeff_vs_pT,trackeff_vs_eta',
        help='Comma-separated list of efficiency object keys to convert (default: trackeff_vs_pT,trackeff_vs_eta)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: auto-detect, max 16)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually converting'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )
    
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Reconvert files even if output already exists'
    )
    
    args = parser.parse_args()
    
    try:
        stats = batch_convert(
            base_dir=args.base_dir,
            pattern=args.pattern,
            output_suffix=args.output_suffix,
            keys=args.keys,
            workers=args.workers,
            dry_run=args.dry_run,
            verbose=args.verbose,
            skip_existing=not args.no_skip_existing
        )
        
        # Exit with error if there were failures
        if stats['failed'] > 0:
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
