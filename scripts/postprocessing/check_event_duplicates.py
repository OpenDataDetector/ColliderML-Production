#!/usr/bin/env python3
"""
Check for duplicate events in HepMC3 files based on vertex and particle counts.
If the random seed is working properly, duplicate (vertex_count, particle_count) 
pairs should be extremely rare.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def parse_events_from_file(filepath):
    """
    Parse HepMC3 file and extract event information.
    Returns a list of tuples: (event_num, n_vertices, n_particles)
    """
    events = []
    
    with open(filepath, 'rb') as f:
        for line in f:
            if line.startswith(b'E '):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        event_num = int(parts[1])
                        n_vertices = int(parts[2])
                        n_particles = int(parts[3]) if len(parts) > 3 else 0
                        events.append((event_num, n_vertices, n_particles))
                    except (ValueError, IndexError):
                        print(f"Warning: Could not parse line: {line.decode('utf-8', errors='ignore').strip()}")
    
    return events


def check_duplicates_single_file(filepath, verbose=False):
    """
    Check for duplicate events in a single file.
    Returns a dict with analysis results.
    """
    if verbose:
        print(f"\nAnalyzing: {filepath}")
    
    events = parse_events_from_file(filepath)
    
    if not events:
        return {
            'filepath': filepath,
            'status': 'error',
            'message': 'No events found'
        }
    
    # Create DataFrame
    df = pd.DataFrame(events, columns=['event_num', 'n_vertices', 'n_particles'])
    
    # Find duplicates based on (n_vertices, n_particles) pairs
    df['signature'] = list(zip(df['n_vertices'], df['n_particles']))
    duplicates = df[df.duplicated(subset=['signature'], keep=False)]
    
    result = {
        'filepath': str(filepath),
        'status': 'success',
        'total_events': len(df),
        'unique_signatures': df['signature'].nunique(),
        'duplicate_count': len(duplicates),
        'duplicate_signatures': duplicates['signature'].nunique() if len(duplicates) > 0 else 0,
    }
    
    if len(duplicates) > 0:
        # Group duplicates by signature, include file path
        dup_groups = {}
        for signature, group in duplicates.groupby('signature'):
            dup_groups[signature] = {
                'filepath': str(filepath),
                'event_nums': group['event_num'].tolist()
            }
        result['duplicate_groups'] = dup_groups
    
    if verbose:
        print(f"  Total events: {result['total_events']}")
        print(f"  Unique signatures: {result['unique_signatures']}")
        print(f"  Duplicates found: {result['duplicate_count']}")
    
    return result


def check_duplicates_multiple_files(filepaths, verbose=False):
    """
    Check for duplicates across multiple files.
    This checks if the same (n_vertices, n_particles) signature appears across files.
    """
    print(f"\nAnalyzing {len(filepaths)} files for cross-file duplicates...")
    
    all_events = []
    
    for filepath in tqdm(filepaths, desc="Parsing files", unit="file"):
        events = parse_events_from_file(filepath)
        for event_num, n_vertices, n_particles in events:
            all_events.append({
                'filepath': str(filepath),
                'event_num': event_num,
                'n_vertices': n_vertices,
                'n_particles': n_particles
            })
    
    df = pd.DataFrame(all_events)
    df['signature'] = list(zip(df['n_vertices'], df['n_particles']))
    
    # Find duplicates
    duplicates = df[df.duplicated(subset=['signature'], keep=False)]
    
    result = {
        'status': 'success',
        'total_events': len(df),
        'total_files': len(filepaths),
        'unique_signatures': df['signature'].nunique(),
        'duplicate_count': len(duplicates),
        'duplicate_signatures': duplicates['signature'].nunique() if len(duplicates) > 0 else 0,
    }
    
    if len(duplicates) > 0:
        # Group by signature and show full file paths with event numbers
        dup_groups = {}
        for signature, group in duplicates.groupby('signature'):
            occurrences = []
            for _, row in group.iterrows():
                occurrences.append({
                    'filepath': row['filepath'],
                    'event_num': row['event_num']
                })
            dup_groups[signature] = occurrences
        result['duplicate_groups'] = dup_groups
    
    print(f"\nTotal events across all files: {result['total_events']}")
    print(f"Unique signatures: {result['unique_signatures']}")
    print(f"Events with duplicate signatures: {result['duplicate_count']}")
    
    return result, df


def main():
    parser = argparse.ArgumentParser(
        description='Check for duplicate events in HepMC3 files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check a single file
  %(prog)s /path/to/merged_events.hepmc3
  
  # Check multiple files
  %(prog)s /path/to/runs/0/merged_events.hepmc3 /path/to/runs/1/merged_events.hepmc3
  
  # Check all files in run directories using glob
  %(prog)s /path/to/runs/*/merged_events.hepmc3
  
  # Check all files in a parent directory (searches recursively)
  %(prog)s --directory /path/to/runs
  
  # Check all files with custom pattern
  %(prog)s --directory /path/to/runs --pattern "merged_events_original.hepmc3"
  
  # Save results to CSV
  %(prog)s --directory /path/to/runs --save-csv results.csv
        """
    )
    parser.add_argument('files', nargs='*', type=str,
                        help='HepMC3 file(s) to check (can use wildcards)')
    parser.add_argument('-d', '--directory', type=str, default=None,
                        help='Parent directory to search for HepMC3 files')
    parser.add_argument('-p', '--pattern', type=str, default='merged_events.hepmc3',
                        help='Filename pattern to search for (default: merged_events.hepmc3)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--save-csv', type=str, default=None,
                        help='Save event data to CSV file')
    parser.add_argument('--cross-file', action='store_true',
                        help='Check for duplicates across multiple files (default if multiple files given)')
    
    args = parser.parse_args()
    
    # Expand globs and convert to Path objects
    filepaths = []
    
    # If directory is specified, search recursively for pattern
    if args.directory:
        parent_dir = Path(args.directory)
        if not parent_dir.exists():
            print(f"Error: Directory '{parent_dir}' not found", file=sys.stderr)
            return 1
        
        if not parent_dir.is_dir():
            print(f"Error: '{parent_dir}' is not a directory", file=sys.stderr)
            return 1
        
        print(f"Searching for '{args.pattern}' in {parent_dir}...")
        # Use rglob for recursive search
        filepaths = list(parent_dir.rglob(args.pattern))
        
        if not filepaths:
            print(f"Error: No files matching '{args.pattern}' found in '{parent_dir}'", file=sys.stderr)
            return 1
        
        # Sort by path for consistent ordering
        filepaths.sort()
        print(f"Found {len(filepaths)} matching file(s)")
    
    # Otherwise, process explicitly provided files
    elif args.files:
        for pattern in args.files:
            path = Path(pattern)
            if path.exists():
                filepaths.append(path)
            else:
                # Try glob expansion
                from glob import glob
                matches = glob(pattern)
                filepaths.extend([Path(m) for m in matches])
        
        if not filepaths:
            print("Error: No valid files found", file=sys.stderr)
            return 1
        
        filepaths = [f for f in filepaths if f.exists()]
        
        if not filepaths:
            print("Error: No existing files found", file=sys.stderr)
            return 1
        
        print(f"Found {len(filepaths)} file(s) to analyze")
    
    else:
        print("Error: Must provide either files or --directory", file=sys.stderr)
        parser.print_help()
        return 1
    
    # Single file analysis
    if len(filepaths) == 1:
        result = check_duplicates_single_file(filepaths[0], verbose=args.verbose)
        
        if result['status'] == 'error':
            print(f"Error: {result['message']}")
            return 1
        
        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print(f"File: {result['filepath']}")
        print(f"Total events: {result['total_events']}")
        print(f"Unique (vertices, particles) signatures: {result['unique_signatures']}")
        print(f"Duplicate events found: {result['duplicate_count']}")
        
        if result['duplicate_count'] > 0:
            print(f"\n⚠️  WARNING: Found {result['duplicate_count']} events with duplicate signatures!")
            print(f"Number of unique duplicate signatures: {result['duplicate_signatures']}")
            print("\nDuplicate signatures and their event numbers:")
            for signature, dup_info in result['duplicate_groups'].items():
                print(f"  {signature}:")
                print(f"    File: {dup_info['filepath']}")
                print(f"    Events: {dup_info['event_nums']}")
        else:
            print("\n✓ No duplicates found - random seed appears to be working correctly!")
        
        print(f"{'='*70}\n")
    
    # Multi-file analysis
    else:
        result, df = check_duplicates_multiple_files(filepaths, verbose=args.verbose)
        
        print(f"\n{'='*70}")
        print("CROSS-FILE DUPLICATE CHECK RESULTS")
        print(f"{'='*70}")
        print(f"Files analyzed: {result['total_files']}")
        print(f"Total events: {result['total_events']}")
        print(f"Unique (vertices, particles) signatures: {result['unique_signatures']}")
        print(f"Events with duplicate signatures: {result['duplicate_count']}")
        
        if result['duplicate_count'] > 0:
            print(f"\n⚠️  WARNING: Found {result['duplicate_count']} events with duplicate signatures!")
            print(f"Number of unique duplicate signatures: {result['duplicate_signatures']}")
            print("\nDuplicate signatures (showing first 10):")
            count = 0
            for signature, occurrences in result['duplicate_groups'].items():
                if count >= 10:
                    print(f"  ... and {len(result['duplicate_groups']) - 10} more duplicate signatures")
                    break
                print(f"\n  Signature {signature} ({len(occurrences)} occurrences):")
                for i, occ in enumerate(occurrences[:10]):  # Show first 10 occurrences
                    print(f"    [{i+1}] {occ['filepath']} - event {occ['event_num']}")
                if len(occurrences) > 10:
                    print(f"    ... and {len(occurrences) - 10} more occurrences")
                count += 1
        else:
            print("\n✓ No duplicates found across files - random seed appears to be working correctly!")
        
        print(f"{'='*70}\n")
        
        # Save to CSV if requested
        if args.save_csv:
            df.to_csv(args.save_csv, index=False)
            print(f"Event data saved to: {args.save_csv}")
    
    return 0


if __name__ == '__main__':
    exit(main())
