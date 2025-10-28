#!/usr/bin/env python3
"""
Batch process HepMC files: rename, split, and reorganize across run directories.
Optimized for parallel processing with multiprocessing.
"""

import sys
import argparse
import time
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
import traceback
from tqdm import tqdm


def find_event_offsets_fast(filepath):
    """Find byte offsets of all event starts."""
    event_offsets = []
    header_end = 0
    in_header = True
    
    with open(filepath, 'rb') as f:
        while True:
            line_start = f.tell()
            line = f.readline()
            
            if not line:
                break
                
            if line.startswith(b'E '):
                if in_header:
                    header_end = line_start
                    in_header = False
                
                parts = line.split(None, 3)
                if len(parts) >= 2:
                    try:
                        event_num = int(parts[1])
                        event_offsets.append((event_num, line_start))
                    except ValueError:
                        pass
    
    return event_offsets, header_end


def split_and_copy_events(input_file, output_file, event_offsets, header, 
                          start_idx, end_idx, buffer_size=8*1024*1024):
    """
    Copy a range of events to an output file, renumbering events starting from 0.
    """
    start_offset = event_offsets[start_idx][1]
    
    # Determine end offset
    with open(input_file, 'rb') as f_in:
        if end_idx < len(event_offsets):
            end_offset = event_offsets[end_idx][1]
        else:
            f_in.seek(0, 2)
            end_offset = f_in.tell()
        
        # Write output file with renumbered events
        with open(output_file, 'wb') as f_out:
            f_out.write(header)
            
            f_in.seek(start_offset)
            new_event_num = 0
            
            # Read and process line by line to renumber events
            current_pos = start_offset
            while current_pos < end_offset:
                line_start = f_in.tell()
                line = f_in.readline()
                
                if not line or f_in.tell() > end_offset:
                    break
                
                # Check if this is an event line that needs renumbering
                if line.startswith(b'E '):
                    # Parse and renumber: "E oldnum ..." -> "E newnum ..."
                    parts = line.split(None, 3)  # Split into ["E", "oldnum", "vertices", "particles", ...]
                    if len(parts) >= 3:
                        # Reconstruct with new event number
                        new_line = b'E ' + str(new_event_num).encode() + b' ' + b' '.join(parts[2:]) + b'\n'
                        f_out.write(new_line)
                        new_event_num += 1
                    else:
                        f_out.write(line)
                else:
                    f_out.write(line)
                
                current_pos = f_in.tell()
    
    return end_offset - start_offset


def process_single_run(args):
    """Process a single run directory. Designed to be called in parallel."""
    run_dir, offset_n, events_per_split, verbose = args
    
    try:
        run_num = int(run_dir.name)
        input_file = run_dir / "merged_events.hepmc3"
        renamed_file = run_dir / "merged_events_original.hepmc3"
        
        if not input_file.exists():
            return {
                'run': run_num,
                'status': 'skipped',
                'reason': 'merged_events.hepmc3 not found'
            }
        
        start_time = time.time()
        
        # Step 1: Rename original file
        if renamed_file.exists():
            if verbose:
                print(f"Run {run_num}: merged_events_original.hepmc3 already exists, skipping rename")
        else:
            shutil.move(str(input_file), str(renamed_file))
        
        # Step 2: Find event offsets
        event_offsets, header_end = find_event_offsets_fast(renamed_file)
        total_events = len(event_offsets)
        
        if total_events == 0:
            return {
                'run': run_num,
                'status': 'error',
                'reason': 'No events found'
            }
        
        # Read header
        with open(renamed_file, 'rb') as f:
            f.seek(0)
            header = f.read(header_end)
        
        # Step 3: Split file
        num_splits = (total_events + events_per_split - 1) // events_per_split
        
        if num_splits < 2:
            return {
                'run': run_num,
                'status': 'error',
                'reason': f'Not enough events ({total_events}) to split into {events_per_split} per file'
            }
        
        # First split goes to original directory
        first_output = run_dir / "merged_events.hepmc3"
        split_and_copy_events(
            renamed_file, first_output, event_offsets, header,
            0, events_per_split
        )
        
        # Second split goes to new directory (run_num + offset_n)
        new_run_num = run_num + offset_n
        new_run_dir = run_dir.parent / str(new_run_num)
        new_run_dir.mkdir(parents=True, exist_ok=True)
        
        second_output = new_run_dir / "merged_events.hepmc3"
        split_and_copy_events(
            renamed_file, second_output, event_offsets, header,
            events_per_split, min(2 * events_per_split, total_events)
        )
        
        elapsed = time.time() - start_time
        
        return {
            'run': run_num,
            'status': 'success',
            'total_events': total_events,
            'new_run': new_run_num,
            'time': elapsed,
            'first_events': f"{event_offsets[0][0]}-{event_offsets[min(events_per_split-1, total_events-1)][0]}",
            'second_events': f"{event_offsets[events_per_split][0]}-{event_offsets[min(2*events_per_split-1, total_events-1)][0]}" if events_per_split < total_events else "N/A"
        }
        
    except Exception as e:
        return {
            'run': run_num,
            'status': 'error',
            'reason': f'{type(e).__name__}: {str(e)}',
            'traceback': traceback.format_exc()
        }


def main():
    parser = argparse.ArgumentParser(
        description='Batch rename and split HepMC files across run directories',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # Process runs 0-999, offset new runs by 10000, split into 64 events each
  %(prog)s /path/to/runs -N 10000 -n 64 -j 32
  
  # Dry run to see what would happen
  %(prog)s /path/to/runs -N 10000 -n 64 --dry-run
        """
    )
    parser.add_argument('runs_dir', type=str,
                        help='Directory containing numbered run subdirectories')
    parser.add_argument('-N', '--offset', type=int, required=True,
                        help='Offset for new run directories (second split goes to run_num + N)')
    parser.add_argument('-n', '--events-per-split', type=int, default=64,
                        help='Number of events per split (default: 64)')
    parser.add_argument('-j', '--jobs', type=int, default=4,
                        help='Number of parallel jobs (default: 4, recommended 2-8 for I/O bound tasks)')
    parser.add_argument('--min-run', type=int, default=0,
                        help='Minimum run number to process')
    parser.add_argument('--max-run', type=int, default=None,
                        help='Maximum run number to process')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without actually doing it')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    runs_dir = Path(args.runs_dir)
    
    if not runs_dir.exists():
        print(f"Error: Directory '{runs_dir}' not found", file=sys.stderr)
        sys.exit(1)
    
    # Find all run directories
    print(f"Scanning for run directories in {runs_dir}...")
    run_dirs = []
    for d in runs_dir.iterdir():
        if d.is_dir() and d.name.isdigit():
            run_num = int(d.name)
            if run_num >= args.min_run:
                if args.max_run is None or run_num <= args.max_run:
                    run_dirs.append(d)
    
    run_dirs.sort(key=lambda x: int(x.name))
    
    print(f"Found {len(run_dirs)} run directories to process")
    print(f"Run range: {run_dirs[0].name} to {run_dirs[-1].name}")
    print(f"Offset: +{args.offset} (new runs will be {int(run_dirs[0].name) + args.offset} to {int(run_dirs[-1].name) + args.offset})")
    print(f"Events per split: {args.events_per_split}")
    
    if args.dry_run:
        print("\nDRY RUN - No changes will be made")
        print(f"\nWould process {len(run_dirs)} runs:")
        for rd in run_dirs[:5]:
            print(f"  {rd.name} -> splits to {rd.name} and {int(rd.name) + args.offset}")
        if len(run_dirs) > 5:
            print(f"  ... and {len(run_dirs) - 5} more")
        return
    
    # Determine number of parallel jobs
    n_jobs = args.jobs
    if n_jobs > 8:
        print(f"WARNING: Using {n_jobs} jobs may cause I/O contention and slow down processing.")
        print(f"         Recommended: 2-8 jobs for I/O-bound operations on shared filesystems.")
    print(f"Using {n_jobs} parallel jobs")
    
    # Prepare arguments for parallel processing
    process_args = [
        (rd, args.offset, args.events_per_split, args.verbose)
        for rd in run_dirs
    ]
    
    print(f"\n{'='*70}")
    print("Starting processing...")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    # Process in parallel with tqdm progress bar
    with Pool(n_jobs) as pool:
        results = []
        with tqdm(total=len(run_dirs), desc="Processing runs", unit="run") as pbar:
            for result in pool.imap_unordered(process_single_run, process_args):
                results.append(result)
                
                # Update progress bar with status
                status_symbol = "✓" if result['status'] == 'success' else "✗" if result['status'] == 'error' else "⊘"
                pbar.set_postfix_str(f"Run {result['run']}: {status_symbol}")
                pbar.update(1)
    
    total_time = time.time() - start_time
    
    # Summarize results
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = sum(1 for r in results if r['status'] == 'error')
    skipped_count = sum(1 for r in results if r['status'] == 'skipped')
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per run: {total_time/len(results):.3f} seconds")
    print(f"Processing rate: {len(results)/total_time:.2f} runs/second")
    print(f"\nSuccessful: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Skipped: {skipped_count}")
    
    if error_count > 0:
        print(f"\nErrors encountered:")
        for r in results:
            if r['status'] == 'error':
                print(f"  Run {r['run']}: {r['reason']}")
    
    if args.verbose and success_count > 0:
        print(f"\nDetailed results (first 10 successful):")
        count = 0
        for r in results:
            if r['status'] == 'success':
                print(f"  Run {r['run']} -> {r['new_run']}: "
                      f"{r['total_events']} events split in {r['time']:.3f}s")
                count += 1
                if count >= 10:
                    break
    
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
