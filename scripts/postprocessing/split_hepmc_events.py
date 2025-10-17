#!/usr/bin/env python3
"""
BLAZINGLY FAST HepMC event splitter.
Strategy: Record byte offsets of events, then use direct file seeks and block copies.
Optimized for multi-GB files.
"""

import sys
import argparse
import time
from pathlib import Path


def find_event_offsets(filepath, verbose=False):
    """
    Find byte offsets of all event starts.
    Returns: (list of (event_num, byte_offset), header_end_offset)
    """
    event_offsets = []
    header_end = 0
    in_header = True
    
    with open(filepath, 'rb') as f:  # Binary mode for byte offsets
        offset = 0
        
        # Read header (before first event)
        while True:
            line_start = f.tell()
            line = f.readline()
            
            if not line:
                break
                
            # Check if this is an event line
            if line.startswith(b'E '):
                if in_header:
                    header_end = line_start
                    in_header = False
                
                # Parse event number
                parts = line.split(None, 3)
                if len(parts) >= 2:
                    try:
                        event_num = int(parts[1])
                        event_offsets.append((event_num, line_start))
                    except ValueError:
                        pass
    
    if verbose:
        print(f"Found {len(event_offsets)} events")
        print(f"Header ends at byte: {header_end}")
    
    return event_offsets, header_end


def split_file_fast(filepath, events_per_file, output_dir=None, output_prefix="split", verbose=False):
    """
    Split HepMC file into chunks with N events each.
    Uses byte offsets and block copying for maximum speed.
    """
    start_time = time.time()
    
    # Find all event offsets
    if verbose:
        print("Phase 1: Finding event positions...")
    
    event_offsets, header_end = find_event_offsets(filepath, verbose)
    
    if not event_offsets:
        print("Error: No events found in file!")
        return
    
    # Check if events are in order
    event_numbers = [e[0] for e in event_offsets]
    is_ordered = all(event_numbers[i] < event_numbers[i+1] for i in range(len(event_numbers)-1))
    
    if not is_ordered:
        print("Warning: Events are NOT in order! Proceeding anyway...")
    
    total_events = len(event_offsets)
    num_files = (total_events + events_per_file - 1) // events_per_file
    
    if verbose:
        print(f"\nPhase 2: Splitting into {num_files} files...")
        print(f"Events per file: {events_per_file}")
        print(f"Total events: {total_events}")
    
    # Setup output directory
    if output_dir is None:
        output_dir = filepath.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read header once
    with open(filepath, 'rb') as f:
        f.seek(0)
        header = f.read(header_end)
    
    # Split into files using block copying
    BUFFER_SIZE = 1024 * 1024 * 8  # 8MB buffer for fast copying
    
    with open(filepath, 'rb') as f_in:
        for file_idx in range(num_files):
            start_event_idx = file_idx * events_per_file
            end_event_idx = min(start_event_idx + events_per_file, total_events)
            
            # Determine byte range for this chunk
            start_offset = event_offsets[start_event_idx][1]
            
            # End offset is either the start of the next chunk or end of file
            if end_event_idx < total_events:
                end_offset = event_offsets[end_event_idx][1]
            else:
                # Go to end of file
                f_in.seek(0, 2)  # Seek to end
                end_offset = f_in.tell()
            
            bytes_to_copy = end_offset - start_offset
            
            # Create output file
            output_path = output_dir / f"{output_prefix}_{file_idx:04d}.hepmc"
            
            with open(output_path, 'wb') as f_out:
                # Write header
                f_out.write(header)
                
                # Copy event data in blocks
                f_in.seek(start_offset)
                remaining = bytes_to_copy
                
                while remaining > 0:
                    chunk_size = min(BUFFER_SIZE, remaining)
                    chunk = f_in.read(chunk_size)
                    f_out.write(chunk)
                    remaining -= len(chunk)
            
            if verbose:
                event_range = f"{event_offsets[start_event_idx][0]}-{event_offsets[end_event_idx-1][0]}"
                print(f"  Created: {output_path.name} (events {event_range}, {bytes_to_copy/(1024**2):.2f} MB)")
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"SPLIT COMPLETE")
    print(f"{'='*60}")
    print(f"Processing time: {elapsed:.3f} seconds")
    print(f"Total events: {total_events}")
    print(f"Files created: {num_files}")
    print(f"Events per file: {events_per_file} (last file may have fewer)")
    print(f"Output directory: {output_dir}")
    print(f"Speed: {filepath.stat().st_size / (1024**3) / elapsed:.2f} GB/s")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='BLAZINGLY FAST HepMC event splitter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split into files with 64 events each
  %(prog)s events.hepmc -n 64
  
  # Split with custom output prefix and directory
  %(prog)s events.hepmc -n 100 -o output_dir/ -p chunk
  
  # Verbose output
  %(prog)s events.hepmc -n 64 --verbose
        """
    )
    parser.add_argument('file', type=str, help='Path to HepMC file')
    parser.add_argument('-n', '--events-per-file', type=int, required=True,
                        help='Number of events per output file')
    parser.add_argument('-o', '--output-dir', type=str, default=None,
                        help='Output directory (default: same as input file)')
    parser.add_argument('-p', '--prefix', type=str, default='split',
                        help='Output file prefix (default: "split")')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print detailed progress')
    
    args = parser.parse_args()
    
    filepath = Path(args.file)
    
    if not filepath.exists():
        print(f"Error: File '{filepath}' not found", file=sys.stderr)
        sys.exit(1)
    
    if args.events_per_file <= 0:
        print(f"Error: events-per-file must be positive", file=sys.stderr)
        sys.exit(1)
    
    split_file_fast(
        filepath,
        args.events_per_file,
        output_dir=args.output_dir,
        output_prefix=args.prefix,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
