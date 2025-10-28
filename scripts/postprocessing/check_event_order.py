#!/usr/bin/env python3
"""
Fast event counter and order checker for HepMC ASCII files.
Optimized for large files (GBs) by reading line-by-line and only processing event lines.
"""

import sys
import argparse
import time
from pathlib import Path


def check_events(filepath, verbose=False):
    """
    Count events and check if they're in order.
    
    Args:
        filepath: Path to the HepMC file
        verbose: If True, print all event numbers found
    
    Returns:
        tuple: (event_count, is_ordered, event_numbers)
    """
    event_numbers = []
    
    # Use buffered reading for speed
    with open(filepath, 'r', buffering=8192*16) as f:
        for line in f:
            # Only process lines that start with "E "
            if line.startswith('E '):
                # Extract event number (second field after "E ")
                parts = line.split(None, 3)  # split on whitespace, max 3 splits
                if len(parts) >= 2:
                    try:
                        event_num = int(parts[1])
                        event_numbers.append(event_num)
                    except ValueError:
                        print(f"Warning: Could not parse event number from line: {line.strip()}")
    
    # Check ordering
    event_count = len(event_numbers)
    is_ordered = True
    first_out_of_order = None
    
    if event_count > 1:
        for i in range(1, event_count):
            if event_numbers[i] <= event_numbers[i-1]:
                is_ordered = False
                first_out_of_order = i
                break
    
    return event_count, is_ordered, event_numbers, first_out_of_order


def main():
    parser = argparse.ArgumentParser(
        description='Check event count and ordering in HepMC ASCII files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s events.hepmc
  %(prog)s events.hepmc --verbose
  %(prog)s events.hepmc --list
        """
    )
    parser.add_argument('file', type=str, help='Path to HepMC file')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print progress messages')
    parser.add_argument('-l', '--list', action='store_true',
                        help='List all event numbers found')
    
    args = parser.parse_args()
    
    filepath = Path(args.file)
    
    if not filepath.exists():
        print(f"Error: File '{filepath}' not found", file=sys.stderr)
        sys.exit(1)
    
    if args.verbose:
        print(f"Checking file: {filepath}")
        print(f"File size: {filepath.stat().st_size / (1024**3):.2f} GB")
        print("Processing...")
    
    # Check events with timing
    start_time = time.time()
    event_count, is_ordered, event_numbers, first_out_of_order = check_events(filepath, args.verbose)
    elapsed_time = time.time() - start_time
    
    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Processing time: {elapsed_time:.3f} seconds")
    print(f"Total events found: {event_count}")
    
    if event_count > 0:
        print(f"First event number: {event_numbers[0]}")
        print(f"Last event number:  {event_numbers[-1]}")
        print(f"Expected range:     {event_numbers[0]} to {event_numbers[0] + event_count - 1}")
        
        # Check if ordered
        if is_ordered:
            print(f"\n✓ Events are IN ORDER")
            
            # Check if they're sequential (0, 1, 2, ... or 1, 2, 3, ...)
            expected_sequence = list(range(event_numbers[0], event_numbers[0] + event_count))
            if event_numbers == expected_sequence:
                print(f"✓ Events are SEQUENTIAL ({event_numbers[0]} to {event_numbers[-1]})")
            else:
                print(f"⚠ Events are ordered but NOT sequential")
                # Find gaps
                missing = set(expected_sequence) - set(event_numbers)
                if missing and len(missing) <= 20:
                    print(f"  Missing event numbers: {sorted(missing)}")
                elif missing:
                    print(f"  {len(missing)} event numbers are missing from the sequence")
        else:
            print(f"\n✗ Events are OUT OF ORDER")
            print(f"  First out-of-order event at position {first_out_of_order}:")
            print(f"    Event {first_out_of_order-1}: {event_numbers[first_out_of_order-1]}")
            print(f"    Event {first_out_of_order}: {event_numbers[first_out_of_order]}")
    else:
        print("No events found in file!")
    
    # List all events if requested
    if args.list:
        print(f"\n{'='*60}")
        print("ALL EVENT NUMBERS:")
        print(f"{'='*60}")
        for i, num in enumerate(event_numbers):
            print(f"Event {i}: {num}")
    
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
