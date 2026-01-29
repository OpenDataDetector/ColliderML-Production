#!/usr/bin/env python3
"""
Quick test to diagnose parquet loading issues.
"""

from pathlib import Path
from datasets import load_dataset
import traceback

# Test loading ttbar particles
data_path = Path("/global/cfs/cdirs/m4958/data/ColliderML/simulation/hard_scatter/ttbar/v1/parquet/truth/particles")

print(f"Testing parquet load from: {data_path}")
print(f"Number of parquet files: {len(list(data_path.glob('*.parquet')))}")

try:
    print("\nAttempting to load dataset...")
    dataset = load_dataset(
        "parquet",
        data_files=str(data_path / "*.parquet"),
        split="train"
    )
    print(f"✓ Success! Loaded {len(dataset)} events")
    print(f"Dataset schema: {dataset.features}")

except Exception as e:
    print(f"✗ Failed to load dataset")
    print(f"Error: {e}")
    print(f"Exception type: {type(e).__name__}")
    print(f"\nFull traceback:")
    traceback.print_exc()
