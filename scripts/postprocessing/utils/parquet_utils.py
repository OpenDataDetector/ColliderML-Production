#!/usr/bin/env python3
"""
Parquet writing utilities for EDM4HEP to Parquet conversion.

This module provides reusable functions for writing particle physics data
to Parquet format with proper handling of nested/ragged arrays.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def optimize_dtypes_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame dtypes for Parquet storage.
    
    Converts to smaller types where possible to reduce file size:
    - float64 -> float32 (where appropriate)
    - int64 -> int32 (where values fit)
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with optimized dtypes
    """
    df = df.copy()
    
    for col in df.columns:
        if col == 'event_id':
            # Keep event_id as int64 for safety
            continue
            
        dtype = df[col].dtype
        
        # Downcast integers (except cell_id which needs int64)
        if pd.api.types.is_integer_dtype(dtype) and col != 'cell_id':
            try:
                # Check if all values fit in int32
                col_min = df[col].min() if len(df) > 0 else 0
                col_max = df[col].max() if len(df) > 0 else 0
                if col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype('int32')
            except Exception:
                pass
                
        # Downcast floats
        elif pd.api.types.is_float_dtype(dtype):
            try:
                df[col] = df[col].astype('float32')
            except Exception:
                pass
                
    return df


def group_by_event_to_lists(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group DataFrame by event_id and aggregate all other columns into lists.
    Uses PyArrow for efficient nested array handling.
    
    Args:
        df: Input DataFrame with event_id column and per-particle/hit data
        
    Returns:
        Grouped DataFrame with one row per event and list columns
    """
    if df.empty:
        logger.warning("group_by_event_to_lists received empty DataFrame")
        return df
    
    if 'event_id' not in df.columns:
        raise ValueError("DataFrame must have 'event_id' column for grouping")
    
    # Use PyArrow for efficient nested array handling
    table = pa.Table.from_pandas(df)
    
    # Group by event_id - creates proper Arrow list arrays
    grouped_table = table.group_by('event_id').aggregate([
        (col, 'list') for col in table.column_names if col != 'event_id'
    ])
    
    grouped = grouped_table.to_pandas()
    logger.debug(f"Grouped {len(df)} rows into {len(grouped)} events")
    
    return grouped


def write_parquet_table(
    df: pd.DataFrame,
    output_file: str,
    compression: str = 'snappy',
    optimize_dtypes: bool = True,
) -> None:
    """
    Write a DataFrame to Parquet format.
    
    Args:
        df: DataFrame to write (should already be grouped by event if needed)
        output_file: Path to output Parquet file
        compression: Compression codec ('snappy', 'gzip', 'zstd', 'none')
        optimize_dtypes: Whether to optimize dtypes before writing
    """
    if df.empty:
        logger.warning(f"Skipping write of empty DataFrame to {output_file}")
        return
    
    try:
        # Optimize dtypes if requested
        if optimize_dtypes:
            df = optimize_dtypes_for_parquet(df)
        
        # Convert to PyArrow table
        table = pa.Table.from_pandas(df)
        
        # Write to Parquet
        pq.write_table(
            table,
            output_file,
            compression=compression,
            use_dictionary=True,  # Enable dictionary encoding for repeated values
        )
        
        logger.debug(f"Wrote Parquet file: {output_file} ({len(df)} rows)")
        
    except Exception as e:
        logger.error(f"Failed to write Parquet file {output_file}: {e}")
        raise


def build_parquet_from_flat_df(
    df: pd.DataFrame,
    output_file: str,
    compression: str = 'snappy',
) -> None:
    """
    Build Parquet file from a flat DataFrame (with per-particle/hit rows).
    
    This is the main entry point for converting flat data to Parquet format:
    1. Groups by event_id into lists
    2. Optimizes dtypes
    3. Writes to Parquet
    
    Args:
        df: Flat DataFrame with event_id and per-particle/hit columns
        output_file: Path to output Parquet file
        compression: Compression codec to use
    """
    if df.empty:
        logger.warning(f"Skipping empty DataFrame for {output_file}")
        return
    
    # Group by event
    grouped = group_by_event_to_lists(df)
    
    # Write to Parquet
    write_parquet_table(grouped, output_file, compression=compression)



