import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import polars as pl
def load_all_particles_parquet(parquet_path, event_id=None):
    """# Plan: Polars-Based Calo Explode

1. **Inspect Loader Context**

- Read current `load_all_calohits_parquet` in `22_DM_parquet_processing.ipynb` to confirm column names and list nesting.
- Note Polars version (`pl.__version__`) to ensure required expr APIs (e.g., `with_row_index`, `map`) are available.

2. **Design Cell-Level Extraction**

- Build a lazy expression (`pl.scan_parquet` → filter by `event_id`) to avoid eager full-file reads.
- Select `event_id` + scalar cell columns, call `.explode(scalar_list_cols)` to flatten one row per cell, then `.with_row_index("cell_index")` so cells and contributions share an index.

3. **Design Contribution-Level Extraction**

- From the same filtered lazy frame, keep `event_id`, `cell_index`, and contribution list columns.
- Use `pl.map(contrib_cols, lambda cols: list(zip(*cols)))` (or `arr.zip` if available) to build a list of structs per row; `explode("contrib_structs")` + `unnest` yields one row per contribution with aligned particle/energy/time.
- Collect both lazy frames, convert to pandas only if callers demand pandas.

4. **Plumb Return Values & Tests**

- Update call sites to unpack `(cells_df, contrib_df)` and adjust summary prints.
- Re-run timing cell to ensure `calohits` stage now matches other loaders and row counts are reasonable.
    Load particles data from Parquet file using Polars for fast exploding.
    
    Args:
        parquet_path: Path to Parquet file
        event_id: Optional specific event ID to load. If None, loads all events.
    
    Returns:
        DataFrame with particles data (flat format with one row per particle)
    """
    # Read parquet file with Polars
    df = pl.read_parquet(parquet_path)
    
    if df.is_empty():
        return pd.DataFrame()
    
    if event_id is not None:
        # Filter to specific event
        df = df.filter(pl.col('event_id') == event_id)
        if df.is_empty():
            return pd.DataFrame()
    
    # Check if data needs exploding by examining first non-event_id column
    non_event_cols = [c for c in df.columns if c != 'event_id']
    if not non_event_cols:
        return df.to_pandas()
    
    # Check if the column dtype is a List type
    if df[non_event_cols[0]].dtype == pl.List:
        # Use Polars explode for fast unnesting
        df_exploded = df.explode(non_event_cols)
        return df_exploded.to_pandas()
    else:
        # Data is already flat
        return df.to_pandas()
def load_all_digihits_parquet(parquet_path, event_id=None):
    """
    Load digihits data from Parquet file using Polars.
    
    Args:
        parquet_path: Path to Parquet file
        event_id: Optional specific event ID to load. If None, loads all events.
    
    Returns:
        DataFrame with digihits data (flat format with one row per hit)
    """
    # Read parquet file with Polars
    df = pl.read_parquet(parquet_path)
    
    if df.is_empty():
        return pd.DataFrame()
    
    if event_id is not None:
        df = df.filter(pl.col('event_id') == event_id)
        if df.is_empty():
            return pd.DataFrame()
    
    # Check if data needs exploding
    non_event_cols = [c for c in df.columns if c != 'event_id']
    if not non_event_cols:
        return df.to_pandas()
    
    # Check if the column dtype is a List type
    if df[non_event_cols[0]].dtype == pl.List:
        # Use Polars explode for fast unnesting
        df_exploded = df.explode(non_event_cols)
        return df_exploded.to_pandas()
    else:
        # Data is already flat
        return df.to_pandas()
def load_all_tracks_parquet(parquet_path, event_id=None):
    """
    Load tracks data from Parquet file using Polars.
    
    Args:
        parquet_path: Path to Parquet file
        event_id: Optional specific event ID to load. If None, loads all events.
    
    Returns:
        Tuple of (tracks_df, hits_df) where hits_df contains track hit associations
    """
    # Read parquet file with Polars
    df = pl.read_parquet(parquet_path)
    
    if df.is_empty():
        return pd.DataFrame(), None
    
    if event_id is not None:
        df = df.filter(pl.col('event_id') == event_id)
        if df.is_empty():
            return pd.DataFrame(), None
    
    # Check if data needs exploding
    non_event_cols = [c for c in df.columns if c != 'event_id']
    if not non_event_cols:
        df_pandas = df.to_pandas()
        if 'hit_ids' in df_pandas.columns:
            hits_df = df_pandas[['event_id', 'track_id', 'hit_ids']].copy()
            tracks_df = df_pandas.drop(columns=['hit_ids'])
            return tracks_df, hits_df
        return df_pandas, None
    
    # Check if the column dtype is a List type
    if df[non_event_cols[0]].dtype != pl.List:
        # Data is already flat
        df_pandas = df.to_pandas()
        if 'hit_ids' in df_pandas.columns:
            hits_df = df_pandas[['event_id', 'track_id', 'hit_ids']].copy()
            tracks_df = df_pandas.drop(columns=['hit_ids'])
            return tracks_df, hits_df
        return df_pandas, None
    
    # Handle hit_ids separately (nested lists that should be preserved)
    has_hit_ids = 'hit_ids' in df.columns
    
    # Separate hit_ids from other columns
    track_cols = [c for c in non_event_cols if c != 'hit_ids']
    
    if track_cols:
        # Explode track data
        df_tracks = df.select(['event_id'] + track_cols).explode(track_cols)
        tracks_df = df_tracks.to_pandas()
    else:
        tracks_df = pd.DataFrame()
    
    # Handle hit_ids separately if present
    hits_df = None
    if has_hit_ids and 'track_id' in tracks_df.columns:
        # Explode hit_ids but keep inner lists
        df_hits = df.select(['event_id', 'hit_ids']).explode('hit_ids')
        hits_df = df_hits.to_pandas()
        # Add track_id from tracks_df
        hits_df['track_id'] = tracks_df['track_id'].values
    
    return tracks_df, hits_df

def load_all_calohits_parquet(parquet_path, event_id=None):
    """
    Load calorimeter hits data from Parquet file using Polars.
    
    Returns two DataFrames:
    1. Cell-level hits (one row per calorimeter cell) without contribution columns
    2. Contribution-level data (one row per particle contribution to a cell)
    """
    import polars as pl
    import pandas as pd
    
    # 1. Lazy Scan & Filter
    # Use lazy API to avoid reading full file if we filter
    lf = pl.scan_parquet(parquet_path)
    
    if event_id is not None:
        lf = lf.filter(pl.col('event_id') == event_id)
    
    # Inspect schema to separate cell vs. contribution columns
    # We use limit(0) to get schema without reading data if collect_schema isn't available
    try:
        schema = lf.collect_schema()
    except AttributeError:
        schema = lf.limit(0).collect().schema
        
    all_cols = schema.names()
    
    # Identify contribution columns (List(List))
    contrib_cols = [c for c in all_cols if c.startswith("contrib_")]
    
    # Identify scalar cell columns (List(Scalar))
    # These are all other columns except event_id
    cell_cols = [c for c in all_cols if c != "event_id" and c not in contrib_cols]
    
    if not cell_cols:
        return pd.DataFrame(), pd.DataFrame()
        
    # 2. Cell-Level Extraction
    # We explode ALL list columns (cells + contribs) simultaneously.
    # Polars guarantees alignment (lock-step explode) when a list of columns is passed,
    # preventing Cartesian products.
    # Result: 'cell_cols' become scalars. 'contrib_cols' become List[Scalar].
    
    lf_exploded = lf.explode(cell_cols + contrib_cols).with_row_index("cell_index")
    
    # Collect cells DataFrame
    cells_df = (
        lf_exploded
        .select(["event_id", "cell_index"] + cell_cols)
        .collect()
        .to_pandas()
    )
    
    # 3. Contribution-Level Extraction
    # From the same exploded frame (now one row per cell), we select contrib columns.
    # These are now simple lists (one list per cell).
    # We explode them again to get one row per contribution.
    
    if contrib_cols:
        # Prepare rename map to remove 'contrib_' prefix
        rename_exprs = [pl.col(c).alias(c.replace("contrib_", "")) for c in contrib_cols]
        
        contributions_df = (
            lf_exploded
            .select(["event_id", "cell_index"] + contrib_cols)
            .explode(contrib_cols)  # Explode inner lists in lock-step
            .select(["event_id", "cell_index"] + rename_exprs)
            .collect()
            .to_pandas()
        )
    else:
        contributions_df = pd.DataFrame()
        
    return cells_df, contributions_df
