"""
Base classes and utilities for ColliderML data consistency tests.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import uproot
import pyhepmc as hep
import logging
import time

# Set up logging
logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Status of a test result."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    status: TestStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    
    def __str__(self):
        status_emoji = {
            TestStatus.PASSED: "✅",
            TestStatus.FAILED: "❌",
            TestStatus.SKIPPED: "⏭️",
            TestStatus.ERROR: "💥",
        }
        return f"{status_emoji[self.status]} {self.name}: {self.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "duration_ms": self.duration_ms,
        }


class DataLoader:
    """
    Utility class for loading data from various sources.
    Caches loaded data to avoid repeated I/O.
    """
    
    def __init__(self, base_path: str, run_id: int, run_size: int = 128, chunk_size: int = 100):
        """
        Initialize the data loader.
        
        Args:
            base_path: Base path to the ColliderML data directory
            run_id: Run ID to load data from
            run_size: Number of events per run (default 128)
            chunk_size: Number of events per parquet chunk (default 100)
        """
        self.base_path = Path(base_path)
        self.run_id = run_id
        self.run_size = run_size
        self.chunk_size = chunk_size
        self._cache: Dict[str, Any] = {}
        
        # Import polars here for fast loading
        try:
            import polars as pl
            self.pl = pl
        except ImportError:
            logger.warning("Polars not available, falling back to pandas")
            self.pl = None
    
    def _get_parquet_path(self, object_type: str, global_event_id: int) -> Path:
        """Get the parquet file path for a given object type and event."""
        chunk_number = global_event_id // self.chunk_size
        event_range = f"events{chunk_number*self.chunk_size}-{chunk_number*self.chunk_size+self.chunk_size-1}"
        
        object_paths = {
            "particles": f"parquet/truth/particles/full_pileup.ttbar.v1.truth.particles.{event_range}.parquet",
            "tracker_hits": f"parquet/reco/tracker_hits/full_pileup.ttbar.v1.reco.tracker_hits.{event_range}.parquet",
            "tracks": f"parquet/reco/tracks/full_pileup.ttbar.v1.reco.tracks.{event_range}.parquet",
            "calo_hits": f"parquet/reco/calo_hits/full_pileup.ttbar.v1.reco.calo_hits.{event_range}.parquet",
        }
        
        return self.base_path / object_paths[object_type]
    
    def _get_edm4hep_path(self) -> Path:
        """Get the EDM4hep file path for the current run."""
        return self.base_path / f"runs/{self.run_id}/edm4hep.root"
    
    def _get_hepmc_hs_path(self) -> Path:
        """Get the HepMC hard scatter file path."""
        return self.base_path / f"runs/{self.run_id}/events.hepmc"
    
    def _get_hepmc_merged_path(self) -> Path:
        """Get the merged HepMC file path."""
        return self.base_path / f"runs/{self.run_id}/merged_events.hepmc3"
    
    def _get_particles_root_path(self) -> Path:
        """Get the ACTS particles.root file path."""
        return self.base_path / f"runs/{self.run_id}/particles.root"
    
    def _get_measurements_root_path(self) -> Path:
        """Get the ACTS measurements.root file path."""
        return self.base_path / f"runs/{self.run_id}/measurements.root"
    
    def _get_tracks_root_path(self) -> Path:
        """Get the ACTS tracksummary_ambi.root file path."""
        return self.base_path / f"runs/{self.run_id}/tracksummary_ambi.root"
    
    def load_parquet_particles(self, global_event_id: Optional[int] = None) -> pd.DataFrame:
        """Load particles from parquet file.
        
        Args:
            global_event_id: The GLOBAL event ID (not local). In parquet files,
                event_id column contains global event IDs.
        """
        cache_key = f"parquet_particles_{global_event_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if global_event_id is not None:
            path = self._get_parquet_path("particles", global_event_id)
        else:
            # Load all events in the chunk
            path = self._get_parquet_path("particles", self.run_id * self.run_size)
        
        # NOTE: parquet event_id is the GLOBAL event ID, not local index
        df = self._load_parquet_file(path, event_id=global_event_id)
        self._cache[cache_key] = df
        return df
    
    def load_parquet_tracker_hits(self, global_event_id: Optional[int] = None) -> pd.DataFrame:
        """Load tracker hits from parquet file.
        
        Args:
            global_event_id: The GLOBAL event ID (not local). In parquet files,
                event_id column contains global event IDs.
        """
        cache_key = f"parquet_tracker_hits_{global_event_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if global_event_id is not None:
            path = self._get_parquet_path("tracker_hits", global_event_id)
        else:
            path = self._get_parquet_path("tracker_hits", self.run_id * self.run_size)
        
        # NOTE: parquet event_id is the GLOBAL event ID, not local index
        df = self._load_parquet_file(path, event_id=global_event_id)
        self._cache[cache_key] = df
        return df
    
    def load_parquet_tracks(self, global_event_id: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load tracks from parquet file. Returns (tracks_df, track_hits_df).
        
        Args:
            global_event_id: The GLOBAL event ID (not local). In parquet files,
                event_id column contains global event IDs.
        """
        cache_key = f"parquet_tracks_{global_event_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if global_event_id is not None:
            path = self._get_parquet_path("tracks", global_event_id)
        else:
            path = self._get_parquet_path("tracks", self.run_id * self.run_size)
        
        # NOTE: parquet event_id is the GLOBAL event ID, not local index
        result = self._load_parquet_tracks(path, event_id=global_event_id)
        self._cache[cache_key] = result
        return result
    
    def load_parquet_calo_hits(self, global_event_id: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load calo hits from parquet file. Returns (cells_df, contributions_df).
        
        Args:
            global_event_id: The GLOBAL event ID (not local). In parquet files,
                event_id column contains global event IDs.
        """
        cache_key = f"parquet_calo_hits_{global_event_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if global_event_id is not None:
            path = self._get_parquet_path("calo_hits", global_event_id)
        else:
            path = self._get_parquet_path("calo_hits", self.run_id * self.run_size)
        
        # NOTE: parquet event_id is the GLOBAL event ID, not local index
        result = self._load_parquet_calo(path, event_id=global_event_id)
        self._cache[cache_key] = result
        return result
    
    def load_edm4hep_event(self, local_event: int) -> "EDM4hepEventBatch":
        """Load EDM4hep event batch for a specific local event."""
        cache_key = f"edm4hep_{local_event}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        from pyedm4hep import EDM4hepEventBatch
        
        path = self._get_edm4hep_path()
        event_batch = EDM4hepEventBatch(str(path), events=(local_event, local_event + 1), condense_calo=False)
        self._cache[cache_key] = event_batch
        return event_batch
    
    def load_edm4hep_all_events(self) -> "EDM4hepEventBatch":
        """Load all EDM4hep events for the run."""
        cache_key = "edm4hep_all"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        from pyedm4hep import EDM4hepEventBatch
        
        path = self._get_edm4hep_path()
        event_batch = EDM4hepEventBatch(str(path), events=(0, self.run_size), condense_calo=False)
        self._cache[cache_key] = event_batch
        return event_batch
    
    def load_hepmc_hs_events(self) -> Dict[int, Any]:
        """Load all HepMC hard scatter events into a dictionary keyed by event number."""
        cache_key = "hepmc_hs"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        path = self._get_hepmc_hs_path()
        events_dict = {}
        with hep.open(str(path)) as f:
            for evt in f:
                events_dict[evt.event_number] = evt
        
        self._cache[cache_key] = events_dict
        return events_dict
    
    def load_event_number_mapping(self) -> List[int]:
        """Load the mapping from local event index to HepMC event number."""
        cache_key = "event_number_mapping"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        path = self._get_edm4hep_path()
        root_tree = uproot.open(str(path))["events"]
        event_numbers_array = root_tree["EventHeader/EventHeader.eventNumber"].arrays()
        
        mapping = []
        for local_evt_idx in range(self.run_size):
            evt_num = event_numbers_array[local_evt_idx]["EventHeader.eventNumber"].tolist()[0]
            mapping.append(evt_num)
        
        self._cache[cache_key] = mapping
        return mapping
    
    def load_acts_particles(self) -> pd.DataFrame:
        """Load particles from ACTS particles.root file."""
        cache_key = "acts_particles"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        path = self._get_particles_root_path()
        with uproot.open(str(path)) as f:
            tree = f["particles"]
            df = tree.arrays(library="pd")
        
        self._cache[cache_key] = df
        return df
    
    def load_acts_measurements(self) -> pd.DataFrame:
        """Load measurements from ACTS measurements.root file."""
        cache_key = "acts_measurements"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        path = self._get_measurements_root_path()
        with uproot.open(str(path)) as f:
            tree = f["measurements"]
            df = tree.arrays(library="pd")
        
        self._cache[cache_key] = df
        return df
    
    def load_acts_tracks(self) -> pd.DataFrame:
        """Load tracks from ACTS tracksummary_ambi.root file."""
        cache_key = "acts_tracks"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        path = self._get_tracks_root_path()
        with uproot.open(str(path)) as f:
            tree = f["tracksummary"]
            df = tree.arrays(library="pd")
        
        self._cache[cache_key] = df
        return df
    
    def _load_parquet_file(self, path: Path, event_id: Optional[int] = None) -> pd.DataFrame:
        """Load and explode a parquet file using Polars."""
        if self.pl is None:
            import pyarrow.parquet as pq
            df = pq.read_table(str(path)).to_pandas()
            if event_id is not None:
                df = df[df['event_id'] == event_id]
            return df
        
        df = self.pl.read_parquet(str(path))
        
        if df.is_empty():
            return pd.DataFrame()
        
        if event_id is not None:
            df = df.filter(self.pl.col('event_id') == event_id)
            if df.is_empty():
                return pd.DataFrame()
        
        # Check if data needs exploding
        non_event_cols = [c for c in df.columns if c != 'event_id']
        if not non_event_cols:
            return df.to_pandas()
        
        # Check if the column dtype is a List type
        if df[non_event_cols[0]].dtype == self.pl.List:
            df_exploded = df.explode(non_event_cols)
            return df_exploded.to_pandas()
        else:
            return df.to_pandas()
    
    def _load_parquet_tracks(self, path: Path, event_id: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load tracks parquet with special handling for hit_ids."""
        if self.pl is None:
            import pyarrow.parquet as pq
            df = pq.read_table(str(path)).to_pandas()
            if event_id is not None:
                df = df[df['event_id'] == event_id]
            hits_df = df[['event_id', 'track_id', 'hit_ids']].copy() if 'hit_ids' in df.columns else None
            tracks_df = df.drop(columns=['hit_ids']) if 'hit_ids' in df.columns else df
            return tracks_df, hits_df
        
        df = self.pl.read_parquet(str(path))
        
        if df.is_empty():
            return pd.DataFrame(), None
        
        if event_id is not None:
            df = df.filter(self.pl.col('event_id') == event_id)
            if df.is_empty():
                return pd.DataFrame(), None
        
        non_event_cols = [c for c in df.columns if c != 'event_id']
        has_hit_ids = 'hit_ids' in df.columns
        
        if df[non_event_cols[0]].dtype != self.pl.List:
            df_pandas = df.to_pandas()
            if has_hit_ids:
                hits_df = df_pandas[['event_id', 'track_id', 'hit_ids']].copy()
                tracks_df = df_pandas.drop(columns=['hit_ids'])
                return tracks_df, hits_df
            return df_pandas, None
        
        track_cols = [c for c in non_event_cols if c != 'hit_ids']
        
        if track_cols:
            df_tracks = df.select(['event_id'] + track_cols).explode(track_cols)
            tracks_df = df_tracks.to_pandas()
        else:
            tracks_df = pd.DataFrame()
        
        hits_df = None
        if has_hit_ids and 'track_id' in tracks_df.columns:
            df_hits = df.select(['event_id', 'hit_ids']).explode('hit_ids')
            hits_df = df_hits.to_pandas()
            hits_df['track_id'] = tracks_df['track_id'].values
        
        return tracks_df, hits_df
    
    def _load_parquet_calo(self, path: Path, event_id: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load calo hits parquet with contribution handling."""
        if self.pl is None:
            import pyarrow.parquet as pq
            df = pq.read_table(str(path)).to_pandas()
            if event_id is not None:
                df = df[df['event_id'] == event_id]
            return df, pd.DataFrame()
        
        lf = self.pl.scan_parquet(str(path))
        
        if event_id is not None:
            lf = lf.filter(self.pl.col('event_id') == event_id)
        
        try:
            schema = lf.collect_schema()
        except AttributeError:
            schema = lf.limit(0).collect().schema
        
        all_cols = schema.names()
        contrib_cols = [c for c in all_cols if c.startswith("contrib_")]
        cell_cols = [c for c in all_cols if c != "event_id" and c not in contrib_cols]
        
        if not cell_cols:
            return pd.DataFrame(), pd.DataFrame()
        
        lf_exploded = lf.explode(cell_cols + contrib_cols).with_row_index("cell_index")
        
        cells_df = (
            lf_exploded
            .select(["event_id", "cell_index"] + cell_cols)
            .collect()
            .to_pandas()
        )
        
        if contrib_cols:
            rename_exprs = [self.pl.col(c).alias(c.replace("contrib_", "")) for c in contrib_cols]
            contributions_df = (
                lf_exploded
                .select(["event_id", "cell_index"] + contrib_cols)
                .explode(contrib_cols)
                .select(["event_id", "cell_index"] + rename_exprs)
                .collect()
                .to_pandas()
            )
        else:
            contributions_df = pd.DataFrame()
        
        return cells_df, contributions_df
    
    def clear_cache(self):
        """Clear the data cache."""
        self._cache.clear()


class ConsistencyTest(ABC):
    """Base class for individual consistency tests."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
    
    @abstractmethod
    def run(self, loader: DataLoader, **kwargs) -> TestResult:
        """Run the test and return the result."""
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class TestSuite:
    """A collection of related tests."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.tests: List[ConsistencyTest] = []
    
    def add_test(self, test: ConsistencyTest):
        """Add a test to the suite."""
        self.tests.append(test)
    
    def run_all(self, loader: DataLoader, **kwargs) -> List[TestResult]:
        """Run all tests in the suite."""
        results = []
        for test in self.tests:
            start_time = time.time()
            try:
                result = test.run(loader, **kwargs)
                result.duration_ms = (time.time() - start_time) * 1000
            except Exception as e:
                result = TestResult(
                    name=test.name,
                    status=TestStatus.ERROR,
                    message=f"Exception: {str(e)}",
                    duration_ms=(time.time() - start_time) * 1000,
                )
                logger.exception(f"Error running test {test.name}")
            results.append(result)
        return results
    
    def __repr__(self):
        return f"TestSuite(name='{self.name}', tests={len(self.tests)})"


def print_test_results(results: List[TestResult], suite_name: str = ""):
    """Pretty print test results."""
    print("=" * 80)
    if suite_name:
        print(f"Test Suite: {suite_name}")
        print("=" * 80)
    
    passed = sum(1 for r in results if r.status == TestStatus.PASSED)
    failed = sum(1 for r in results if r.status == TestStatus.FAILED)
    skipped = sum(1 for r in results if r.status == TestStatus.SKIPPED)
    errors = sum(1 for r in results if r.status == TestStatus.ERROR)
    
    for result in results:
        print(result)
        if result.status in (TestStatus.FAILED, TestStatus.ERROR) and result.details:
            for key, value in result.details.items():
                print(f"    {key}: {value}")
    
    print("-" * 80)
    print(f"Summary: {passed} passed, {failed} failed, {skipped} skipped, {errors} errors")
    print(f"Total time: {sum(r.duration_ms for r in results):.1f} ms")
    print("=" * 80)
    
    return passed, failed, skipped, errors
