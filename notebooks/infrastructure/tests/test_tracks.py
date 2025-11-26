"""
Track consistency tests for ColliderML data validation.

Tests:
- Majority particle matching: majority_particle_id correctly computed from hit modes
- Hit ID validity: All hit_ids reference valid tracker hits
- Track parameter ranges: phi, theta, d0, z0, qop in reasonable ranges
- Efficiency/purity metrics: Track matching to true particles
- Track completeness: All ACTS tracks present in parquet
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from collections import Counter

from .test_base import (
    ConsistencyTest,
    TestResult,
    TestStatus,
    TestSuite,
    DataLoader,
)


class TrackHitIdValidityTest(ConsistencyTest):
    """Test that all hit_ids in tracks reference valid tracker hits."""
    
    def __init__(self):
        super().__init__(
            name="Track Hit ID Validity",
            description="Verify all hit_ids in tracks exist in tracker hits"
        )
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        
        tracks_df, track_hits_df = loader.load_parquet_tracks(global_event_id)
        tracker_hits = loader.load_parquet_tracker_hits(global_event_id)
        
        if track_hits_df is None or 'hit_ids' not in track_hits_df.columns:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="hit_ids not available in tracks",
            )
        
        # Get all valid hit indices
        valid_hit_indices = set(range(len(tracker_hits)))
        
        # Collect all hit_ids from tracks
        all_track_hit_ids = set()
        invalid_hit_ids = []
        
        for idx, row in track_hits_df.iterrows():
            hit_ids = row['hit_ids']
            if isinstance(hit_ids, (list, np.ndarray)):
                for hid in hit_ids:
                    all_track_hit_ids.add(hid)
                    if hid not in valid_hit_indices:
                        invalid_hit_ids.append((idx, hid))
        
        if len(invalid_hit_ids) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"All {len(all_track_hit_ids)} unique hit_ids are valid",
                details={
                    "unique_hit_ids": len(all_track_hit_ids),
                    "total_tracks": len(tracks_df),
                    "total_tracker_hits": len(tracker_hits),
                }
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"{len(invalid_hit_ids)} invalid hit_ids found",
                details={
                    "invalid_count": len(invalid_hit_ids),
                    "invalid_sample": invalid_hit_ids[:10],
                    "unique_hit_ids": len(all_track_hit_ids),
                }
            )


class TrackMajorityParticleTest(ConsistencyTest):
    """Test that majority_particle_id is correctly computed from hit modes."""
    
    def __init__(self, sample_size: int = 10):
        super().__init__(
            name="Track Majority Particle Computation",
            description="Verify majority_particle_id = mode of particle_ids from hits"
        )
        self.sample_size = sample_size
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        
        tracks_df, track_hits_df = loader.load_parquet_tracks(global_event_id)
        tracker_hits = loader.load_parquet_tracker_hits(global_event_id)
        
        if track_hits_df is None or 'hit_ids' not in track_hits_df.columns:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="hit_ids not available in tracks",
            )
        
        if 'majority_particle_id' not in tracks_df.columns:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="majority_particle_id not in tracks",
            )
        
        if 'particle_id' not in tracker_hits.columns:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="particle_id not in tracker hits",
            )
        
        # Sample tracks to verify
        sample_indices = np.random.choice(
            len(tracks_df), 
            size=min(self.sample_size, len(tracks_df)), 
            replace=False
        )
        
        mismatches = []
        
        for idx in sample_indices:
            track_row = tracks_df.iloc[idx]
            hits_row = track_hits_df.iloc[idx]
            
            hit_ids = hits_row['hit_ids']
            if not isinstance(hit_ids, (list, np.ndarray)) or len(hit_ids) == 0:
                continue
            
            # Get particle_ids for these hits
            valid_hit_ids = [hid for hid in hit_ids if hid < len(tracker_hits)]
            particle_ids = tracker_hits.iloc[valid_hit_ids]['particle_id'].tolist()
            
            if len(particle_ids) == 0:
                continue
            
            # Compute mode
            counter = Counter(particle_ids)
            computed_majority = counter.most_common(1)[0][0]
            stored_majority = track_row['majority_particle_id']
            
            if computed_majority != stored_majority:
                mismatches.append({
                    'track_idx': idx,
                    'stored': int(stored_majority),
                    'computed': int(computed_majority),
                    'hit_count': len(hit_ids),
                })
        
        if len(mismatches) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"All {len(sample_indices)} sampled tracks have correct majority_particle_id",
                details={"sampled_tracks": len(sample_indices)}
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"{len(mismatches)}/{len(sample_indices)} tracks have incorrect majority_particle_id",
                details={
                    "mismatch_count": len(mismatches),
                    "mismatches": mismatches,
                }
            )


class TrackParameterRangesTest(ConsistencyTest):
    """Test that track parameters are within physical ranges."""
    
    def __init__(self):
        super().__init__(
            name="Track Parameter Ranges",
            description="Verify phi, theta, d0, z0, qop are in reasonable ranges"
        )
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        tracks_df, _ = loader.load_parquet_tracks(global_event_id)
        
        if len(tracks_df) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="No tracks found",
            )
        
        issues = []
        details = {}
        
        # Check phi: should be in [-π, π]
        if 'phi' in tracks_df.columns:
            phi = tracks_df['phi']
            out_of_range = ((phi < -np.pi) | (phi > np.pi)).sum()
            details['phi_range'] = [float(phi.min()), float(phi.max())]
            if out_of_range > 0:
                issues.append(f"{out_of_range} tracks with phi outside [-π, π]")
        
        # Check theta: should be in [0, π]
        if 'theta' in tracks_df.columns:
            theta = tracks_df['theta']
            out_of_range = ((theta < 0) | (theta > np.pi)).sum()
            details['theta_range'] = [float(theta.min()), float(theta.max())]
            if out_of_range > 0:
                issues.append(f"{out_of_range} tracks with theta outside [0, π]")
        
        # Check d0: typically < 100mm for reasonable tracks
        if 'd0' in tracks_df.columns:
            d0 = tracks_df['d0']
            large_d0 = (np.abs(d0) > 100).sum()
            details['d0_range'] = [float(d0.min()), float(d0.max())]
            if large_d0 > len(tracks_df) * 0.1:  # More than 10% with large d0
                issues.append(f"{large_d0} tracks with |d0| > 100mm")
        
        # Check z0: typically < 500mm for reasonable tracks
        if 'z0' in tracks_df.columns:
            z0 = tracks_df['z0']
            large_z0 = (np.abs(z0) > 500).sum()
            details['z0_range'] = [float(z0.min()), float(z0.max())]
            if large_z0 > len(tracks_df) * 0.1:  # More than 10% with large z0
                issues.append(f"{large_z0} tracks with |z0| > 500mm")
        
        # Check qop: should be non-zero for valid tracks
        if 'qop' in tracks_df.columns:
            qop = tracks_df['qop']
            zero_qop = (qop == 0).sum()
            details['qop_range'] = [float(qop.min()), float(qop.max())]
            if zero_qop > 0:
                issues.append(f"{zero_qop} tracks with qop = 0")
        
        details['total_tracks'] = len(tracks_df)
        
        if len(issues) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"All {len(tracks_df)} tracks have valid parameter ranges",
                details=details,
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"{len(issues)} parameter range issues found",
                details={**details, "issues": issues},
            )


class TrackEfficiencyPurityTest(ConsistencyTest):
    """Test track efficiency and purity metrics."""
    
    def __init__(self, min_efficiency: float = 0.5, min_purity: float = 0.5):
        super().__init__(
            name="Track Efficiency and Purity",
            description="Verify track-particle matching efficiency and purity"
        )
        self.min_efficiency = min_efficiency
        self.min_purity = min_purity
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        
        tracks_df, track_hits_df = loader.load_parquet_tracks(global_event_id)
        tracker_hits = loader.load_parquet_tracker_hits(global_event_id)
        
        if track_hits_df is None or len(tracks_df) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="No tracks or hit associations found",
            )
        
        # Compute efficiency and purity for each track
        efficiencies = []
        purities = []
        
        for idx in range(len(tracks_df)):
            track_row = tracks_df.iloc[idx]
            hits_row = track_hits_df.iloc[idx]
            
            hit_ids = hits_row['hit_ids']
            if not isinstance(hit_ids, (list, np.ndarray)) or len(hit_ids) == 0:
                continue
            
            majority_particle_id = track_row.get('majority_particle_id', None)
            if majority_particle_id is None or majority_particle_id == 0:
                continue
            
            # Get hits for this track
            valid_hit_ids = [hid for hid in hit_ids if hid < len(tracker_hits)]
            track_particle_ids = set(tracker_hits.iloc[valid_hit_ids]['particle_id'].tolist())
            
            # Get all hits for the majority particle
            particle_hit_mask = tracker_hits['particle_id'] == majority_particle_id
            particle_hit_indices = set(tracker_hits[particle_hit_mask].index.tolist())
            
            # Compute shared hits
            shared_hits = len(set(valid_hit_ids) & particle_hit_indices)
            
            # Efficiency: shared_hits / total_particle_hits
            if len(particle_hit_indices) > 0:
                efficiency = shared_hits / len(particle_hit_indices)
                efficiencies.append(efficiency)
            
            # Purity: shared_hits / total_track_hits
            if len(valid_hit_ids) > 0:
                purity = shared_hits / len(valid_hit_ids)
                purities.append(purity)
        
        if len(efficiencies) == 0 or len(purities) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="Could not compute efficiency/purity",
            )
        
        mean_efficiency = np.mean(efficiencies)
        mean_purity = np.mean(purities)
        
        details = {
            "mean_efficiency": float(mean_efficiency),
            "mean_purity": float(mean_purity),
            "std_efficiency": float(np.std(efficiencies)),
            "std_purity": float(np.std(purities)),
            "num_tracks_analyzed": len(efficiencies),
        }
        
        if mean_efficiency >= self.min_efficiency and mean_purity >= self.min_purity:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"Mean efficiency: {mean_efficiency:.2f}, Mean purity: {mean_purity:.2f}",
                details=details,
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"Low efficiency ({mean_efficiency:.2f}) or purity ({mean_purity:.2f})",
                details=details,
            )


class TrackHitCountTest(ConsistencyTest):
    """Test that tracks have reasonable numbers of hits."""
    
    def __init__(self, min_hits: int = 3, max_hits: int = 30):
        super().__init__(
            name="Track Hit Count",
            description="Verify tracks have reasonable numbers of hits"
        )
        self.min_hits = min_hits
        self.max_hits = max_hits
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        
        tracks_df, track_hits_df = loader.load_parquet_tracks(global_event_id)
        
        if track_hits_df is None or 'hit_ids' not in track_hits_df.columns:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="hit_ids not available",
            )
        
        # Count hits per track
        hit_counts = track_hits_df['hit_ids'].apply(
            lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
        )
        
        too_few = (hit_counts < self.min_hits).sum()
        too_many = (hit_counts > self.max_hits).sum()
        
        details = {
            "mean_hits": float(hit_counts.mean()),
            "min_hits": int(hit_counts.min()),
            "max_hits": int(hit_counts.max()),
            "tracks_below_min": int(too_few),
            "tracks_above_max": int(too_many),
            "total_tracks": len(tracks_df),
        }
        
        if too_few == 0 and too_many == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"All tracks have {self.min_hits}-{self.max_hits} hits (mean: {hit_counts.mean():.1f})",
                details=details,
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"{too_few} tracks with <{self.min_hits} hits, {too_many} with >{self.max_hits}",
                details=details,
            )


class TrackIdUniquenessTest(ConsistencyTest):
    """Test that track_ids are unique within an event."""
    
    def __init__(self):
        super().__init__(
            name="Track ID Uniqueness",
            description="Verify track_ids are unique within each event"
        )
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        tracks_df, _ = loader.load_parquet_tracks(global_event_id)
        
        if 'track_id' not in tracks_df.columns:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="track_id column not present",
            )
        
        total_tracks = len(tracks_df)
        unique_tracks = tracks_df['track_id'].nunique()
        
        if total_tracks == unique_tracks:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"All {total_tracks} track_ids are unique",
            )
        else:
            duplicates = tracks_df[tracks_df.duplicated(subset=['track_id'], keep=False)]
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"{total_tracks - unique_tracks} duplicate track_ids",
                details={
                    "total_tracks": total_tracks,
                    "unique_tracks": unique_tracks,
                    "duplicate_sample": duplicates['track_id'].head(10).tolist(),
                }
            )


class TrackTests(TestSuite):
    """Test suite for track data consistency."""
    
    def __init__(self):
        super().__init__(
            name="Track Tests",
            description="Validate reconstructed track data and truth matching"
        )
        
        # Add all track tests
        self.add_test(TrackIdUniquenessTest())
        self.add_test(TrackHitIdValidityTest())
        self.add_test(TrackMajorityParticleTest())
        self.add_test(TrackParameterRangesTest())
        self.add_test(TrackHitCountTest())
        self.add_test(TrackEfficiencyPurityTest())
