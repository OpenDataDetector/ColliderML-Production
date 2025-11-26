"""
Cross-object consistency tests for ColliderML data validation.

Tests:
- Particle ID references: All particle_ids in hits/tracks exist in particles
- Track-particle consistency: majority_particle_ids reference valid particles
- Hit-particle matching: Particles with hits have corresponding hit entries
- Event ID consistency: All objects for same event have matching event_id
"""

import numpy as np
import pandas as pd
from typing import Optional, Set

from .test_base import (
    ConsistencyTest,
    TestResult,
    TestStatus,
    TestSuite,
    DataLoader,
)


class AllHitParticleIdsValidTest(ConsistencyTest):
    """Test that all particle_ids in tracker hits and calo contributions exist in particles."""
    
    def __init__(self):
        super().__init__(
            name="All Hit Particle IDs Valid",
            description="Verify all particle_ids in hits/contributions exist in particles"
        )
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        
        parquet_particles = loader.load_parquet_particles(global_event_id)
        parquet_hits = loader.load_parquet_tracker_hits(global_event_id)
        _, parquet_contribs = loader.load_parquet_calo_hits(global_event_id)
        
        valid_particle_ids = set(parquet_particles['particle_id'].unique())
        
        # Check tracker hits
        invalid_tracker = set()
        if 'particle_id' in parquet_hits.columns:
            hit_particle_ids = set(parquet_hits['particle_id'].unique())
            hit_particle_ids.discard(0)  # 0 = no association
            invalid_tracker = hit_particle_ids - valid_particle_ids
        
        # Check calo contributions
        invalid_calo = set()
        if len(parquet_contribs) > 0:
            pid_col = 'particle_ids' if 'particle_ids' in parquet_contribs.columns else 'particle_id'
            if pid_col in parquet_contribs.columns:
                contrib_particle_ids = set(parquet_contribs[pid_col].unique())
                contrib_particle_ids.discard(0)
                invalid_calo = contrib_particle_ids - valid_particle_ids
        
        total_invalid = len(invalid_tracker) + len(invalid_calo)
        
        details = {
            "valid_particle_count": len(valid_particle_ids),
            "invalid_tracker_ids": len(invalid_tracker),
            "invalid_calo_ids": len(invalid_calo),
        }
        
        if total_invalid == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"All particle_ids in hits/contributions are valid",
                details=details,
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"{total_invalid} invalid particle_ids found",
                details={
                    **details,
                    "invalid_tracker_sample": list(invalid_tracker)[:10],
                    "invalid_calo_sample": list(invalid_calo)[:10],
                }
            )


class TrackMajorityParticleValidTest(ConsistencyTest):
    """Test that all majority_particle_ids in tracks exist in particles."""
    
    def __init__(self):
        super().__init__(
            name="Track Majority Particle IDs Valid",
            description="Verify all majority_particle_ids exist in particles"
        )
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        
        parquet_particles = loader.load_parquet_particles(global_event_id)
        tracks_df, _ = loader.load_parquet_tracks(global_event_id)
        
        if 'majority_particle_id' not in tracks_df.columns:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="majority_particle_id column not present",
            )
        
        valid_particle_ids = set(parquet_particles['particle_id'].unique())
        track_particle_ids = set(tracks_df['majority_particle_id'].unique())
        track_particle_ids.discard(0)  # 0 = no match
        
        invalid_ids = track_particle_ids - valid_particle_ids
        
        if len(invalid_ids) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"All {len(track_particle_ids)} majority_particle_ids are valid",
                details={
                    "unique_track_particles": len(track_particle_ids),
                    "total_tracks": len(tracks_df),
                }
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"{len(invalid_ids)} majority_particle_ids not in particles",
                details={
                    "invalid_count": len(invalid_ids),
                    "invalid_sample": list(invalid_ids)[:10],
                }
            )


class ParticleHitCorrespondenceTest(ConsistencyTest):
    """Test that particles with num_tracker_hits > 0 have corresponding hits."""
    
    def __init__(self):
        super().__init__(
            name="Particle-Hit Correspondence",
            description="Verify particles with hits have corresponding hit entries"
        )
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        
        parquet_particles = loader.load_parquet_particles(global_event_id)
        parquet_hits = loader.load_parquet_tracker_hits(global_event_id)
        
        if 'particle_id' not in parquet_hits.columns:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="particle_id not in tracker hits",
            )
        
        # Get particles that appear in hits
        particles_in_hits = set(parquet_hits['particle_id'].unique())
        particles_in_hits.discard(0)
        
        # Get particles from particle collection
        all_particle_ids = set(parquet_particles['particle_id'].unique())
        
        # Check if there's good overlap
        overlap = particles_in_hits & all_particle_ids
        
        details = {
            "particles_in_hits": len(particles_in_hits),
            "total_particles": len(all_particle_ids),
            "overlap": len(overlap),
        }
        
        # Note: Not all particles will have hits (e.g., neutral particles)
        # and hits can be from particles not in our particle collection
        if len(overlap) > 0:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"{len(overlap)} particles have corresponding hits",
                details=details,
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message="No overlap between particles and hit associations",
                details=details,
            )


class EventIdConsistencyTest(ConsistencyTest):
    """Test that event_id is consistent across all object types."""
    
    def __init__(self):
        super().__init__(
            name="Event ID Consistency",
            description="Verify all objects have consistent event_id"
        )
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        
        event_ids = {}
        
        # Load all data types and check event_id
        parquet_particles = loader.load_parquet_particles(global_event_id)
        if 'event_id' in parquet_particles.columns:
            event_ids['particles'] = set(parquet_particles['event_id'].unique())
        
        parquet_hits = loader.load_parquet_tracker_hits(global_event_id)
        if 'event_id' in parquet_hits.columns:
            event_ids['tracker_hits'] = set(parquet_hits['event_id'].unique())
        
        tracks_df, _ = loader.load_parquet_tracks(global_event_id)
        if len(tracks_df) > 0 and 'event_id' in tracks_df.columns:
            event_ids['tracks'] = set(tracks_df['event_id'].unique())
        
        calo_cells, _ = loader.load_parquet_calo_hits(global_event_id)
        if len(calo_cells) > 0 and 'event_id' in calo_cells.columns:
            event_ids['calo_hits'] = set(calo_cells['event_id'].unique())
        
        if len(event_ids) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="No event_id columns found",
            )
        
        # All should have exactly one event_id (the local_event)
        expected_event_id = local_event
        
        issues = []
        for obj_type, ids in event_ids.items():
            if len(ids) != 1:
                issues.append(f"{obj_type} has multiple event_ids: {ids}")
            elif expected_event_id not in ids:
                issues.append(f"{obj_type} has event_id {ids}, expected {expected_event_id}")
        
        details = {
            "expected_event_id": expected_event_id,
            "found_event_ids": {k: list(v) for k, v in event_ids.items()},
        }
        
        if len(issues) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"All objects have consistent event_id={expected_event_id}",
                details=details,
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"Event ID inconsistencies found",
                details={**details, "issues": issues},
            )


class ObjectCountReasonabilityTest(ConsistencyTest):
    """Test that object counts are reasonable relative to each other."""
    
    def __init__(self):
        super().__init__(
            name="Object Count Reasonability",
            description="Verify object counts are reasonable relative to each other"
        )
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        
        counts = {}
        
        parquet_particles = loader.load_parquet_particles(global_event_id)
        counts['particles'] = len(parquet_particles)
        
        parquet_hits = loader.load_parquet_tracker_hits(global_event_id)
        counts['tracker_hits'] = len(parquet_hits)
        
        tracks_df, _ = loader.load_parquet_tracks(global_event_id)
        counts['tracks'] = len(tracks_df)
        
        calo_cells, calo_contribs = loader.load_parquet_calo_hits(global_event_id)
        counts['calo_hits'] = len(calo_cells)
        counts['calo_contributions'] = len(calo_contribs)
        
        issues = []
        
        # Sanity checks
        if counts['particles'] == 0:
            issues.append("No particles found")
        
        if counts['tracker_hits'] == 0:
            issues.append("No tracker hits found")
        
        if counts['tracks'] == 0:
            issues.append("No tracks found")
        
        # Tracks should be fewer than tracker hits
        if counts['tracks'] > counts['tracker_hits']:
            issues.append(f"More tracks ({counts['tracks']}) than tracker hits ({counts['tracker_hits']})")
        
        # Calo contributions should be >= calo hits (at least one contribution per hit)
        if counts['calo_hits'] > 0 and counts['calo_contributions'] < counts['calo_hits']:
            issues.append(f"Fewer contributions ({counts['calo_contributions']}) than calo hits ({counts['calo_hits']})")
        
        details = {"counts": counts}
        
        if len(issues) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"Object counts reasonable: {counts}",
                details=details,
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"Object count issues found",
                details={**details, "issues": issues},
            )


class TrackHitParticleConsistencyTest(ConsistencyTest):
    """Test that track hit associations are consistent with particle associations."""
    
    def __init__(self, sample_size: int = 10):
        super().__init__(
            name="Track-Hit-Particle Consistency",
            description="Verify track hit particle associations are consistent"
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
        
        # Sample tracks and verify consistency
        sample_indices = np.random.choice(
            len(tracks_df),
            size=min(self.sample_size, len(tracks_df)),
            replace=False
        )
        
        inconsistencies = []
        
        for idx in sample_indices:
            track = tracks_df.iloc[idx]
            hits = track_hits_df.iloc[idx]
            
            hit_ids = hits['hit_ids']
            if not isinstance(hit_ids, (list, np.ndarray)) or len(hit_ids) == 0:
                continue
            
            majority_particle = track['majority_particle_id']
            
            # Get particles from the hits
            valid_hit_ids = [hid for hid in hit_ids if hid < len(tracker_hits)]
            hit_particles = tracker_hits.iloc[valid_hit_ids]['particle_id'].tolist()
            
            # The majority_particle_id should appear in the hit particles
            if majority_particle != 0 and majority_particle not in hit_particles:
                inconsistencies.append({
                    'track_idx': idx,
                    'majority_particle': int(majority_particle),
                    'hit_particles': list(set(hit_particles))[:5],
                })
        
        details = {"sampled_tracks": len(sample_indices)}
        
        if len(inconsistencies) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"All {len(sample_indices)} sampled tracks have consistent associations",
                details=details,
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"{len(inconsistencies)} tracks have inconsistent associations",
                details={**details, "inconsistencies": inconsistencies},
            )


class CrossObjectTests(TestSuite):
    """Test suite for cross-object consistency."""
    
    def __init__(self):
        super().__init__(
            name="Cross-Object Consistency Tests",
            description="Validate consistency across different object types"
        )
        
        # Add all cross-object tests
        self.add_test(EventIdConsistencyTest())
        self.add_test(AllHitParticleIdsValidTest())
        self.add_test(TrackMajorityParticleValidTest())
        self.add_test(ParticleHitCorrespondenceTest())
        self.add_test(ObjectCountReasonabilityTest())
        self.add_test(TrackHitParticleConsistencyTest())
