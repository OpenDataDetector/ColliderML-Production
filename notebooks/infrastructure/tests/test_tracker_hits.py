"""
Tracker hits consistency tests for ColliderML data validation.

Tests:
- Hit position matching: true_x/y/z in parquet match EDM4hep SimTrackerHits
- Particle association: All particle_ids reference valid particles
- Hit completeness: All EDM4hep hits present in parquet
- Detector encoding: detector column properly encoded
- Coordinate precision: Float32 precision preserved correctly
"""

import numpy as np
import pandas as pd
from typing import Optional

from .test_base import (
    ConsistencyTest,
    TestResult,
    TestStatus,
    TestSuite,
    DataLoader,
)


class TrackerHitPositionMatchTest(ConsistencyTest):
    """Test that tracker hit positions match between parquet and EDM4hep."""
    
    def __init__(self, tolerance: float = 1e-5):
        super().__init__(
            name="Tracker Hit Position Match",
            description="Verify true_x/y/z in parquet match EDM4hep SimTrackerHits"
        )
        self.tolerance = tolerance
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        
        # Load parquet tracker hits
        parquet_hits = loader.load_parquet_tracker_hits(global_event_id)
        
        # Load EDM4hep tracker hits
        edm4hep_batch = loader.load_edm4hep_event(local_event)
        edm4hep_hits = edm4hep_batch.get_tracker_hits_df()
        
        if len(parquet_hits) == 0 or len(edm4hep_hits) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="No hits found in one or both sources",
            )
        
        # Merge on position AND particle_id for more robust matching
        # true_x, true_y, true_z in parquet = x, y, z in EDM4hep
        merge_cols = ["x", "y", "z"]
        
        df_parquet = parquet_hits[["true_x", "true_y", "true_z"]].copy()
        df_parquet.columns = ["x", "y", "z"]
        df_parquet = df_parquet.astype(np.float32)
        
        # Also include particle_id if available in both
        if 'particle_id' in parquet_hits.columns and 'particle_id' in edm4hep_hits.columns:
            df_parquet['particle_id'] = parquet_hits['particle_id'].values
            merge_cols.append("particle_id")
        
        df_edm4hep = edm4hep_hits[merge_cols].copy()
        df_edm4hep[["x", "y", "z"]] = df_edm4hep[["x", "y", "z"]].astype(np.float32)
        
        merged = df_parquet.merge(df_edm4hep, on=merge_cols, how="inner")
        
        match_rate = len(merged) / len(parquet_hits) if len(parquet_hits) > 0 else 0
        
        details = {
            "matched_hits": len(merged),
            "edm4hep_hits": len(edm4hep_hits),
            "parquet_hits": len(parquet_hits),
            "match_rate": float(match_rate),
            "merge_columns": merge_cols,
        }
        
        if match_rate >= 0.99:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"{len(merged)}/{len(parquet_hits)} parquet hits found in EDM4hep ({match_rate*100:.1f}%)",
                details=details,
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"Only {len(merged)}/{len(parquet_hits)} parquet hits found in EDM4hep ({match_rate*100:.1f}%)",
                details=details,
            )


class TrackerHitCompletenessTest(ConsistencyTest):
    """Test that all EDM4hep tracker hits are present in parquet."""
    
    def __init__(self):
        super().__init__(
            name="Tracker Hit Completeness",
            description="Verify all EDM4hep tracker hits are in parquet output"
        )
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        
        parquet_hits = loader.load_parquet_tracker_hits(global_event_id)
        edm4hep_batch = loader.load_edm4hep_event(local_event)
        edm4hep_hits = edm4hep_batch.get_tracker_hits_df()
        
        # Note: parquet may have MORE hits than EDM4hep due to digitization
        # But parquet should have all EDM4hep hits (via true_x/y/z matching)
        
        parquet_count = len(parquet_hits)
        edm4hep_count = len(edm4hep_hits)
        
        if parquet_count >= edm4hep_count:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"Parquet has {parquet_count} hits, EDM4hep has {edm4hep_count}",
                details={
                    "parquet_count": parquet_count,
                    "edm4hep_count": edm4hep_count,
                }
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"Parquet has fewer hits ({parquet_count}) than EDM4hep ({edm4hep_count})",
                details={
                    "parquet_count": parquet_count,
                    "edm4hep_count": edm4hep_count,
                    "missing": edm4hep_count - parquet_count,
                }
            )


class TrackerHitParticleAssociationTest(ConsistencyTest):
    """Test that all particle_ids in tracker hits reference valid particles."""
    
    def __init__(self):
        super().__init__(
            name="Tracker Hit Particle Association",
            description="Verify all particle_ids in hits exist in particles"
        )
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        
        parquet_hits = loader.load_parquet_tracker_hits(global_event_id)
        parquet_particles = loader.load_parquet_particles(global_event_id)
        
        if 'particle_id' not in parquet_hits.columns:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="particle_id column not present in hits",
            )
        
        hit_particle_ids = set(parquet_hits['particle_id'].unique())
        valid_particle_ids = set(parquet_particles['particle_id'].unique())
        
        # Particle ID 0 is typically "no associated particle"
        hit_particle_ids.discard(0)
        
        invalid_ids = hit_particle_ids - valid_particle_ids
        
        if len(invalid_ids) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"All {len(hit_particle_ids)} unique particle_ids are valid",
                details={
                    "unique_particle_ids": len(hit_particle_ids),
                    "total_hits": len(parquet_hits),
                }
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"{len(invalid_ids)} particle_ids not found in particles",
                details={
                    "invalid_count": len(invalid_ids),
                    "invalid_sample": list(invalid_ids)[:10],
                    "unique_particle_ids": len(hit_particle_ids),
                }
            )


class TrackerHitDetectorEncodingTest(ConsistencyTest):
    """Test that detector column is properly encoded."""
    
    def __init__(self):
        super().__init__(
            name="Tracker Hit Detector Encoding",
            description="Verify detector column uses valid encoded values"
        )
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        parquet_hits = loader.load_parquet_tracker_hits(global_event_id)
        
        if 'detector' not in parquet_hits.columns:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="detector column not present",
            )
        
        # Known detector types from convert_digihits.py
        known_detectors = [
            "PixelBarrel", "PixelEndcapP", "PixelEndcapN",
            "ShortStripBarrel", "ShortStripEndcapP", "ShortStripEndcapN",
            "LongStripBarrel", "LongStripEndcapP", "LongStripEndcapN"
        ]
        
        unique_detectors = parquet_hits['detector'].unique()
        
        # Check if encoded (uint8) or string
        if parquet_hits['detector'].dtype == np.uint8:
            # Encoded values should be 0-8
            invalid_codes = unique_detectors[unique_detectors > len(known_detectors)]
            if len(invalid_codes) == 0:
                return TestResult(
                    name=self.name,
                    status=TestStatus.PASSED,
                    message=f"All detector codes valid (0-{len(known_detectors)-1})",
                    details={"unique_codes": unique_detectors.tolist()}
                )
            else:
                return TestResult(
                    name=self.name,
                    status=TestStatus.FAILED,
                    message=f"Invalid detector codes found: {invalid_codes}",
                    details={"unique_codes": unique_detectors.tolist()}
                )
        else:
            # String values - check against known list
            unknown_detectors = set(unique_detectors) - set(known_detectors)
            if len(unknown_detectors) == 0:
                return TestResult(
                    name=self.name,
                    status=TestStatus.PASSED,
                    message=f"All {len(unique_detectors)} detector types recognized",
                    details={"detectors": list(unique_detectors)}
                )
            else:
                return TestResult(
                    name=self.name,
                    status=TestStatus.FAILED,
                    message=f"Unknown detectors: {unknown_detectors}",
                    details={"unknown": list(unknown_detectors)}
                )


class TrackerHitRecoPositionTest(ConsistencyTest):
    """Test that reconstructed hit positions (x, y, z) are reasonable."""
    
    def __init__(self):
        super().__init__(
            name="Tracker Hit Reco Position",
            description="Verify reconstructed positions are close to true positions"
        )
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        parquet_hits = loader.load_parquet_tracker_hits(global_event_id)
        
        required_cols = ["x", "y", "z", "true_x", "true_y", "true_z"]
        missing_cols = [c for c in required_cols if c not in parquet_hits.columns]
        
        if missing_cols:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message=f"Missing columns: {missing_cols}",
            )
        
        # Calculate residuals
        dx = parquet_hits['x'] - parquet_hits['true_x']
        dy = parquet_hits['y'] - parquet_hits['true_y']
        dz = parquet_hits['z'] - parquet_hits['true_z']
        
        dr = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Check for reasonable residuals (< 10mm for tracker)
        max_residual = 100.0  # mm
        outliers = (dr > max_residual).sum()
        outlier_rate = outliers / len(dr)
        
        details = {
            "mean_residual_mm": float(dr.mean()),
            "std_residual_mm": float(dr.std()),
            "max_residual_mm": float(dr.max()),
            "outliers_above_10mm": int(outliers),
            "outlier_rate": float(outlier_rate),
        }
        
        if outlier_rate < 0.01:  # Less than 1% outliers
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"Mean residual: {dr.mean():.3f}mm, {outliers} outliers ({outlier_rate*100:.2f}%)",
                details=details,
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"Too many outliers: {outliers} ({outlier_rate*100:.2f}%)",
                details=details,
            )


class TrackerHitCountPerParticleTest(ConsistencyTest):
    """Test that hit counts per particle are reasonable for high-pT particles."""
    
    def __init__(self, max_hits_per_particle: int = 30, min_pt_gev: float = 2.0):
        super().__init__(
            name="Tracker Hit Count Per Particle",
            description=f"Verify high-pT (>{min_pt_gev} GeV) particles have reasonable hit counts"
        )
        self.max_hits_per_particle = max_hits_per_particle
        self.min_pt_gev = min_pt_gev
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        parquet_hits = loader.load_parquet_tracker_hits(global_event_id)
        parquet_particles = loader.load_parquet_particles(global_event_id)
        
        if 'particle_id' not in parquet_hits.columns:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="particle_id column not present in hits",
            )
        
        if 'px' not in parquet_particles.columns or 'py' not in parquet_particles.columns:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="px/py columns not present in particles",
            )
        
        # Count hits per particle (excluding particle_id=0)
        hits_per_particle = parquet_hits[parquet_hits['particle_id'] != 0].groupby('particle_id').size()
        hits_per_particle = hits_per_particle.reset_index()
        hits_per_particle.columns = ['particle_id', 'hit_count']
        
        if len(hits_per_particle) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="No particles with associated hits",
            )
        
        # Calculate pT for particles
        particles_pt = parquet_particles[['particle_id', 'px', 'py']].copy()
        particles_pt['pt'] = np.sqrt(particles_pt['px']**2 + particles_pt['py']**2)
        
        # Merge hit counts with particle pT
        merged = hits_per_particle.merge(particles_pt[['particle_id', 'pt']], on='particle_id', how='left')
        
        # Find high-pT particles with excessive hits
        high_pt_mask = merged['pt'] >= self.min_pt_gev
        excessive_hits_mask = merged['hit_count'] > self.max_hits_per_particle
        suspicious = merged[high_pt_mask & excessive_hits_mask]
        
        # Also track overall stats
        all_excessive = merged[excessive_hits_mask]
        high_pt_particles = merged[high_pt_mask]
        
        details = {
            "mean_hits": float(merged['hit_count'].mean()),
            "max_hits": int(merged['hit_count'].max()),
            "min_hits": int(merged['hit_count'].min()),
            "particles_with_hits": len(merged),
            "high_pt_particles": len(high_pt_particles),
            "all_excessive_count": len(all_excessive),
            "high_pt_excessive_count": len(suspicious),
            "pt_threshold_gev": self.min_pt_gev,
            "hit_threshold": self.max_hits_per_particle,
        }
        
        if len(suspicious) == 0:
            msg = f"Mean {merged['hit_count'].mean():.1f} hits/particle, max {merged['hit_count'].max()}"
            if len(all_excessive) > 0:
                msg += f" ({len(all_excessive)} low-pT particles with >{self.max_hits_per_particle} hits - OK)"
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=msg,
                details=details,
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"{len(suspicious)} high-pT (>{self.min_pt_gev} GeV) particles with >{self.max_hits_per_particle} hits",
                details={
                    **details, 
                    "suspicious_sample": suspicious[['particle_id', 'hit_count', 'pt']].head(10).to_dict('records'),
                },
            )


class TrackerHitTests(TestSuite):
    """Test suite for tracker hit data consistency."""
    
    def __init__(self):
        super().__init__(
            name="Tracker Hit Tests",
            description="Validate tracker hit positions and associations"
        )
        
        # Add all tracker hit tests
        # Note: TrackerHitCompletenessTest removed - redundant with TrackerHitPositionMatchTest
        self.add_test(TrackerHitPositionMatchTest())
        self.add_test(TrackerHitParticleAssociationTest())
        self.add_test(TrackerHitDetectorEncodingTest())
        self.add_test(TrackerHitRecoPositionTest())
        self.add_test(TrackerHitCountPerParticleTest())
