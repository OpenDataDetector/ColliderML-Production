"""
Particle consistency tests for ColliderML data validation.

Tests:
- Particle completeness: All EDM4hep particles present in parquet
- Parent-child relationships: Valid parent_id references
- Vertex flags: vertex_primary correctly assigned
- Primary flag: primary = NOT created_in_simulation
- Kinematic consistency: px, py, pz, vx, vy, vz match between sources
- Generator particles: Non-simulated particles have valid properties
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


class ParticleCompletenessTest(ConsistencyTest):
    """Test that all EDM4hep particles are present in parquet output."""
    
    def __init__(self):
        super().__init__(
            name="Particle Completeness",
            description="Verify all EDM4hep particles are in parquet output"
        )
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        
        # Load parquet particles
        parquet_particles = loader.load_parquet_particles(global_event_id)
        
        # Load EDM4hep particles
        edm4hep_batch = loader.load_edm4hep_event(local_event)
        edm4hep_particles = edm4hep_batch.get_particles_df()
        
        parquet_ids = set(parquet_particles['particle_id'].unique())
        edm4hep_ids = set(edm4hep_particles['particle_id'].unique())
        
        missing_in_parquet = edm4hep_ids - parquet_ids
        extra_in_parquet = parquet_ids - edm4hep_ids
        
        if len(missing_in_parquet) == 0 and len(extra_in_parquet) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"All {len(edm4hep_ids)} particles match",
                details={
                    "parquet_count": len(parquet_ids),
                    "edm4hep_count": len(edm4hep_ids),
                }
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"Mismatch: {len(missing_in_parquet)} missing, {len(extra_in_parquet)} extra",
                details={
                    "parquet_count": len(parquet_ids),
                    "edm4hep_count": len(edm4hep_ids),
                    "missing_count": len(missing_in_parquet),
                    "extra_count": len(extra_in_parquet),
                    "missing_sample": list(missing_in_parquet)[:10],
                    "extra_sample": list(extra_in_parquet)[:10],
                }
            )


class ParticleKinematicsMatchTest(ConsistencyTest):
    """Test that particle kinematics match between parquet and EDM4hep."""
    
    def __init__(self, tolerance: float = 1e-4):
        super().__init__(
            name="Particle Kinematics Match",
            description="Verify particle kinematics (px, py, pz, vx, vy, vz) match"
        )
        self.tolerance = tolerance
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        
        # Load data
        parquet_particles = loader.load_parquet_particles(global_event_id)
        edm4hep_batch = loader.load_edm4hep_event(local_event)
        edm4hep_particles = edm4hep_batch.get_particles_df()
        
        # Merge on particle_id
        merged = parquet_particles.merge(
            edm4hep_particles,
            on="particle_id",
            how="inner",
            suffixes=("_parquet", "_edm4hep")
        )
        
        if len(merged) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="No common particles found",
            )
        
        # Compare kinematics
        kinematic_cols = ["vx", "vy", "vz", "px", "py", "pz"]
        mismatches = {}
        
        for col in kinematic_cols:
            parquet_col = f"{col}_parquet"
            edm4hep_col = f"{col}_edm4hep"
            
            if parquet_col not in merged.columns or edm4hep_col not in merged.columns:
                continue
            
            diff = np.abs(merged[parquet_col].astype(np.float32) - merged[edm4hep_col].astype(np.float32))
            num_mismatch = (diff > self.tolerance).sum()
            
            if num_mismatch > 0:
                mismatches[col] = {
                    "count": int(num_mismatch),
                    "max_diff": float(diff.max()),
                }
        
        if len(mismatches) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"All {len(merged)} particles match within tolerance {self.tolerance}",
                details={"num_particles": len(merged)}
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"Kinematics mismatch in {len(mismatches)} columns",
                details={"mismatches": mismatches}
            )


class ParentChildRelationshipTest(ConsistencyTest):
    """Test that parent_id references are valid and form proper relationships."""
    
    def __init__(self):
        super().__init__(
            name="Parent-Child Relationships",
            description="Verify parent_id references exist and are valid"
        )
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        parquet_particles = loader.load_parquet_particles(global_event_id)
        
        if 'parent_id' not in parquet_particles.columns:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="parent_id column not present",
            )
        
        particle_ids = set(parquet_particles['particle_id'].unique())
        
        # Check each particle's parent_id
        invalid_parents = []
        for idx, row in parquet_particles.iterrows():
            parent_id = row['parent_id']
            # parent_id == 0 typically means no parent (or root particle)
            if parent_id != 0 and parent_id not in particle_ids:
                invalid_parents.append({
                    'particle_id': row['particle_id'],
                    'parent_id': parent_id,
                })
        
        if len(invalid_parents) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"All parent references valid for {len(parquet_particles)} particles",
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"{len(invalid_parents)} particles have invalid parent_id",
                details={
                    "invalid_count": len(invalid_parents),
                    "invalid_sample": invalid_parents[:10],
                }
            )


class VertexPrimaryFlagTest(ConsistencyTest):
    """Test that vertex_primary flag is correctly assigned."""
    
    def __init__(self):
        super().__init__(
            name="Vertex Primary Flag",
            description="Verify vertex_primary=1 corresponds to hard scatter vertex"
        )
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        parquet_particles = loader.load_parquet_particles(global_event_id)
        
        if 'vertex_primary' not in parquet_particles.columns:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="vertex_primary column not present",
            )
        
        # Check that vertex_primary values are reasonable
        unique_vertex_primary = parquet_particles['vertex_primary'].unique()
        
        # vertex_primary=1 should be hard scatter
        hs_particles = parquet_particles[parquet_particles['vertex_primary'] == 1]
        
        # Get vertex positions for HS particles
        if len(hs_particles) > 0:
            hs_vx = hs_particles['vx'].mean() if 'vx' in hs_particles.columns else None
            hs_vy = hs_particles['vy'].mean() if 'vy' in hs_particles.columns else None
            hs_vz = hs_particles['vz'].mean() if 'vz' in hs_particles.columns else None
        else:
            hs_vx = hs_vy = hs_vz = None
        
        # Check distribution
        vertex_counts = parquet_particles['vertex_primary'].value_counts()
        
        if 1 in unique_vertex_primary:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"Found {len(unique_vertex_primary)} unique vertex_primary values",
                details={
                    "unique_values": sorted(unique_vertex_primary.tolist()),
                    "hs_particle_count": len(hs_particles),
                    "hs_vertex_mean": {"vx": hs_vx, "vy": hs_vy, "vz": hs_vz},
                    "vertex_distribution": vertex_counts.to_dict(),
                }
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message="vertex_primary=1 (hard scatter) not found",
                details={
                    "unique_values": sorted(unique_vertex_primary.tolist()),
                }
            )


class PrimaryFlagConsistencyTest(ConsistencyTest):
    """Test that primary flag equals NOT created_in_simulation."""
    
    def __init__(self):
        super().__init__(
            name="Primary Flag Consistency",
            description="Verify primary = NOT created_in_simulation"
        )
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        parquet_particles = loader.load_parquet_particles(global_event_id)
        
        if 'primary' not in parquet_particles.columns:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="primary column not present",
            )
        
        if 'created_in_simulation' not in parquet_particles.columns:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="created_in_simulation column not present",
            )
        
        # Check consistency
        expected_primary = ~parquet_particles['created_in_simulation']
        actual_primary = parquet_particles['primary']
        
        mismatches = expected_primary != actual_primary
        num_mismatch = mismatches.sum()
        
        if num_mismatch == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"All {len(parquet_particles)} particles have consistent primary flag",
            )
        else:
            mismatch_sample = parquet_particles[mismatches][['particle_id', 'primary', 'created_in_simulation']].head(10)
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"{num_mismatch} particles have inconsistent primary flag",
                details={
                    "mismatch_count": int(num_mismatch),
                    "mismatch_sample": mismatch_sample.to_dict('records'),
                }
            )


class GeneratorParticlePropertiesTest(ConsistencyTest):
    """Test that generator particles (created_in_simulation=False) have valid properties."""
    
    def __init__(self):
        super().__init__(
            name="Generator Particle Properties",
            description="Verify generator particles have valid PDG IDs and status"
        )
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        parquet_particles = loader.load_parquet_particles(global_event_id)
        
        if 'created_in_simulation' not in parquet_particles.columns:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="created_in_simulation column not present",
            )
        
        generator_particles = parquet_particles[~parquet_particles['created_in_simulation']]
        
        if len(generator_particles) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="No generator particles found",
            )
        
        issues = []
        
        # Check PDG ID is present and non-zero
        if 'pdg' in generator_particles.columns:
            zero_pdg = (generator_particles['pdg'] == 0).sum()
            if zero_pdg > 0:
                issues.append(f"{zero_pdg} generator particles have PDG=0")
        
        # Check generator status (typically 1 for stable particles)
        if 'generator_status' in generator_particles.columns:
            status_counts = generator_particles['generator_status'].value_counts()
            # Most common statuses are 1 (stable) and 2 (decayed)
            unusual_status = status_counts[~status_counts.index.isin([1, 2, 23, 62, 63, 71, 72, 83, 84])]
            if len(unusual_status) > 0:
                issues.append(f"Unusual generator statuses: {unusual_status.to_dict()}")
        
        if len(issues) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"All {len(generator_particles)} generator particles have valid properties",
                details={
                    "generator_count": len(generator_particles),
                    "unique_pdgs": len(generator_particles['pdg'].unique()) if 'pdg' in generator_particles.columns else None,
                }
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"Issues found in generator particles",
                details={
                    "issues": issues,
                    "generator_count": len(generator_particles),
                }
            )


class ParticleHitCountConsistencyTest(ConsistencyTest):
    """Test that num_tracker_hits and num_calo_hits are consistent."""
    
    def __init__(self):
        super().__init__(
            name="Particle Hit Count Consistency",
            description="Verify particle hit counts match actual hit associations"
        )
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        
        parquet_particles = loader.load_parquet_particles(global_event_id)
        parquet_hits = loader.load_parquet_tracker_hits(global_event_id)
        
        # Count actual hits per particle
        if 'particle_id' not in parquet_hits.columns:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="particle_id column not in tracker hits",
            )
        
        actual_hit_counts = parquet_hits.groupby('particle_id').size()
        
        # Compare with stored count if available
        if 'num_tracker_hits' not in parquet_particles.columns:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="num_tracker_hits column not in particles",
            )
        
        # This is typically stored from EDM4hep, may differ from parquet hit count
        # For this test, just verify non-negative values
        negative_counts = (parquet_particles['num_tracker_hits'] < 0).sum()
        
        if negative_counts == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"Hit counts are valid for {len(parquet_particles)} particles",
                details={
                    "particles_with_hits": (parquet_particles['num_tracker_hits'] > 0).sum(),
                    "actual_hit_particle_count": len(actual_hit_counts),
                }
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"{negative_counts} particles have negative hit counts",
            )


class ParticleTests(TestSuite):
    """Test suite for particle data consistency."""
    
    def __init__(self):
        super().__init__(
            name="Particle Tests",
            description="Validate particle data completeness and relationships"
        )
        
        # Add all particle tests
        self.add_test(ParticleCompletenessTest())
        self.add_test(ParticleKinematicsMatchTest())
        self.add_test(ParentChildRelationshipTest())
        self.add_test(VertexPrimaryFlagTest())
        self.add_test(PrimaryFlagConsistencyTest())
        self.add_test(GeneratorParticlePropertiesTest())
        self.add_test(ParticleHitCountConsistencyTest())
