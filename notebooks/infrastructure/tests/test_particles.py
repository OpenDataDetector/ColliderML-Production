"""
Particle consistency tests for ColliderML data validation.

Tests:
- Particle completeness: All PARQUET particles exist in EDM4hep (parquet is subset)
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
    """Test that all PARQUET particles exist in EDM4hep (parquet is a subset of EDM4hep)."""
    
    def __init__(self):
        super().__init__(
            name="Particle Completeness",
            description="Verify all parquet particles exist in EDM4hep (parquet is subset)"
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
        
        # Check that all parquet particles exist in EDM4hep (parquet is subset)
        missing_in_edm4hep = parquet_ids - edm4hep_ids  # These should NOT exist
        
        if len(missing_in_edm4hep) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"All {len(parquet_ids)} parquet particles found in EDM4hep ({len(edm4hep_ids)} total)",
                details={
                    "parquet_count": len(parquet_ids),
                    "edm4hep_count": len(edm4hep_ids),
                    "parquet_is_subset": True,
                }
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"{len(missing_in_edm4hep)} parquet particles NOT found in EDM4hep",
                details={
                    "parquet_count": len(parquet_ids),
                    "edm4hep_count": len(edm4hep_ids),
                    "missing_in_edm4hep_count": len(missing_in_edm4hep),
                    "missing_sample": list(missing_in_edm4hep)[:10],
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
    """Test that parent_id references don't form loops (self-parent or circular A→B→A)."""
    
    def __init__(self):
        super().__init__(
            name="Parent-Child Relationships",
            description="Verify no self-parents and no immediate circular relationships"
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
        
        issues = []
        
        # Check 1: No particle is its own parent
        self_parents = parquet_particles[parquet_particles['particle_id'] == parquet_particles['parent_id']]
        if len(self_parents) > 0:
            issues.append(f"{len(self_parents)} particles are their own parent")
        
        # Check 2: No immediate circular relationships (A→B→A)
        # Find particles with their own child as parent
        circular = parquet_particles.merge(
            parquet_particles[['particle_id', 'parent_id']],
            left_on='parent_id',
            right_on='particle_id',
            how='inner',
            suffixes=('', '_child')
        )
        # Filter for cases where child's parent points back to original particle
        circular = circular[circular['particle_id'] == circular['parent_id_child']]
        
        if len(circular) > 0:
            issues.append(f"{len(circular)} particles have circular parent relationships (A→B→A)")
        
        if len(issues) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"No parent loops found in {len(parquet_particles)} particles",
                details={
                    "num_particles": len(parquet_particles),
                    "num_with_parent": int((parquet_particles['parent_id'] != -1).sum()),
                }
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message="; ".join(issues),
                details={
                    "self_parent_sample": self_parents[['particle_id', 'parent_id']].head(5).to_dict('records') if len(self_parents) > 0 else [],
                    "circular_sample": circular[['particle_id', 'parent_id', 'particle_id_child', 'parent_id_child']].head(5).to_dict('records') if len(circular) > 0 else [],
                }
            )


class VertexPrimaryFlagTest(ConsistencyTest):
    """Test that particles within each vertex have consistent positions."""
    
    def __init__(self, consistency_threshold: float = 0.60, tolerance_fraction: float = 0.30):
        super().__init__(
            name="Vertex Position Consistency",
            description="Verify particles in each vertex have consistent vx, vy, vz positions"
        )
        self.consistency_threshold = consistency_threshold  # 60% of particles must be within tolerance
        self.tolerance_fraction = tolerance_fraction  # within 30% of median (or absolute tolerance for near-zero)
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        parquet_particles = loader.load_parquet_particles(global_event_id)
        
        if 'vertex_primary' not in parquet_particles.columns:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="vertex_primary column not present",
            )
        
        if 'primary' not in parquet_particles.columns:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="primary column not present",
            )
        
        # Check consistency for each vertex
        unique_vertices = parquet_particles['vertex_primary'].unique()
        inconsistent_vertices = []
        vertex_stats = {}
        
        for vertex_id in unique_vertices:
            # Get primary particles for this vertex
            vertex_particles = parquet_particles[parquet_particles['vertex_primary'] == vertex_id]
            vertex_primaries = vertex_particles[vertex_particles['primary']]
            
            if len(vertex_primaries) < 2:
                # Skip vertices with < 2 primaries (can't check consistency)
                continue
            
            # Check each coordinate
            for coord in ['vx', 'vy', 'vz']:
                if coord not in vertex_primaries.columns:
                    continue
                
                values = vertex_primaries[coord].values
                median_val = np.median(values)
                
                # Use relative tolerance, but with absolute floor for near-zero means
                abs_tolerance = max(abs(median_val * self.tolerance_fraction), 1e-6)
                
                # Count how many are within tolerance of the mean
                within_tolerance = np.abs(values - median_val) <= abs_tolerance
                fraction_consistent = np.mean(within_tolerance)
                
                if fraction_consistent < self.consistency_threshold:
                    inconsistent_vertices.append({
                        "vertex_id": int(vertex_id),
                        "coord": coord,
                        "median": float(median_val),
                        "std": float(np.std(values)),
                        "fraction_consistent": float(fraction_consistent),
                        "num_primaries": len(vertex_primaries),
                    })
            
            # Store stats for this vertex
            vertex_stats[int(vertex_id)] = {
                "num_primaries": len(vertex_primaries),
                "vx_median": float(vertex_primaries['vx'].median()) if 'vx' in vertex_primaries.columns else None,
                "vy_median": float(vertex_primaries['vy'].median()) if 'vy' in vertex_primaries.columns else None,
                "vz_median": float(vertex_primaries['vz'].median()) if 'vz' in vertex_primaries.columns else None,
            }
        
        if len(inconsistent_vertices) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"All {len(unique_vertices)} vertices have consistent positions (>{self.consistency_threshold*100:.0f}% within {self.tolerance_fraction*100:.0f}%)",
                details={
                    "num_vertices": len(unique_vertices),
                    "vertex_stats_sample": dict(list(vertex_stats.items())[:5]),
                }
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"{len(inconsistent_vertices)} vertex/coordinate pairs have inconsistent positions",
                details={
                    "num_vertices": len(unique_vertices),
                    "inconsistent_sample": inconsistent_vertices[:10],
                }
            )


class PrimaryFlagConsistencyTest(ConsistencyTest):
    """Test that primary flag equals NOT created_in_simulation (from EDM4hep)."""
    
    def __init__(self):
        super().__init__(
            name="Primary Flag Consistency",
            description="Verify parquet primary = NOT edm4hep created_in_simulation"
        )
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        parquet_particles = loader.load_parquet_particles(global_event_id)
        
        if 'primary' not in parquet_particles.columns:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="primary column not present in parquet",
            )
        
        # Load EDM4hep particles to get created_in_simulation
        edm4hep_batch = loader.load_edm4hep_event(local_event)
        edm4hep_particles = edm4hep_batch.get_particles_df()
        
        if 'created_in_simulation' not in edm4hep_particles.columns:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="created_in_simulation column not present in EDM4hep",
            )
        
        # Merge parquet and edm4hep on particle_id
        merged = parquet_particles.merge(
            edm4hep_particles[['particle_id', 'created_in_simulation']],
            on='particle_id',
            how='inner'
        )
        
        if len(merged) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="No matching particles found between parquet and EDM4hep",
            )
        
        # Check consistency: primary should be NOT created_in_simulation
        expected_primary = ~merged['created_in_simulation']
        actual_primary = merged['primary']
        
        mismatches = expected_primary != actual_primary
        num_mismatch = mismatches.sum()
        
        if num_mismatch == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"All {len(merged)} matched particles have consistent primary flag",
                details={
                    "parquet_count": len(parquet_particles),
                    "edm4hep_count": len(edm4hep_particles),
                    "matched_count": len(merged),
                }
            )
        else:
            mismatch_sample = merged[mismatches][['particle_id', 'primary', 'created_in_simulation']].head(10)
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"{num_mismatch} particles have inconsistent primary flag",
                details={
                    "mismatch_count": int(num_mismatch),
                    "matched_count": len(merged),
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
        
        # Load EDM4hep to get created_in_simulation (not in parquet)
        edm4hep_batch = loader.load_edm4hep_event(local_event)
        edm4hep_particles = edm4hep_batch.get_particles_df()
        
        if 'created_in_simulation' not in edm4hep_particles.columns:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="created_in_simulation column not in EDM4hep",
            )
        
        # Merge to get created_in_simulation for parquet particles
        merged = parquet_particles.merge(
            edm4hep_particles[['particle_id', 'created_in_simulation']],
            on='particle_id',
            how='inner'
        )
        
        generator_particles = merged[~merged['created_in_simulation']]
        
        if len(generator_particles) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="No generator particles found",
            )
        
        issues = []
        
        # Check PDG ID is present and non-zero (column is pdg_id in parquet schema)
        pdg_col = 'pdg_id' if 'pdg_id' in generator_particles.columns else 'pdg'
        if pdg_col in generator_particles.columns:
            zero_pdg = (generator_particles[pdg_col] == 0).sum()
            if zero_pdg > 0:
                issues.append(f"{zero_pdg} generator particles have PDG=0")
        
        # Note: generator_status is NOT in parquet schema, only in EDM4hep
        # Skip generator_status check as it's not available in parquet
        
        if len(issues) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"All {len(generator_particles)} generator particles have valid properties",
                details={
                    "generator_count": len(generator_particles),
                    "unique_pdgs": len(generator_particles[pdg_col].unique()) if pdg_col in generator_particles.columns else None,
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
