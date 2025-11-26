"""
HepMC validation tests for ColliderML data validation.

Tests:
- Hard scatter matching: vertex_primary=1 particles match HepMC HS file
- Generator particle provenance: All generator particles traceable to HepMC
- Vertex smearing: Check Gaussian parameters (σ_xy=0.0125mm, σ_z=55.5mm, σ_t=0.185ns)
- Event number mapping: EDM4hep event headers correctly map to HepMC events
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from scipy import stats

from .test_base import (
    ConsistencyTest,
    TestResult,
    TestStatus,
    TestSuite,
    DataLoader,
)


class HardScatterMatchTest(ConsistencyTest):
    """Test that vertex_primary=1 particles match HepMC hard scatter file."""
    
    def __init__(self, match_threshold: float = 0.9):
        super().__init__(
            name="Hard Scatter Particle Match",
            description="Verify vertex_primary=1 generator particles match HepMC HS"
        )
        self.match_threshold = match_threshold
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        
        # Load parquet particles
        parquet_particles = loader.load_parquet_particles(global_event_id)
        
        # Load EDM4hep for merging
        edm4hep_batch = loader.load_edm4hep_event(local_event)
        edm4hep_particles = edm4hep_batch.get_particles_df()
        
        # Merge parquet and EDM4hep
        merged = parquet_particles.merge(
            edm4hep_particles,
            on="particle_id",
            how="inner",
            suffixes=("_parquet", "_edm4hep")
        )
        
        # Filter for hard scatter generator particles
        if 'vertex_primary' not in merged.columns:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="vertex_primary column not present",
            )
        
        hs_gen = merged[(merged['vertex_primary'] == 1) & (~merged['created_in_simulation'])]
        
        if len(hs_gen) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="No hard scatter generator particles found",
            )
        
        # Load HepMC hard scatter events
        hepmc_events = loader.load_hepmc_hs_events()
        event_number_mapping = loader.load_event_number_mapping()
        
        hepmc_event_num = event_number_mapping[local_event]
        
        if hepmc_event_num not in hepmc_events:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"HepMC event {hepmc_event_num} not found",
            )
        
        hepmc_evt = hepmc_events[hepmc_event_num]
        
        # Get HepMC particle momenta
        hepmc_momentum = pd.DataFrame({
            "px": hepmc_evt.numpy.particles.px,
            "py": hepmc_evt.numpy.particles.py,
            "pz": hepmc_evt.numpy.particles.pz,
        }).astype(np.float32).drop_duplicates()
        
        # Get parquet HS generator particle momenta
        px_col = "px_parquet" if "px_parquet" in hs_gen.columns else "px"
        py_col = "py_parquet" if "py_parquet" in hs_gen.columns else "py"
        pz_col = "pz_parquet" if "pz_parquet" in hs_gen.columns else "pz"
        
        parquet_momentum = hs_gen[[px_col, py_col, pz_col]].astype(np.float32)
        parquet_momentum.columns = ["px", "py", "pz"]
        parquet_momentum = parquet_momentum.drop_duplicates()
        
        # Match on momentum (float32 precision)
        matched = parquet_momentum.merge(hepmc_momentum, on=["px", "py", "pz"], how="inner")
        
        match_rate = len(matched) / len(parquet_momentum) if len(parquet_momentum) > 0 else 0
        
        details = {
            "parquet_hs_gen_count": len(parquet_momentum),
            "hepmc_particle_count": len(hepmc_momentum),
            "matched_count": len(matched),
            "match_rate": float(match_rate),
            "hepmc_event_number": hepmc_event_num,
        }
        
        if match_rate >= self.match_threshold:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"{len(matched)}/{len(parquet_momentum)} HS particles match ({match_rate*100:.1f}%)",
                details=details,
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"Only {len(matched)}/{len(parquet_momentum)} HS particles match ({match_rate*100:.1f}%)",
                details=details,
            )


class EventNumberMappingTest(ConsistencyTest):
    """Test that event number mapping is consistent."""
    
    def __init__(self):
        super().__init__(
            name="Event Number Mapping",
            description="Verify EDM4hep event headers map to valid HepMC events"
        )
    
    def run(self, loader: DataLoader, **kwargs) -> TestResult:
        event_number_mapping = loader.load_event_number_mapping()
        hepmc_events = loader.load_hepmc_hs_events()
        
        missing_events = []
        for local_idx, hepmc_num in enumerate(event_number_mapping):
            if hepmc_num not in hepmc_events:
                missing_events.append((local_idx, hepmc_num))
        
        details = {
            "total_events": len(event_number_mapping),
            "hepmc_events_available": len(hepmc_events),
            "missing_count": len(missing_events),
        }
        
        if len(missing_events) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"All {len(event_number_mapping)} events map to valid HepMC events",
                details=details,
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"{len(missing_events)} events have no HepMC match",
                details={**details, "missing_sample": missing_events[:10]},
            )


class VertexSmearingXYTest(ConsistencyTest):
    """Test that vertex x,y smearing matches expected Gaussian (σ_xy=0.0125mm)."""
    
    def __init__(self, expected_sigma: float = 0.0125, tolerance: float = 0.5):
        super().__init__(
            name="Vertex Smearing XY",
            description=f"Verify vertex x,y smearing ~ N(0, {expected_sigma}mm)"
        )
        self.expected_sigma = expected_sigma
        self.tolerance = tolerance  # Relative tolerance for sigma comparison
    
    def run(self, loader: DataLoader, **kwargs) -> TestResult:
        # Load all particles for multiple events to get statistics
        all_particles = []
        
        for local_event in range(min(loader.run_size, 20)):  # Sample up to 20 events
            global_event_id = loader.run_id * loader.run_size + local_event
            try:
                particles = loader.load_parquet_particles(global_event_id)
                if 'vertex_primary' in particles.columns:
                    # Get hard scatter particles at vertex_primary=1
                    hs = particles[particles['vertex_primary'] == 1]
                    all_particles.append(hs)
            except Exception:
                continue
        
        if len(all_particles) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="Could not load particle data",
            )
        
        combined = pd.concat(all_particles, ignore_index=True)
        
        if 'vx' not in combined.columns or 'vy' not in combined.columns:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="vx/vy columns not present",
            )
        
        # Get vertex positions (per vertex_primary=1, should be roughly consistent per event)
        # Group by event and take first vertex position as representative
        if 'event_id' in combined.columns:
            vertex_positions = combined.groupby('event_id')[['vx', 'vy']].first()
        else:
            vertex_positions = combined[['vx', 'vy']]
        
        # Compute spread
        vx_std = vertex_positions['vx'].std()
        vy_std = vertex_positions['vy'].std()
        
        # Combined xy spread
        vxy_std = np.sqrt((vx_std**2 + vy_std**2) / 2)
        
        relative_error = abs(vxy_std - self.expected_sigma) / self.expected_sigma
        
        details = {
            "vx_std_mm": float(vx_std),
            "vy_std_mm": float(vy_std),
            "combined_xy_std_mm": float(vxy_std),
            "expected_sigma_mm": self.expected_sigma,
            "relative_error": float(relative_error),
            "num_events_sampled": len(vertex_positions),
        }
        
        if relative_error < self.tolerance:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"Vertex xy smearing σ={vxy_std:.4f}mm (expected {self.expected_sigma}mm)",
                details=details,
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"Vertex xy smearing σ={vxy_std:.4f}mm differs from expected {self.expected_sigma}mm",
                details=details,
            )


class VertexSmearingZTest(ConsistencyTest):
    """Test that vertex z smearing matches expected Gaussian (σ_z=55.5mm)."""
    
    def __init__(self, expected_sigma: float = 55.5, tolerance: float = 0.5):
        super().__init__(
            name="Vertex Smearing Z",
            description=f"Verify vertex z smearing ~ N(0, {expected_sigma}mm)"
        )
        self.expected_sigma = expected_sigma
        self.tolerance = tolerance
    
    def run(self, loader: DataLoader, **kwargs) -> TestResult:
        all_particles = []
        
        for local_event in range(min(loader.run_size, 20)):
            global_event_id = loader.run_id * loader.run_size + local_event
            try:
                particles = loader.load_parquet_particles(global_event_id)
                if 'vertex_primary' in particles.columns:
                    hs = particles[particles['vertex_primary'] == 1]
                    all_particles.append(hs)
            except Exception:
                continue
        
        if len(all_particles) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="Could not load particle data",
            )
        
        combined = pd.concat(all_particles, ignore_index=True)
        
        if 'vz' not in combined.columns:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="vz column not present",
            )
        
        if 'event_id' in combined.columns:
            vertex_positions = combined.groupby('event_id')['vz'].first()
        else:
            vertex_positions = combined['vz']
        
        vz_std = vertex_positions.std()
        
        relative_error = abs(vz_std - self.expected_sigma) / self.expected_sigma
        
        details = {
            "vz_std_mm": float(vz_std),
            "expected_sigma_mm": self.expected_sigma,
            "relative_error": float(relative_error),
            "num_events_sampled": len(vertex_positions),
        }
        
        if relative_error < self.tolerance:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"Vertex z smearing σ={vz_std:.2f}mm (expected {self.expected_sigma}mm)",
                details=details,
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"Vertex z smearing σ={vz_std:.2f}mm differs from expected {self.expected_sigma}mm",
                details=details,
            )


class GeneratorParticleCountTest(ConsistencyTest):
    """Test that generator particle counts are reasonable."""
    
    def __init__(self, min_gen_particles: int = 10, max_gen_particles: int = 5000):
        super().__init__(
            name="Generator Particle Count",
            description="Verify generator particle counts are reasonable"
        )
        self.min_particles = min_gen_particles
        self.max_particles = max_gen_particles
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        parquet_particles = loader.load_parquet_particles(global_event_id)
        
        if 'created_in_simulation' not in parquet_particles.columns:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="created_in_simulation column not present",
            )
        
        gen_particles = parquet_particles[~parquet_particles['created_in_simulation']]
        count = len(gen_particles)
        
        details = {
            "generator_particle_count": count,
            "total_particles": len(parquet_particles),
            "min_expected": self.min_particles,
            "max_expected": self.max_particles,
        }
        
        if self.min_particles <= count <= self.max_particles:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"{count} generator particles in expected range",
                details=details,
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"Generator particle count {count} outside range [{self.min_particles}, {self.max_particles}]",
                details=details,
            )


class HardScatterVertexIsOneTest(ConsistencyTest):
    """Test that vertex_primary=1 consistently represents the hard scatter."""
    
    def __init__(self):
        super().__init__(
            name="Hard Scatter Is Vertex 1",
            description="Verify vertex_primary=1 is consistently the hard scatter vertex"
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
        
        if 'created_in_simulation' not in parquet_particles.columns:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="created_in_simulation column not present",
            )
        
        # Get particles at vertex_primary=1
        v1_particles = parquet_particles[parquet_particles['vertex_primary'] == 1]
        
        # All should be generator particles (not created_in_simulation)
        v1_gen = v1_particles[~v1_particles['created_in_simulation']]
        
        # Check vertex distribution of generator particles
        gen_particles = parquet_particles[~parquet_particles['created_in_simulation']]
        gen_vertex_dist = gen_particles['vertex_primary'].value_counts()
        
        # Hard scatter should have the most generator particles at vertex 1
        most_common_vertex = gen_vertex_dist.idxmax() if len(gen_vertex_dist) > 0 else None
        
        details = {
            "vertex_1_total": len(v1_particles),
            "vertex_1_gen": len(v1_gen),
            "gen_vertex_distribution": gen_vertex_dist.to_dict(),
            "most_common_gen_vertex": int(most_common_vertex) if most_common_vertex is not None else None,
        }
        
        if most_common_vertex == 1 and len(v1_gen) > 0:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"Vertex 1 has {len(v1_gen)} generator particles (most of any vertex)",
                details=details,
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"Vertex 1 is not the primary hard scatter vertex (most common: {most_common_vertex})",
                details=details,
            )


class HepMCValidationTests(TestSuite):
    """Test suite for HepMC provenance validation."""
    
    def __init__(self):
        super().__init__(
            name="HepMC Validation Tests",
            description="Validate generator particle provenance and vertex smearing"
        )
        
        # Add all HepMC validation tests
        self.add_test(EventNumberMappingTest())
        self.add_test(HardScatterVertexIsOneTest())
        self.add_test(HardScatterMatchTest())
        self.add_test(GeneratorParticleCountTest())
        self.add_test(VertexSmearingXYTest())
        self.add_test(VertexSmearingZTest())
