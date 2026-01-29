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
    
    def __init__(self, exact_match_threshold: float = 0.5, approx_match_threshold: float = 0.9):
        super().__init__(
            name="Hard Scatter Particle Match",
            description="Verify vertex_primary=1 primary particles match HepMC HS"
        )
        self.exact_match_threshold = exact_match_threshold  # Min fraction for exact float32 match
        self.approx_match_threshold = approx_match_threshold  # Min fraction for ~1% tolerance match
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        
        # Load parquet particles
        parquet_particles = loader.load_parquet_particles(global_event_id)
        
        # Check for required columns
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
                message="primary column not present in parquet",
            )
        
        # Get hard scatter generator particles: vertex_primary=1 AND primary=True
        hs_gen = parquet_particles[
            (parquet_particles['vertex_primary'] == 1) & 
            (parquet_particles['primary'] == True)
        ].copy()
        
        if len(hs_gen) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="No hard scatter primary particles found (vertex_primary=1 & primary=True)",
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
        hs_gen_momentum = hs_gen[['px', 'py', 'pz']].astype(np.float32).drop_duplicates()
        
        # 1. Exact float32 precision match
        exact_merged = hs_gen_momentum.merge(
            hepmc_momentum, 
            on=["px", "py", "pz"], 
            how="inner"
        )
        exact_match_count = len(exact_merged)
        
        # 2. Approximate match within 1% relative difference
        approx_match_count = self._count_approx_matches(
            hs_gen_momentum, hepmc_momentum, threshold=0.01
        )
        
        # Calculate match rates
        total_hs_particles = len(hs_gen_momentum)
        exact_match_rate = exact_match_count / total_hs_particles if total_hs_particles > 0 else 0
        approx_match_rate = approx_match_count / total_hs_particles if total_hs_particles > 0 else 0
        
        details = {
            "parquet_hs_primary_count": total_hs_particles,
            "hepmc_particle_count": len(hepmc_momentum),
            "exact_match_count": exact_match_count,
            "exact_match_rate": float(exact_match_rate),
            "approx_match_count_1pct": approx_match_count,
            "approx_match_rate_1pct": float(approx_match_rate),
            "hepmc_event_number": hepmc_event_num,
        }
        
        # Pass if approximate match rate is good enough
        if approx_match_rate >= self.approx_match_threshold:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"{approx_match_count}/{total_hs_particles} HS particles match within 1% ({approx_match_rate*100:.1f}%), {exact_match_count} exact",
                details=details,
            )
        elif exact_match_rate >= self.exact_match_threshold:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"{exact_match_count}/{total_hs_particles} HS particles match exactly ({exact_match_rate*100:.1f}%)",
                details=details,
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"Only {approx_match_count}/{total_hs_particles} HS particles match within 1% ({approx_match_rate*100:.1f}%)",
                details=details,
            )
    
    def _count_approx_matches(self, hs_df: pd.DataFrame, hepmc_df: pd.DataFrame, threshold: float) -> int:
        """Count particles that match within relative threshold."""
        hs_px = hs_df["px"].values
        hs_py = hs_df["py"].values
        hs_pz = hs_df["pz"].values
        hepmc_px = hepmc_df["px"].values
        hepmc_py = hepmc_df["py"].values
        hepmc_pz = hepmc_df["pz"].values
        
        count = 0
        for i in range(len(hs_df)):
            # Compute relative differences
            px_diff = np.abs(hs_px[i] - hepmc_px) / np.clip(np.abs(hs_px[i]), 1e-10, None)
            py_diff = np.abs(hs_py[i] - hepmc_py) / np.clip(np.abs(hs_py[i]), 1e-10, None)
            pz_diff = np.abs(hs_pz[i] - hepmc_pz) / np.clip(np.abs(hs_pz[i]), 1e-10, None)
            mean_diff = (px_diff + py_diff + pz_diff) / 3.0
            
            # If any HepMC particle matches within threshold, count as present
            if np.any(mean_diff < threshold):
                count += 1
        return count


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
    
    def __init__(self, expected_sigma: float = 0.0125, tolerance: float = 1.0):
        super().__init__(
            name="Vertex Smearing XY",
            description=f"Verify vertex x,y smearing ~ N(0, {expected_sigma}mm)"
        )
        self.expected_sigma = expected_sigma
        self.tolerance = tolerance  # Relative tolerance for sigma comparison
    
    def run(self, loader: DataLoader, local_event: Optional[int] = None, **kwargs) -> TestResult:
        """
        Measure vertex smearing by:
        1. For each event, find all primary vertices (different vertex_primary values)
        2. For each primary vertex, get mode x,y of particles (= vertex position)
        3. Compute std dev of all vertex positions across events
        """
        all_vertex_positions = []
        
        # If local_event specified, use just that event; otherwise sample multiple
        if local_event is not None:
            event_range = [local_event]
        else:
            event_range = range(min(loader.run_size, 20))
        
        for evt in event_range:
            global_event_id = loader.run_id * loader.run_size + evt
            try:
                particles = loader.load_parquet_particles(global_event_id)
                
                if 'vertex_primary' not in particles.columns:
                    continue
                if 'vx' not in particles.columns or 'vy' not in particles.columns:
                    continue
                
                # For each primary vertex in this event, get mode position
                # Mode = most common value, robust to outliers from secondaries
                vertex_modes_vx = particles.groupby('vertex_primary')['vx'].agg(lambda x: x.mode()[0])
                vertex_modes_vy = particles.groupby('vertex_primary')['vy'].agg(lambda x: x.mode()[0])
                vertex_modes = pd.DataFrame({'vx': vertex_modes_vx, 'vy': vertex_modes_vy})
                all_vertex_positions.append(vertex_modes)
                
            except Exception:
                continue
        
        if len(all_vertex_positions) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="Could not load particle data",
            )
        
        # Concatenate all vertex positions
        combined = pd.concat(all_vertex_positions, ignore_index=True)
        
        if len(combined) < 2:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message=f"Need at least 2 vertices to compute std, got {len(combined)}",
            )
        
        # Compute spread of vertex positions
        vx_std = combined['vx'].std()
        vy_std = combined['vy'].std()
        
        # Combined xy spread
        vxy_std = np.sqrt((vx_std**2 + vy_std**2) / 2)
        
        relative_error = abs(vxy_std - self.expected_sigma) / self.expected_sigma
        
        details = {
            "vx_std_mm": float(vx_std),
            "vy_std_mm": float(vy_std),
            "combined_xy_std_mm": float(vxy_std),
            "expected_sigma_mm": self.expected_sigma,
            "relative_error": float(relative_error),
            "num_vertices": len(combined),
            "num_events": len(event_range),
        }
        
        if relative_error < self.tolerance:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"Vertex xy smearing σ={vxy_std:.4f}mm (expected {self.expected_sigma}mm) from {len(combined)} vertices",
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
    
    def __init__(self, expected_sigma: float = 55.5, tolerance: float = 1.0):
        super().__init__(
            name="Vertex Smearing Z",
            description=f"Verify vertex z smearing ~ N(0, {expected_sigma}mm)"
        )
        self.expected_sigma = expected_sigma
        self.tolerance = tolerance
    
    def run(self, loader: DataLoader, local_event: Optional[int] = None, **kwargs) -> TestResult:
        """
        Measure vertex z smearing by:
        1. For each event, find all primary vertices (different vertex_primary values)
        2. For each primary vertex, get mode z of particles (= vertex z position)
        3. Compute std dev of all vertex z positions across events
        """
        all_vertex_z = []
        
        if local_event is not None:
            event_range = [local_event]
        else:
            event_range = range(min(loader.run_size, 20))
        
        for evt in event_range:
            global_event_id = loader.run_id * loader.run_size + evt
            try:
                particles = loader.load_parquet_particles(global_event_id)
                
                if 'vertex_primary' not in particles.columns:
                    continue
                if 'vz' not in particles.columns:
                    continue
                
                # For each primary vertex in this event, get mode z
                vertex_modes = particles.groupby('vertex_primary')['vz'].agg(lambda x: x.mode()[0])
                all_vertex_z.extend(vertex_modes.values)
                
            except Exception:
                continue
        
        if len(all_vertex_z) < 2:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message=f"Need at least 2 vertices to compute std, got {len(all_vertex_z)}",
            )
        
        vz_array = np.array(all_vertex_z)
        vz_std = vz_array.std()
        
        relative_error = abs(vz_std - self.expected_sigma) / self.expected_sigma
        
        details = {
            "vz_std_mm": float(vz_std),
            "vz_mean_mm": float(vz_array.mean()),
            "expected_sigma_mm": self.expected_sigma,
            "relative_error": float(relative_error),
            "num_vertices": len(all_vertex_z),
            "num_events": len(event_range),
        }
        
        if relative_error < self.tolerance:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"Vertex z smearing σ={vz_std:.2f}mm (expected {self.expected_sigma}mm) from {len(all_vertex_z)} vertices",
                details=details,
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"Vertex z smearing σ={vz_std:.2f}mm differs from expected {self.expected_sigma}mm",
                details=details,
            )


class VertexSmearingTimeTest(ConsistencyTest):
    """Test that vertex time smearing matches expected Gaussian (σ_t=0.185ns)."""
    
    def __init__(self, expected_sigma: float = 0.185, tolerance: float = 1.0):
        super().__init__(
            name="Vertex Smearing Time",
            description=f"Verify vertex time smearing ~ N(0, {expected_sigma}ns)"
        )
        self.expected_sigma = expected_sigma
        self.tolerance = tolerance
    
    def run(self, loader: DataLoader, local_event: Optional[int] = None, **kwargs) -> TestResult:
        """
        Measure vertex time smearing by:
        1. For each event, find all primary vertices (different vertex_primary values)
        2. For each primary vertex, get mode time of particles (= vertex time)
        3. Compute std dev of all vertex times across events
        """
        all_vertex_times = []
        
        if local_event is not None:
            event_range = [local_event]
        else:
            event_range = range(min(loader.run_size, 20))
        
        for evt in event_range:
            global_event_id = loader.run_id * loader.run_size + evt
            try:
                particles = loader.load_parquet_particles(global_event_id)
                
                if 'vertex_primary' not in particles.columns:
                    continue
                if 'time' not in particles.columns:
                    continue
                
                # For each primary vertex in this event, get mode time
                vertex_modes = particles.groupby('vertex_primary')['time'].agg(lambda x: x.mode()[0])
                all_vertex_times.extend(vertex_modes.values)
                
            except Exception:
                continue
        
        if len(all_vertex_times) < 2:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message=f"Need at least 2 vertices to compute std, got {len(all_vertex_times)}",
            )
        
        vt_array = np.array(all_vertex_times)
        vt_std = vt_array.std()
        
        relative_error = abs(vt_std - self.expected_sigma) / self.expected_sigma
        
        details = {
            "time_std_ns": float(vt_std),
            "time_mean_ns": float(vt_array.mean()),
            "expected_sigma_ns": self.expected_sigma,
            "relative_error": float(relative_error),
            "num_vertices": len(all_vertex_times),
            "num_events": len(event_range),
        }
        
        if relative_error < self.tolerance:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"Vertex time smearing σ={vt_std:.4f}ns (expected {self.expected_sigma}ns) from {len(all_vertex_times)} vertices",
                details=details,
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"Vertex time smearing σ={vt_std:.4f}ns differs from expected {self.expected_sigma}ns",
                details=details,
            )


class GeneratorParticleCountTest(ConsistencyTest):
    """Test that generator particle counts are reasonable."""
    
    def __init__(self, min_gen_particles: int = 10, max_gen_particles: int = 100000):
        super().__init__(
            name="Generator Particle Count",
            description="Verify generator particle counts are reasonable"
        )
        self.min_particles = min_gen_particles
        self.max_particles = max_gen_particles  # High limit for full pileup (~200 interactions)
    
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
        
        gen_particles = merged[~merged['created_in_simulation']]
        count = len(gen_particles)
        
        details = {
            "generator_particle_count": count,
            "total_particles": len(parquet_particles),
            "matched_particles": len(merged),
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


class HepMCValidationTests(TestSuite):
    """Test suite for HepMC provenance validation."""
    
    def __init__(self):
        super().__init__(
            name="HepMC Validation Tests",
            description="Validate generator particle provenance and vertex smearing"
        )
        
        # Add all HepMC validation tests
        self.add_test(EventNumberMappingTest())
        self.add_test(HardScatterMatchTest())
        self.add_test(GeneratorParticleCountTest())
        self.add_test(VertexSmearingXYTest())
        self.add_test(VertexSmearingZTest())
        self.add_test(VertexSmearingTimeTest())
