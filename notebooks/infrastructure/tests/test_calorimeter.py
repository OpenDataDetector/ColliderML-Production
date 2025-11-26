"""
Calorimeter consistency tests for ColliderML data validation.

Tests:
- Energy thresholds: ECal ≥ 5e-5 GeV, HCal ≥ 2.5e-4 GeV
- Position matching: Calo hit positions match EDM4hep
- Contribution consistency: Contribution energies sum correctly
- Contribution particle validity: All contrib particle_ids are valid
- Timing filters: Corrected time values are reasonable
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple

from .test_base import (
    ConsistencyTest,
    TestResult,
    TestStatus,
    TestSuite,
    DataLoader,
)


class CaloHitPositionMatchTest(ConsistencyTest):
    """Test that calo hit positions match between parquet and EDM4hep."""
    
    def __init__(self):
        super().__init__(
            name="Calo Hit Position Match",
            description="Verify calo hit positions match EDM4hep"
        )
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        
        # Load parquet calo hits
        parquet_cells, _ = loader.load_parquet_calo_hits(global_event_id)
        
        # Load EDM4hep calo hits
        edm4hep_batch = loader.load_edm4hep_event(local_event)
        edm4hep_calo = edm4hep_batch.get_calo_hits_df()
        
        if len(parquet_cells) == 0 or len(edm4hep_calo) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="No calo hits found in one or both sources",
            )
        
        # Merge on position
        parquet_pos = parquet_cells[["x", "y", "z"]].astype(np.float32)
        edm4hep_pos = edm4hep_calo[["x", "y", "z"]].astype(np.float32)
        
        merged = parquet_pos.merge(edm4hep_pos, on=["x", "y", "z"], how="inner")
        
        # Note: Parquet may have fewer hits due to energy thresholds
        match_rate = len(merged) / len(parquet_cells) if len(parquet_cells) > 0 else 0
        
        details = {
            "matched_hits": len(merged),
            "parquet_hits": len(parquet_cells),
            "edm4hep_hits": len(edm4hep_calo),
            "match_rate": float(match_rate),
        }
        
        if match_rate >= 0.99:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"{len(merged)}/{len(parquet_cells)} parquet hits found in EDM4hep ({match_rate*100:.1f}%)",
                details=details,
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"Only {len(merged)}/{len(parquet_cells)} parquet hits found in EDM4hep ({match_rate*100:.1f}%)",
                details=details,
            )


class CaloHitEnergyThresholdTest(ConsistencyTest):
    """Test that energy thresholds are correctly applied."""
    
    def __init__(self, ecal_threshold: float = 5e-5, hcal_threshold: float = 2.5e-4):
        super().__init__(
            name="Calo Hit Energy Thresholds",
            description="Verify ECal ≥ 5e-5 GeV and HCal ≥ 2.5e-4 GeV thresholds"
        )
        self.ecal_threshold = ecal_threshold
        self.hcal_threshold = hcal_threshold
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        parquet_cells, _ = loader.load_parquet_calo_hits(global_event_id)
        
        if len(parquet_cells) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="No calo hits found",
            )
        
        if 'energy' not in parquet_cells.columns:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="energy column not present",
            )
        
        issues = []
        details = {}
        
        # Check by detector type if available
        if 'system' in parquet_cells.columns or 'detector' in parquet_cells.columns:
            det_col = 'system' if 'system' in parquet_cells.columns else 'detector'
            
            # ECal (system typically 1 or contains 'ECal')
            ecal_mask = parquet_cells[det_col].astype(str).str.contains('ECal|1', case=False, regex=True)
            hcal_mask = parquet_cells[det_col].astype(str).str.contains('HCal|2', case=False, regex=True)
            
            if ecal_mask.any():
                ecal_hits = parquet_cells[ecal_mask]
                below_threshold = (ecal_hits['energy'] < self.ecal_threshold).sum()
                details['ecal_count'] = len(ecal_hits)
                details['ecal_below_threshold'] = int(below_threshold)
                if below_threshold > 0:
                    issues.append(f"{below_threshold} ECal hits below {self.ecal_threshold} GeV threshold")
            
            if hcal_mask.any():
                hcal_hits = parquet_cells[hcal_mask]
                below_threshold = (hcal_hits['energy'] < self.hcal_threshold).sum()
                details['hcal_count'] = len(hcal_hits)
                details['hcal_below_threshold'] = int(below_threshold)
                if below_threshold > 0:
                    issues.append(f"{below_threshold} HCal hits below {self.hcal_threshold} GeV threshold")
        else:
            # No detector info, just check overall minimum
            min_energy = parquet_cells['energy'].min()
            details['min_energy'] = float(min_energy)
            details['total_hits'] = len(parquet_cells)
            # Use the more lenient ECal threshold
            if min_energy < self.ecal_threshold:
                issues.append(f"Minimum energy {min_energy:.2e} GeV is below ECal threshold {self.ecal_threshold}")
        
        if len(issues) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"All calo hits meet energy thresholds",
                details=details,
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"Energy threshold violations found",
                details={**details, "issues": issues},
            )


class CaloContributionSumTest(ConsistencyTest):
    """Test that contribution energies sum to cell energy."""
    
    def __init__(self, tolerance: float = 0.01):
        super().__init__(
            name="Calo Contribution Energy Sum",
            description="Verify contribution energies sum to cell energy"
        )
        self.tolerance = tolerance
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        parquet_cells, parquet_contribs = loader.load_parquet_calo_hits(global_event_id)
        
        if len(parquet_cells) == 0 or len(parquet_contribs) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="No calo hits or contributions found",
            )
        
        if 'energy' not in parquet_cells.columns:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="energy column not in cells",
            )
        
        contrib_energy_col = None
        for col in ['energies', 'energy']:
            if col in parquet_contribs.columns:
                contrib_energy_col = col
                break
        
        if contrib_energy_col is None:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="energy column not in contributions",
            )
        
        # Sum contributions per cell
        contrib_sums = parquet_contribs.groupby('cell_index')[contrib_energy_col].sum()
        
        # Compare with cell energies
        mismatches = []
        for idx in range(len(parquet_cells)):
            cell_energy = parquet_cells.iloc[idx]['energy']
            contrib_sum = contrib_sums.get(idx, 0)
            
            rel_diff = abs(cell_energy - contrib_sum) / max(cell_energy, 1e-10)
            if rel_diff > self.tolerance:
                mismatches.append({
                    'cell_index': idx,
                    'cell_energy': float(cell_energy),
                    'contrib_sum': float(contrib_sum),
                    'rel_diff': float(rel_diff),
                })
        
        details = {
            "cells_checked": len(parquet_cells),
            "mismatch_count": len(mismatches),
        }
        
        if len(mismatches) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"All {len(parquet_cells)} cells have correct contribution sums",
                details=details,
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"{len(mismatches)} cells have mismatched contribution sums",
                details={**details, "mismatch_sample": mismatches[:10]},
            )


class CaloContributionParticleValidityTest(ConsistencyTest):
    """Test that contribution particle_ids reference valid particles."""
    
    def __init__(self):
        super().__init__(
            name="Calo Contribution Particle Validity",
            description="Verify all contribution particle_ids exist in particles"
        )
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        
        _, parquet_contribs = loader.load_parquet_calo_hits(global_event_id)
        parquet_particles = loader.load_parquet_particles(global_event_id)
        
        if len(parquet_contribs) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="No contributions found",
            )
        
        particle_id_col = None
        for col in ['particle_ids', 'particle_id']:
            if col in parquet_contribs.columns:
                particle_id_col = col
                break
        
        if particle_id_col is None:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="particle_id column not in contributions",
            )
        
        contrib_particle_ids = set(parquet_contribs[particle_id_col].unique())
        valid_particle_ids = set(parquet_particles['particle_id'].unique())
        
        # particle_id=0 typically means "no associated particle"
        contrib_particle_ids.discard(0)
        
        invalid_ids = contrib_particle_ids - valid_particle_ids
        
        if len(invalid_ids) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"All {len(contrib_particle_ids)} contribution particle_ids are valid",
                details={
                    "unique_contrib_particles": len(contrib_particle_ids),
                    "total_contributions": len(parquet_contribs),
                }
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"{len(invalid_ids)} contribution particle_ids not found in particles",
                details={
                    "invalid_count": len(invalid_ids),
                    "invalid_sample": list(invalid_ids)[:10],
                }
            )


class CaloTimingFilterTest(ConsistencyTest):
    """Test that calo hit timing is reasonable after correction."""
    
    def __init__(self, max_time_ns: float = 100.0):
        super().__init__(
            name="Calo Timing Filter",
            description="Verify corrected calo hit timing is reasonable"
        )
        self.max_time_ns = max_time_ns
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        parquet_cells, _ = loader.load_parquet_calo_hits(global_event_id)
        
        time_col = None
        for col in ['time', 'corrected_time']:
            if col in parquet_cells.columns:
                time_col = col
                break
        
        if time_col is None:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="time column not present",
            )
        
        times = parquet_cells[time_col]
        
        # Check for unreasonable times (negative or very large)
        negative_times = (times < 0).sum()
        large_times = (times > self.max_time_ns).sum()
        
        details = {
            "mean_time_ns": float(times.mean()),
            "std_time_ns": float(times.std()),
            "min_time_ns": float(times.min()),
            "max_time_ns": float(times.max()),
            "negative_times": int(negative_times),
            "large_times": int(large_times),
        }
        
        issues = []
        if negative_times > 0:
            issues.append(f"{negative_times} hits with negative time")
        if large_times > len(times) * 0.01:  # More than 1% with large time
            issues.append(f"{large_times} hits with time > {self.max_time_ns} ns")
        
        if len(issues) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"Timing values reasonable (mean: {times.mean():.2f} ns)",
                details=details,
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"Timing issues found",
                details={**details, "issues": issues},
            )


class CaloHitCountTest(ConsistencyTest):
    """Test that calo hit counts are reasonable."""
    
    def __init__(self, min_hits: int = 10, max_hits: int = 50000):
        super().__init__(
            name="Calo Hit Count",
            description="Verify calo hit counts are reasonable"
        )
        self.min_hits = min_hits
        self.max_hits = max_hits
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        parquet_cells, _ = loader.load_parquet_calo_hits(global_event_id)
        
        count = len(parquet_cells)
        
        details = {
            "calo_hit_count": count,
            "min_expected": self.min_hits,
            "max_expected": self.max_hits,
        }
        
        if self.min_hits <= count <= self.max_hits:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"{count} calo hits in expected range [{self.min_hits}, {self.max_hits}]",
                details=details,
            )
        elif count < self.min_hits:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"Too few calo hits: {count} < {self.min_hits}",
                details=details,
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"Too many calo hits: {count} > {self.max_hits}",
                details=details,
            )


class CaloContributionCountTest(ConsistencyTest):
    """Test that contribution counts per cell are reasonable."""
    
    def __init__(self, max_contribs_per_cell: int = 100):
        super().__init__(
            name="Calo Contribution Count Per Cell",
            description="Verify contribution counts per cell are reasonable"
        )
        self.max_contribs = max_contribs_per_cell
    
    def run(self, loader: DataLoader, local_event: int = 0, **kwargs) -> TestResult:
        global_event_id = loader.run_id * loader.run_size + local_event
        parquet_cells, parquet_contribs = loader.load_parquet_calo_hits(global_event_id)
        
        if len(parquet_contribs) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message="No contributions found",
            )
        
        # Count contributions per cell
        contribs_per_cell = parquet_contribs.groupby('cell_index').size()
        
        excessive = contribs_per_cell[contribs_per_cell > self.max_contribs]
        
        details = {
            "mean_contribs_per_cell": float(contribs_per_cell.mean()),
            "max_contribs_per_cell": int(contribs_per_cell.max()),
            "cells_with_excessive": len(excessive),
            "total_cells": len(parquet_cells),
        }
        
        if len(excessive) == 0:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message=f"All cells have ≤{self.max_contribs} contributions (mean: {contribs_per_cell.mean():.1f})",
                details=details,
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"{len(excessive)} cells have >{self.max_contribs} contributions",
                details=details,
            )


class CalorimeterTests(TestSuite):
    """Test suite for calorimeter data consistency."""
    
    def __init__(self):
        super().__init__(
            name="Calorimeter Tests",
            description="Validate calorimeter hit data and contributions"
        )
        
        # Add all calorimeter tests
        self.add_test(CaloHitPositionMatchTest())
        self.add_test(CaloHitEnergyThresholdTest())
        self.add_test(CaloHitCountTest())
        self.add_test(CaloTimingFilterTest())
        self.add_test(CaloContributionSumTest())
        self.add_test(CaloContributionParticleValidityTest())
        self.add_test(CaloContributionCountTest())
