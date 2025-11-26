"""
ColliderML Data Consistency Test Suite

This package provides comprehensive closure tests for validating the ColliderML
parquet output files against source EDM4hep, HepMC, and ACTS ROOT files.

Test Categories:
- ParticleTests: Validate particle data completeness and relationships
- TrackerHitTests: Validate tracker hit positions and associations
- TrackTests: Validate reconstructed track data and truth matching
- CalorimeterTests: Validate calorimeter hit data and contributions
- HepMCValidationTests: Validate generator particle provenance
- CrossObjectTests: Validate consistency across different object types
"""

from .test_base import (
    TestResult,
    TestStatus,
    TestSuite,
    ConsistencyTest,
    DataLoader,
)
from .test_particles import ParticleTests
from .test_tracker_hits import TrackerHitTests
from .test_tracks import TrackTests
from .test_calorimeter import CalorimeterTests
from .test_hepmc import HepMCValidationTests
from .test_cross_object import CrossObjectTests
from .run_all_tests import (
    run_all_tests,
    get_all_test_suites,
    get_test_summary_df,
    get_failed_tests,
    generate_test_report,
    run_tests_multiple_events,
)

__all__ = [
    "TestResult",
    "TestStatus",
    "TestSuite",
    "ConsistencyTest",
    "DataLoader",
    "ParticleTests",
    "TrackerHitTests",
    "TrackTests",
    "CalorimeterTests",
    "HepMCValidationTests",
    "CrossObjectTests",
    "run_all_tests",
    "get_all_test_suites",
    "get_test_summary_df",
    "get_failed_tests",
    "generate_test_report",
    "run_tests_multiple_events",
]
