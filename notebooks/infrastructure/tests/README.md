# ColliderML Data Consistency Tests

A comprehensive test framework for validating ColliderML parquet datasets against their source EDM4hep ROOT files.

## Quick Start

```bash
# Activate the environment
conda activate collider-env

# Run all tests using a config file
python run_tests.py --config /path/to/test_config.yaml

# Run specific test suites
python run_tests.py --config /path/to/config.yaml --suite particles --suite hepmc

# List all available tests
python run_tests.py --list
```

## Configuration

Tests can be configured via YAML file or command-line arguments:

```yaml
# test_config.yaml
base_path: /path/to/simulation/data
run_id: 0
run_size: 64
chunk_size: 100
event: 0
```

| Argument | Description | Required |
|----------|-------------|----------|
| `--config` | Path to YAML config file | No |
| `--base-path` | Base path to simulation data | Yes* |
| `--run-id` | Run ID to test | Yes* |
| `--run-size` | Events per run | Yes* |
| `--chunk-size` | Parquet chunk size | Yes* |
| `--event` | Local event index (default: 0) | No |
| `--suite` | Specific suite(s) to run | No |
| `--json` | Output results as JSON | No |

*Required if not provided in config file

---

## Test Suites

### 📦 Particles (`test_particles.py`)

Validates particle data completeness and consistency between parquet and EDM4hep sources.

| Test | Description |
|------|-------------|
| **Particle Completeness** | All parquet particles exist in EDM4hep (parquet is a subset) |
| **Particle Kinematics Match** | Position (vx, vy, vz) and momentum (px, py, pz) match between sources |
| **Parent-Child Relationships** | All `parent_id` references point to valid particles |
| **Vertex Position Consistency** | Particles with same `vertex_primary` share same vertex position |
| **Primary Flag Consistency** | `primary = NOT created_in_simulation` (EDM4hep only) |
| **Generator Particle Properties** | Non-simulated particles have valid PDG codes and kinematics |
| **Particle Hit Count Consistency** | `num_tracker_hits` and `num_calo_hits` match actual hit counts |

---

### 📦 Tracker Hits (`test_tracker_hits.py`)

Validates tracker hit positions and particle associations.

| Test | Description |
|------|-------------|
| **Tracker Hit Position Match** | Hit positions (x, y, z) match between parquet and EDM4hep |
| **Tracker Hit Completeness** | All tracker hits in parquet exist in EDM4hep |
| **Tracker Hit Particle Association** | All `particle_id` values reference valid particles |
| **Tracker Hit Detector Encoding** | `volume_id`, `layer_id`, `module_id` are properly encoded |
| **Tracker Hit Reco Position** | Reconstructed positions are within detector bounds |
| **Tracker Hit Count Per Particle** | Hit counts per particle follow expected distributions |

---

### 📦 Tracks (`test_tracks.py`)

Validates reconstructed track data and truth matching.

| Test | Description |
|------|-------------|
| **Track ID Uniqueness** | All `track_id` values are unique within an event |
| **Track Hit ID Validity** | All hit IDs in tracks reference valid tracker hits |
| **Track Majority Particle Computation** | `majority_particle_id` matches recomputed majority |
| **Track Parameter Ranges** | Track parameters (θ, φ, qop, d0, z0) within physical bounds |
| **Track Hit Count** | Tracks have between 3-50 hits (reasonable range) |
| **Track Efficiency and Purity** | Efficiency and purity values between 0 and 1 |

---

### 📦 Calorimeter (`test_calorimeter.py`)

Validates calorimeter hit data and energy contributions.

| Test | Description |
|------|-------------|
| **Calo Hit Position Match** | Hit positions (x, y, z) match between parquet and EDM4hep |
| **Calo Hit Energy Thresholds** | `total_energy > 0` and within physical ranges |
| **Calo Hit Count** | Number of calo hits reasonable for event type (100-1M hits) |
| **Calo Timing Filter** | Hit times within -50 to 5000 ns window |
| **Calo Contribution Energy Sum** | Sum of `contrib_energies` ≈ `total_energy` (within 1%) |
| **Calo Contribution Particle Validity** | All `contrib_particle_ids` reference valid particles |
| **Calo Contribution Count Per Cell** | Contribution counts follow power-law distribution (log-log R² ≥ 0.85) |

---

### 📦 HepMC Validation (`test_hepmc.py`)

Validates generator particle provenance and vertex smearing.

| Test | Description |
|------|-------------|
| **Event Number Mapping** | All events map to valid HepMC event numbers |
| **Hard Scatter Particle Match** | Generator particles with `primary=True` match HepMC momenta (1% tolerance) |
| **Generator Particle Count** | Generator particle count in range 10-100,000 per event |
| **Vertex Smearing XY** | Primary vertex xy positions: σ ≈ 0.0125 mm |
| **Vertex Smearing Z** | Primary vertex z positions: σ ≈ 55.5 mm |
| **Vertex Smearing Time** | Primary vertex times: σ ≈ 0.185 ns |

---

### 📦 Cross-Object Consistency (`test_cross_object.py`)

Validates consistency across different object types.

| Test | Description |
|------|-------------|
| **Event ID Consistency** | Same `event_id` values across particles, hits, and tracks |
| **All Hit Particle IDs Valid** | All hit `particle_id` values reference existing particles |
| **Track Majority Particle IDs Valid** | All track `majority_particle_id` values are valid |
| **Particle-Hit Correspondence** | Particles with hits have corresponding hit records |
| **Object Count Reasonability** | Object counts follow expected relationships |
| **Track-Hit-Particle Consistency** | Track → Hit → Particle chain is internally consistent |

---

## Architecture

```
tests/
├── README.md           # This file
├── __init__.py         # Package init
├── run_tests.py        # CLI test runner
├── test_base.py        # Base classes: DataLoader, ConsistencyTest, TestSuite
├── test_particles.py   # Particle validation tests
├── test_tracker_hits.py # Tracker hit tests
├── test_tracks.py      # Track reconstruction tests
├── test_calorimeter.py # Calorimeter tests
├── test_hepmc.py       # Generator/HepMC validation
└── test_cross_object.py # Cross-object consistency
```

### Key Classes

- **`DataLoader`**: Loads parquet and ROOT data for a given run
- **`ConsistencyTest`**: Base class for individual tests
- **`TestSuite`**: Groups related tests together
- **`TestResult`**: Stores test outcome with status, message, and timing

### Test Status Values

| Status | Emoji | Description |
|--------|-------|-------------|
| `PASSED` | ✅ | Test passed all checks |
| `FAILED` | ❌ | Test found inconsistencies |
| `SKIPPED` | ⏭️ | Test skipped (missing data) |
| `ERROR` | 💥 | Test encountered an exception |

---

## Example Output

```
📄 Loaded config: /path/to/test_config.yaml

📂 Data path: /path/to/simulation/full_pileup/ttbar/v1
🔢 Run ID: 0, Event: 0
📊 Run size: 64, Chunk size: 100

🧪 Running: Particle Tests
   Validate particle data completeness and relationships
------------------------------------------------------------
================================================================================
Test Suite: particles
================================================================================
✅ Particle Completeness: All 847123 particles exist in EDM4hep
✅ Particle Kinematics Match: Position and momentum match within tolerance
✅ Parent-Child Relationships: All 523401 parent references valid
...

================================================================================
📊 OVERALL SUMMARY
================================================================================
   ✅ Passed:  38
   ❌ Failed:  0
   ⏭️  Skipped: 0
   💥 Errors:  0
   ⏱️  Total time: 45.23s
================================================================================
```

---

## Adding New Tests

1. Create a new test class inheriting from `ConsistencyTest`:

```python
from .test_base import ConsistencyTest, TestResult, TestStatus

class MyNewTest(ConsistencyTest):
    def __init__(self):
        super().__init__(
            name="My New Test",
            description="Description of what this test validates"
        )
    
    def run(self, loader, local_event: int = 0) -> TestResult:
        # Load data
        particles = loader.load_parquet_particles()
        
        # Perform validation
        if some_condition:
            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                message="All checks passed"
            )
        else:
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                message=f"Found {n} issues"
            )
```

2. Add the test to the appropriate `TestSuite` class.

---

## Data Schema Reference

### Particles Parquet
| Column | Type | Description |
|--------|------|-------------|
| `event_id` | int64 | Global event identifier |
| `particle_id` | int64 | Unique particle ID within event |
| `pdg_id` | int32 | PDG particle code |
| `mass`, `energy`, `charge` | float | Particle properties |
| `vx`, `vy`, `vz`, `time` | float | Production vertex |
| `px`, `py`, `pz` | float | Momentum components |
| `primary` | bool | True if generator particle |
| `vertex_primary` | int32 | Primary vertex index |
| `parent_id` | int64 | Parent particle ID (-1 if none) |
| `num_tracker_hits`, `num_calo_hits` | int32 | Hit counts |

### Vertex Smearing Parameters (HL-LHC)
| Parameter | Value | Description |
|-----------|-------|-------------|
| σ_xy | 0.0125 mm | Transverse beam spot |
| σ_z | 55.5 mm | Longitudinal spread |
| σ_t | 0.185 ns | Timing spread |

---

## License

Part of the ColliderML project. See the main repository for license information.
