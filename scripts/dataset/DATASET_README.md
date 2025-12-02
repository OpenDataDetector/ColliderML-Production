---
license: cc-by-4.0
task_categories:
- other
tags:
- physics
- high-energy-physics
- particle-physics
- collider-physics
- tracking
- calorimetry
- machine-learning
- simulation
- particle-tracking
- jet-tagging
pretty_name: ColliderML Dataset Release 1
size_categories:
- 100K<n<1M
---

# ColliderML: Dataset Release 1

## Dataset Description

This dataset contains simulated high-energy physics collision events generated using the **Open Data Detector (ODD)** geometry within the **Key4hep** and **ACTS (A Common Tracking Software)** frameworks, representing a generic collider detector similar to those at the HL-LHC.

### Dataset Summary

- **Collision Energy**: 14 TeV (proton-proton)
- **Detector**: Open Data Detector (ODD)
- **Simulation**: DD4hep + Geant4 + ACTS
- **Format**: Apache Parquet with list columns for variable-length data
- **License**: CC-BY-4.0

### Available Configurations

The dataset is organized into multiple configurations, each representing a combination of:
- **Physics process** (e.g., ttbar, ggf, dihiggs)
- **Pileup condition** (pu0 = no pileup, pu200 = HL-LHC pileup)
- **Object type** (particles, tracker_hits, calo_hits, tracks)

### Supported Tasks

This dataset is designed for machine learning tasks in high-energy physics, including:

- **Particle tracking**: Reconstruct charged particle trajectories from detector hits
- **Track-to-particle matching**: Associate reconstructed tracks with truth particles
- **Jet tagging**: Identify jets originating from top quarks, b-quarks, or light quarks
- **Energy reconstruction**: Predict particle energies from calorimeter deposits
- **Physics analysis**: Event classification (signal vs. background discrimination)
- **Representation learning**: Study hierarchical information at different detector levels

## Quick Start

### Installation

```bash
pip install datasets pyarrow
```

### Load a Configuration

```python
from datasets import load_dataset

# Load truth particles from ttbar (no pileup)
particles = load_dataset(
    "OpenDataDetector/ColliderML-Release-1",
    "ttbar_pu0_particles",
    split="train"
)

print(f"Loaded {len(particles)} events")
print(f"Columns: {particles.column_names}")
```

### Load First 100 Events with Specific Columns

```python
from datasets import load_dataset
import numpy as np

# Load only specific columns
particles = load_dataset(
    "OpenDataDetector/ColliderML-Release-1",
    "ttbar_pu0_particles",
    split="train[:100]",
    columns=["event_id", "px", "py", "pz", "energy", "pdg_id"]
)

# Process events
for event in particles:
    px = np.array(event['px'])
    py = np.array(event['py'])
    pt = np.sqrt(px**2 + py**2)
    print(f"Event {event['event_id']}: {len(px)} particles, mean pT = {pt.mean():.2f} GeV")
```

## Dataset Structure

### Data Instances

Each row represents a single collision event. Variable-length quantities (particles, hits, tracks) are stored as Parquet list columns.

Example event structure:
```python
{
    'event_id': 42,
    'particle_id': [0, 1, 2, 3, ...],
    'pdg_id': [11, -11, 211, ...],
    'px': [1.2, -0.5, 3.4, ...],
    'py': [0.8, 1.1, -0.3, ...],
    'pz': [5.2, -2.1, 10.5, ...],
    'energy': [5.5, 2.3, 11.2, ...],
    # ... additional fields
}
```

### Data Fields by Object Type

#### 1. `particles` (Truth-level)

Truth information about generated particles before detector simulation.

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | uint32 | Unique event identifier |
| `particle_id` | list\<uint64\> | Unique particle ID within event |
| `pdg_id` | list\<int64\> | PDG particle code (11=electron, 13=muon, 211=pion, etc.) |
| `mass` | list\<float32\> | Particle rest mass (GeV/c²) |
| `energy` | list\<float32\> | Particle total energy (GeV) |
| `charge` | list\<float32\> | Electric charge (units of e) |
| `px`, `py`, `pz` | list\<float32\> | Momentum components (GeV/c) |
| `vx`, `vy`, `vz` | list\<float32\> | Vertex position (mm) |
| `time` | list\<float32\> | Production time (ns) |
| `perigee_d0` | list\<float32\> | Perigee transverse impact parameter (mm) |
| `perigee_z0` | list\<float32\> | Perigee longitudinal impact parameter (mm) |
| `num_tracker_hits` | list\<uint16\> | Number of hits in tracker |
| `num_calo_hits` | list\<uint16\> | Number of hits in calorimeter |
| `primary` | list\<bool\> | Whether particle is primary |
| `vertex_primary` | list\<uint16\> | Primary vertex index (1=hard scatter) |
| `parent_id` | list\<int64\> | ID of parent particle (-1 if none) |

#### 2. `tracker_hits` (Detector-level)

Digitized spatial measurements from the tracking detector (silicon sensors).

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | uint32 | Unique event identifier |
| `x`, `y`, `z` | list\<float32\> | Measured hit position (mm) |
| `true_x`, `true_y`, `true_z` | list\<float32\> | True hit position before digitization (mm) |
| `time` | list\<float32\> | Hit time (ns) |
| `particle_id` | list\<uint64\> | Truth particle that created this hit |
| `volume_id` | list\<uint8\> | Detector volume identifier |
| `layer_id` | list\<uint16\> | Detector layer number |
| `surface_id` | list\<uint32\> | Sensor surface identifier |
| `detector` | list\<uint8\> | Detector subsystem code |

#### 3. `calo_hits` (Calorimeter-level)

Energy deposits in the calorimeter system (electromagnetic + hadronic).

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | uint32 | Unique event identifier |
| `detector` | list\<uint8\> | Calorimeter subsystem code |
| `total_energy` | list\<float32\> | Total energy deposited in cell (GeV) |
| `x`, `y`, `z` | list\<float32\> | Cell center position (mm) |
| `contrib_particle_ids` | list\<list\<uint64\>\> | IDs of particles contributing to this cell |
| `contrib_energies` | list\<list\<float32\>\> | Energy contribution from each particle (GeV) |
| `contrib_times` | list\<list\<float32\>\> | Time of each contribution (ns) |

#### 4. `tracks` (Reconstruction-level)

Reconstructed particle tracks from ACTS pattern recognition and track fitting.

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | uint32 | Unique event identifier |
| `track_id` | list\<uint16\> | Unique track identifier within event |
| `majority_particle_id` | list\<uint64\> | Truth particle with most hits on this track |
| `d0` | list\<float32\> | Transverse impact parameter (mm) |
| `z0` | list\<float32\> | Longitudinal impact parameter (mm) |
| `phi` | list\<float32\> | Azimuthal angle (radians) |
| `theta` | list\<float32\> | Polar angle (radians) |
| `qop` | list\<float32\> | Charge divided by momentum (e/GeV) |
| `hit_ids` | list\<list\<uint32\>\> | List of tracker hit IDs on this track |

**Derived quantities for tracks:**
- Transverse momentum: `pt = abs(1/qop) * sin(theta)`
- Pseudorapidity: `eta = -ln(tan(theta/2))`
- Total momentum: `p = abs(1/qop)`

## Dataset Creation

### Simulation Chain

1. **Event Generation**: MadGraph5 + Pythia8 for hard scatter and parton shower
2. **Detector Simulation**: Geant4 via DD4hep with the Open Data Detector geometry
3. **Digitization**: Realistic detector response simulation
4. **Reconstruction**: ACTS track finding and fitting algorithms
5. **Format Conversion**: EDM4HEP → Parquet using the ColliderML pipeline

### Software Stack

- **ACTS**: A Common Tracking Software - https://acts.readthedocs.io/
- **Open Data Detector**: https://github.com/acts-project/odd
- **Key4hep**: https://key4hep.github.io/
- **EDM4HEP**: https://edm4hep.web.cern.ch/

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{colliderml_release1_2025,
  title={{ColliderML Dataset Release 1}},
  author={{ColliderML Collaboration}},
  year={2025},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/OpenDataDetector/ColliderML-Release-1}},
  note={Simulation performed using ACTS and the Open Data Detector}
}
```

## Support

For questions, issues, or feature requests:
- **Email**: daniel.thomas.murnane@cern.ch
- **GitHub**: https://github.com/OpenDataDetector/ColliderML

## Acknowledgments

This work was supported by:
- NERSC computing resources (National Energy Research Scientific Computing Center)
- U.S. Department of Energy, Office of Science
- Danish Data Science Academy (DDSA)

---

**Release Version**: 1.0  
**Last Updated**: November 2025
