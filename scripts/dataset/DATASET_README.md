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

For the recommended way to download and load this data (with control over how many events are downloaded), see the **ColliderML documentation**: [https://opendatadetector.github.io/ColliderML/](https://opendatadetector.github.io/ColliderML/).

You can use either **(a)** the ColliderML library (recommended) or **(b)** the Hugging Face `datasets` library with streaming.

### Option (a): ColliderML library (recommended)

**Install and download**

```bash
pip install colliderml
colliderml download --channels ttbar --pileup pu0 --objects particles,tracker_hits,calo_hits,tracks --max-events 200
```

Adjust `--channels` and `--pileup` for your config (e.g. `ggf`, `pu200`); see the [ColliderML docs](https://opendatadetector.github.io/ColliderML/) for options.

**Load in Python**

```python
from colliderml.core import load_tables, collect_tables

cfg = {
    "dataset_id": "CERN/ColliderML-Release-1",
    "channels": "ttbar",
    "pileup": "pu0",
    "objects": ["particles", "tracker_hits", "calo_hits", "tracks"],
    "split": "train",
    "lazy": False,
    "max_events": 200,
}
tables = load_tables(cfg)
frames = collect_tables(tables)  # dict[str, pl.DataFrame]
# e.g. frames["particles"], frames["tracker_hits"] — one row per event, list columns
```

For exploding event tables into flat (object-per-row) tables, pileup subsampling, and calibration, see the [library docs](https://opendatadetector.github.io/ColliderML/library/overview.html) and the [exploration notebook](https://github.com/OpenDataDetector/ColliderML/blob/main/notebooks/colliderml_loader_exploration.ipynb).

### Option (b): Hugging Face `datasets` with streaming

To iterate over events without downloading the full split, use `streaming=True`. **Without** `streaming=True`, `load_dataset` downloads all files for the chosen config.

```python
from datasets import load_dataset

# Stream first 100 events (only fetches data as you iterate)
ds = load_dataset("CERN/ColliderML-Release-1", "ttbar_pu0_particles", split="train", streaming=True)
for i, event in enumerate(ds):
    if i >= 100:
        break
    # use event (e.g. event["event_id"], event["px"], ...)
```

### Selecting columns and working with tables

With the ColliderML workflow, select columns via Polars, e.g. `frames["particles"].select(["event_id", "px", "py", "pz"])`. For more (exploding, calibration), see the [library docs](https://opendatadetector.github.io/ColliderML/library/overview.html) and [exploration notebook](https://github.com/OpenDataDetector/ColliderML/blob/main/notebooks/colliderml_loader_exploration.ipynb).

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
