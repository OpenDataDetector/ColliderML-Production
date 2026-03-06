---
license: {{ license }}
task_categories:
- other
tags:
{% for tag in tags %}
- {{ tag }}
{% endfor %}
pretty_name: {{ pretty_name }}
size_categories:
- {{ size_category }}
configs:
{% for config_name, urls in data_files.items() %}
- config_name: {{ config_name }}
  data_files:
{% for url in urls %}
  - "{{ url }}"
{% endfor %}
{% endfor %}
---

# ColliderML: {{ pretty_name }}

## Dataset Description

This dataset contains simulated high-energy physics collision events for {{ process_description_long }} generated using the **Open Data Detector (ODD)** geometry within the **Key4hep** and **ACTS (A Common Tracking Software)** frameworks, representing a generic collider detector similar to those at the HL-LHC.

### Dataset Summary

- **Campaign**: `{{ campaign }}`
- **Process**: {{ process_description }}
- **Version**: `{{ version }}`
- **Number of Events**: ~{{ total_events }} events
- **Pileup**: {{ pileup }} {% if pileup == 0 %}(no additional interactions){% endif %}
- **Detector**: Open Data Detector (ODD)
- **Format**: Apache Parquet with list columns for variable-length data
- **License**: {{ license }}

### Supported Tasks

This dataset is designed for machine learning tasks in high-energy physics, including:

- **Particle tracking**: Reconstruct charged particle trajectories from detector hits
- **Track-to-particle matching**: Associate reconstructed tracks with truth particles
- **Jet tagging**: Identify jets originating from top quarks, b-quarks, or light quarks
- **Energy reconstruction**: Predict particle energies from calorimeter deposits
- **Physics analysis**: Event classification (signal vs. background discrimination)
- **Representation learning**: Study hierarchical information at different detector levels

### Languages

N/A (Physics data)

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
    "dataset_id": "{{ repo_id }}",
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
ds = load_dataset("{{ repo_id }}", "particles", split="train", streaming=True)
for i, event in enumerate(ds):
    if i >= 100:
        break
    # use event (e.g. event["event_id"], event["px"], ...)
```

### Selecting columns and working with tables

With the ColliderML workflow, select columns via Polars, e.g. `frames["particles"].select(["event_id", "px", "py", "pz"])`. For more (exploding, calibration), see the [library docs](https://opendatadetector.github.io/ColliderML/library/overview.html) and [exploration notebook](https://github.com/OpenDataDetector/ColliderML/blob/main/notebooks/colliderml_loader_exploration.ipynb).

## Dataset Structure

### Data Instances

Each row in the Parquet files represents a single collision event. Variable-length quantities (e.g., lists of particles, hits, tracks) are stored as Parquet list columns.

Example event structure:

```python
{
    'event_id': 42,
    'particle_id': [0, 1, 2, 3, ...],  # List of particle IDs
    'pdg_id': [11, -11, 211, ...],     # Particle type codes
    'px': [1.2, -0.5, 3.4, ...],       # Momentum components (GeV)
    'py': [0.8, 1.1, -0.3, ...],
    'pz': [5.2, -2.1, 10.5, ...],
    'energy': [5.5, 2.3, 11.2, ...],
    # ... additional fields
}
```

### Data Fields

The dataset contains {{ num_configs }} data types organized by detector hierarchy:

{% if 'particles' in schemas %}
#### 1. `particles` (Truth-level)

Truth information about generated particles before detector simulation.

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | int64 | Unique event identifier |
| `particle_id` | list<int64> | Unique particle ID within event |
| `pdg_id` | list<int64> | PDG particle code (e.g., 11=electron, 13=muon, 211=pion) |
| `mass` | list<float64> | Particle rest mass (GeV/c²) |
| `energy` | list<float64> | Particle total energy (GeV) |
| `charge` | list<float64> | Electric charge (in units of e) |
| `px`, `py`, `pz` | list<float64> | Momentum components (GeV/c) |
| `vx`, `vy`, `vz` | list<float64> | Vertex position (mm) |
| `time` | list<float64> | Production time (ns) |
| `num_tracker_hits` | list<int64> | Number of hits in tracker |
| `num_calo_hits` | list<int64> | Number of hits in calorimeter |
| `vertex_primary` | list<int64> | Primary vertex flag (1 = hard scatter, 2,...,N = pileup) |
| `parent_id` | list<float64> | ID of parent particle |

**Typical event**: ~200-500 particles per event
{% endif %}

{% if 'tracker_hits' in schemas %}
#### 2. `tracker_hits` (Detector-level)

Digitized spatial measurements from the tracking detector (silicon sensors).

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | int64 | Unique event identifier |
| `x`, `y`, `z` | list<float64> | Measured hit position (mm) |
| `true_x`, `true_y`, `true_z` | list<float64> | True (simulated) hit position before digitization (mm) |
| `time` | list<float64> | Hit time (ns) |
| `particle_id` | list<int64> | Truth particle that created this hit |
| `volume_id` | list<int64> | Detector volume identifier |
| `layer_id` | list<int64> | Detector layer number |
| `surface_id` | list<int64> | Sensor surface identifier |
| `cell_id` | list<int64> | Cell/pixel identifier |
| `detector` | list<int64> | Detector subsystem code |

**Typical event**: ~2,000-5,000 hits per event
{% endif %}

{% if 'calo_hits' in schemas %}
#### 3. `calo_hits` (Calorimeter-level)

Energy deposits in the calorimeter system (electromagnetic + hadronic).

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | int64 | Unique event identifier |
| `detector` | list<string> | Calorimeter subsystem name |
| `cell_id` | list<string> | Calorimeter cell identifier |
| `total_energy` | list<float64> | Total energy deposited in cell (GeV) |
| `x`, `y`, `z` | list<float64> | Cell center position (mm) |
| `contrib_particle_ids` | list<list<int64>> | IDs of particles contributing to this cell |
| `contrib_energies` | list<list<float64>> | Energy contribution from each particle (GeV) |
| `contrib_times` | list<list<float64>> | Time of each contribution (ns) |

**Note**: Nested lists for contributions (one cell can have multiple particle deposits).

**Typical event**: ~500-1,000 calorimeter cells with deposits
{% endif %}

{% if 'tracks' in schemas %}
#### 4. `tracks` (Reconstruction-level)

Reconstructed particle tracks from pattern recognition and track fitting algorithms.

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | int64 | Unique event identifier |
| `track_id` | list<int64> | Unique track identifier within event |
| `majority_particle_id` | list<int64> | Truth particle with most hits on this track |
| `d0` | list<float64> | Transverse impact parameter (mm) |
| `z0` | list<float64> | Longitudinal impact parameter (mm) |
| `phi` | list<float64> | Azimuthal angle (radians) |
| `theta` | list<float64> | Polar angle (radians) |
| `qop` | list<float64> | Charge divided by momentum (e/GeV) |
| `hit_ids` | list<list<int32>> | List of tracker hit IDs assigned to this track |

**Track parameters**: Standard ACTS track representation (perigee parameters at origin).

**Derived quantities**:
- Transverse momentum: `pt = abs(1/qop) * sin(theta)`
- Pseudorapidity: `eta = -ln(tan(theta/2))`
- Total momentum: `p = abs(1/qop)`

**Typical event**: ~50-150 reconstructed tracks per event
{% endif %}

### Data Splits

Currently, the dataset does not have predefined train/validation/test splits. Users should implement their own splitting strategy based on their use case. Recommended approach:

```python
from sklearn.model_selection import train_test_split

# Example: 70% train, 15% validation, 15% test
all_events = list(range({{ total_events }}))
train_val, test = train_test_split(all_events, test_size=0.15, random_state=42)
train, val = train_test_split(train_val, test_size=0.176, random_state=42)  # 0.176 * 0.85 ≈ 0.15
```

### Support

For questions, issues, or feature requests:
- Email: {{ contact }}
- You can also open a discussion in the HuggingFace community panel for this dataset.

### Acknowledgments

This work was supported by:
- NERSC computing resources
- U.S. Department of Energy, Office of Science
- Danish Data Science Academy (DDSA)

---

**Last updated**: {{ date }}
**Dataset version**: {{ version }}
