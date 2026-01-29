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

This dataset contains simulated high-energy physics collision events for {{ process_description_long }} generated using the **Open Data Detector (ODD)** geometry within the **Key4hep** framework, representing a generic collider detector similar to those at the LHC.

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

### Installation

```bash
pip install datasets pyarrow
```

### Load First 100 Events (All Objects)

```python
from datasets import load_dataset

# Load first 100 rows of each configuration
particles = load_dataset("{{ repo_id }}", "particles", split="train[:100]")
tracker_hits = load_dataset("{{ repo_id }}", "tracker_hits", split="train[:100]")
calo_hits = load_dataset("{{ repo_id }}", "calo_hits", split="train[:100]")
tracks = load_dataset("{{ repo_id }}", "tracks", split="train[:100]")

print(f"Loaded {len(particles)} particle events")
print(f"Loaded {len(tracker_hits)} tracker hit events")
print(f"Loaded {len(calo_hits)} calo hit events")
print(f"Loaded {len(tracks)} track events")
```

### Load Specific Columns from First 100 Events

```python
from datasets import load_dataset
import numpy as np

# Load only specific columns from particles
particles = load_dataset(
    "{{ repo_id }}",
    "particles",
    split="train[:100]",
    columns=["event_id", "px", "py", "pz", "energy", "pdg_id"]
)

# Access data
for event in particles:
    event_id = event['event_id']

    # Convert to numpy arrays
    px = np.array(event['px'])
    py = np.array(event['py'])
    pz = np.array(event['pz'])

    # Calculate transverse momentum
    pt = np.sqrt(px**2 + py**2)

    print(f"Event {event_id}: {len(px)} particles, mean pt = {pt.mean():.2f} GeV")

# Load only specific columns from tracks
tracks = load_dataset(
    "{{ repo_id }}",
    "tracks",
    split="train[:100]",
    columns=["event_id", "qop", "theta", "phi"]
)

# Calculate derived quantities
for event in tracks:
    qop = np.array(event['qop'])
    theta = np.array(event['theta'])

    # Compute transverse momentum from track parameters
    pt = np.abs(1.0 / qop) * np.sin(theta)
    eta = -np.log(np.tan(theta / 2.0))

    print(f"Event {event['event_id']}: {len(qop)} tracks, pt range [{pt.min():.2f}, {pt.max():.2f}] GeV")
```

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
| `vertex_primary` | list<int64> | Primary vertex flag (1=hard scatter, 2, ..., N = pileup) |
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

## Dataset Creation

### Curation Rationale

This dataset was created to support machine learning research in high-energy physics, specifically for:

1. **Benchmarking tracking algorithms**: Compare traditional and ML-based track reconstruction methods
2. **Hierarchical representation learning**: Study information flow from detector hits → tracks → particles
3. **Physics analysis**: Develop ML models for event classification and particle identification
4. **Open science**: Provide publicly accessible, realistic detector simulation data

{{ curation_notes }}

### Source Data

#### Initial Data Collection and Normalization

The data is generated through the following simulation chain:

1. **Event Generation**: Events generated using a Monte Carlo event generator
2. **Detector Simulation**: Particle propagation through the Open Data Detector using ACTS
3. **Digitization**: Conversion of energy deposits to realistic detector signals
4. **Reconstruction**: Track finding and fitting using ACTS tracking algorithms
5. **Format Conversion**: EDM4HEP → Parquet using the ColliderML data pipeline

#### Who are the source data producers?

The data is produced by the **ColliderML collaboration** as part of the **ATLAS ITk ML Reconstruction** project at NERSC (National Energy Research Scientific Computing Center).

### Annotations

#### Annotation process

The dataset includes truth-level annotations automatically generated during the simulation:

- **Particle-level truth**: Generator-level particle information
- **Hit-to-particle associations**: Which particle created each detector hit
- **Track-to-particle matching**: `majority_particle_id` links reconstructed tracks to truth particles

These annotations enable supervised learning for tasks like:
- Track efficiency (did we reconstruct this particle?)
- Track purity (how many hits belong to the correct particle?)
- Fake rate (how many tracks are not matched to real particles?)

#### Who are the annotators?

N/A (Annotations are from simulation ground truth)

### Personal and Sensitive Information

This dataset contains only simulated physics data. No personal or sensitive information is included.

## Considerations for Using the Data

### Social Impact of Dataset

This dataset supports fundamental physics research and ML algorithm development. It has no direct social impact but contributes to:

- Open science and reproducible research
- Education in HEP and ML
- Development of algorithms that may have broader applications (e.g., pattern recognition, tracking in medical imaging)

### Discussion of Biases

As a simulated dataset, biases may arise from:

1. **Generator-level biases**: The event generator's modeling of the physics process
2. **Detector simulation biases**: Approximations in material interactions, detector response
3. **Reconstruction biases**: Algorithm choices in track finding and fitting
4. **Pileup modeling**: {% if pileup == 0 %}This dataset has no pileup; real LHC data has 20-60 simultaneous collisions{% else %}Pileup modeling may not perfectly match real data{% endif %}

Users should be aware that models trained on this data may not generalize to:
- Real detector data (requires calibration and alignment)
- Different detector geometries
- Different pileup conditions

### Other Known Limitations

- **Limited statistics**: ~{{ total_events }} events (consider data augmentation for large models)
- **Single physics process**: Only {{ process_description }}; does not include background processes
- **Idealized detector**: ODD is a generic detector, not an exact replica of ATLAS/CMS
- **Simplified simulation**: Some detector effects may be simplified

## Additional Information

### Dataset Curators

This dataset is maintained by the ColliderML team:

- Primary contact: {{ contact }}
- Collaboration: ATLAS ITk ML Reconstruction working group
- Infrastructure: NERSC (National Energy Research Scientific Computing Center)

### Licensing Information

This dataset is released under the **{{ license_name }}** license.

You are free to:
- **Share**: Copy and redistribute the material
- **Adapt**: Remix, transform, and build upon the material

Under the following terms:
- **Attribution**: You must give appropriate credit and indicate if changes were made

### Citation Information

If you use this dataset in your research, please cite:

```bibtex
@dataset{colliderml_{{ dataset }}_{{ version.replace('.', '_').replace('-', '_') }}_{{ year }},
  title={ {ColliderML: {{ pretty_name }}} },
  author={ {ColliderML Collaboration} },
  year={ {{ year }} },
  publisher={NERSC},
  howpublished={\url{ https://huggingface.co/datasets/{{ repo_id }} }},
  note={Simulation performed using ACTS and the Open Data Detector}
}
```

### Contributions

This dataset was produced using:

- **ACTS (A Common Tracking Software)**: https://acts.readthedocs.io/
- **Open Data Detector**: https://acts.readthedocs.io/en/latest/examples/open_data_detector.html
- **EDM4HEP**: https://edm4hep.web.cern.ch/
- **ColliderML Pipeline**: https://github.com/ATLAS-ITk-ML/colliderml

## How to Use This Dataset

### Loading the Dataset

The dataset is hosted on the NERSC public portal and can be streamed directly without downloading:

```python
from datasets import load_dataset

{% for config_name in data_files.keys() %}
# Load {{ config_name }}
{{ config_name }}_ds = load_dataset(
    "{{ repo_id }}",
    "{{ config_name }}",
    split="train"
)
{% endfor %}
```

### Example: Iterating Over Events

```python
import numpy as np

# Iterate over first 10 events
for i, event in enumerate(particles_ds.take(10)):
    event_id = event['event_id']
    n_particles = len(event['particle_id'])

    print(f"Event {event_id}: {n_particles} particles")

    # Access list columns as numpy arrays
    px = np.array(event['px'])
    py = np.array(event['py'])
    pz = np.array(event['pz'])

    # Compute transverse momentum
    pt = np.sqrt(px**2 + py**2)
    print(f"  Mean pt: {pt.mean():.2f} GeV")
```

### Example: Computing Track Features

```python
import numpy as np

for event in tracks_ds.take(5):
    # Get track parameters
    qop = np.array(event['qop'])
    theta = np.array(event['theta'])
    phi = np.array(event['phi'])

    # Compute derived quantities
    pt = np.abs(1.0 / qop) * np.sin(theta)
    eta = -np.log(np.tan(theta / 2.0))

    print(f"Event {event['event_id']}: {len(qop)} tracks")
    print(f"  pt range: [{pt.min():.2f}, {pt.max():.2f}] GeV")
    print(f"  eta range: [{eta.min():.2f}, {eta.max():.2f}]")
```

### Example: Matching Tracks to Particles

```python
# Load both datasets
particles = load_dataset("{{ repo_id }}", "particles", split="train")
tracks = load_dataset("{{ repo_id }}", "tracks", split="train")

# Process event-by-event
for particle_event, track_event in zip(particles, tracks):
    assert particle_event['event_id'] == track_event['event_id']

    # Get particle information
    particle_ids = np.array(particle_event['particle_id'])
    particle_px = np.array(particle_event['px'])
    particle_py = np.array(particle_event['py'])

    # Get track information
    track_particle_ids = np.array(track_event['majority_particle_id'])

    # Compute truth pt for particles
    particle_pt = np.sqrt(particle_px**2 + particle_py**2)

    # Find matched tracks
    for i, pid in enumerate(track_particle_ids):
        if pid in particle_ids:
            idx = np.where(particle_ids == pid)[0][0]
            truth_pt = particle_pt[idx]
            print(f"Track {i}: matched to particle {pid}, pt={truth_pt:.2f} GeV")
```

### Data Location

The Parquet files are hosted at:

```
{{ public_url_base }}
{% for config_name in data_files.keys() %}
├── {{ 'truth' if config_name == 'particles' else 'reco' }}/
│   └── {{ config_name }}/
│       └── *.parquet ({{ data_files[config_name]|length }} files)
{% endfor %}
```

### File Naming Convention

Files follow the pattern:
```
<campaign>.<dataset>.<version>.<category>.<object>.<event_range>.parquet
```

Example: `{{ file_example }}`
- Campaign: `{{ campaign }}`
- Dataset: `{{ dataset }}`
- Version: `{{ version }}`
- Category: `{{ 'truth' if 'particles' in data_files else 'reco' }}`
- Object: one of {{ data_files.keys()|list|join(', ') }}
- Event range: `eventsXXXX-YYYY` (inclusive)

### Performance Tips

1. **Streaming**: Use the dataset API for efficient memory usage
2. **Batch processing**: Process events in chunks for better performance
3. **Selective loading**: Only load the data types you need
4. **Caching**: Use dataset caching for repeated experiments

### Related Datasets

{% for related in related_datasets %}
- **{{ related }}** ({{ related_status }})
{% endfor %}

### Support

For questions, issues, or feature requests:
- Email: {{ contact }}
- GitHub: https://github.com/ATLAS-ITk-ML/colliderml/issues

### Acknowledgments

This work was supported by:
- ATLAS ITk ML Reconstruction project
- NERSC computing resources
- U.S. Department of Energy, Office of Science

---

**Last updated**: {{ date }}
**Dataset version**: {{ version }}
