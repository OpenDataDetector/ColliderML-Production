# ColliderML Documentation (bundled context)

This is the full public documentation for ColliderML, concatenated for the assistant.



========================================
# SOURCE: docs/index.md
========================================

# ColliderML

<AboutData>

The ColliderML dataset is the largest-yet source of full-detail simulation in a virtual detector experiment.

**Why virtual?** The simulation choices are not tied to a construction timeline, there are no budget limitations, no politics. The only goals are to produce the most realistic physics on a detailed detector geometry, with significant computating challenges, in an ML-friendly structure.

The ColliderML dataset provides comprehensive simulation data for machine learning applications in high-energy physics, with detailed detector responses and physics object reconstructions.

</AboutData>

::: tip New to the platform?
Start with the [platform tutorial](/guide/tutorial) — a six-chapter,
runnable walkthrough that covers loading data, local simulation,
remote (SaaS) simulation, benchmark submission, and publishing a
model to the zoo.
:::

## What you can do

- **Load** pre-generated events: `colliderml.load("ttbar_pu0")` — downloads on first call, then caches.
- **Simulate** new events yourself: [`colliderml.simulate(preset="ttbar-quick")`](/guide/simulation) — runs the full Pythia → Geant4 → ACTS pipeline locally, or submit to the SaaS backend with `remote=True`.
- **Score** your models against benchmark tasks: [`colliderml.tasks.evaluate(...)`](/guide/tasks) — six built-in tasks covering tracking, jets, anomaly detection, and systems constraints.
- **Explore** interactively: use the [event display](https://huggingface.co/spaces/murnanedaniel/colliderml-event-display) or the [leaderboard](https://huggingface.co/spaces/murnanedaniel/colliderml-leaderboard) HuggingFace Spaces.

## Get the Data

Download data with the **colliderml** CLI, then load it in Python with Polars. The dataset is also on [HuggingFace](https://huggingface.co/datasets/CERN/ColliderML-Release-1) if you prefer the `datasets` library.

### Quick Start

1. Install and download:

```bash
pip install colliderml
colliderml download --channels ttbar --pileup pu0 --objects particles,tracker_hits,calo_hits,tracks --max-events 200
```

2. Load in Python (same config as the download):

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
# e.g. frames["particles"], frames["tracker_hits"]
```

For more (exploding tables, pileup subsampling, calibration), see the [library docs](/library/overview) and the [exploration notebook](https://github.com/OpenDataDetector/ColliderML/blob/main/notebooks/colliderml_loader_exploration.ipynb).

### Interactive Configuration

Use the configurator below to customize your dataset selection and generate the corresponding code:

<DataConfig />

If there are errors or unexpected behavior, please [open an issue](https://github.com/OpenDataDetector/ColliderML/issues) on the GitHub repository.
<!-- CHANGELOG:DATASET:START -->
::: details Dataset Changelog (latest 5)
- (0.4.0 — 2026-05-27) - HF-native leaderboard score store: `huggingface.co/datasets/CERN/colliderml-benchmark-results` receives one JSON per submission at `results/<task>/<user>/<sha12>.json`. Pairs with the new client-side `colliderml.tasks.submit(..., model_repo_id=...)` which also lands a `.eval_results/colliderml_<task>.yaml` on the user's HF model repo so scores auto-render on the model card.
- (0.4.0 — 2026-05-27) - `CERN/ColliderML-Release-1` configs reorganised: per-pileup/per-object split (`{channel}_{pileup}_{tracker_hits,particles,calo_hits,tracks}`) replaces the prior monolithic per-channel layout. Enables event-range-scoped downloads (the `tracking` eval split is now ~700 MB to pull instead of multiple GB).
- (0.2.0 — 2025-11-07) - Datasets now hosted on HuggingFace Hub for easier access and distribution.
- (0.2.0 — 2025-11-07) - Support for standard HuggingFace `datasets` library for data loading.
- (0.2.0 — 2025-11-07) - Migrated from NERSC manifest-based distribution to HuggingFace datasets.
See the full changelog: [Changelog](/changelog).
:::
<!-- CHANGELOG:DATASET:END -->



========================================
# SOURCE: docs/guide/introduction.md
========================================

# Introduction to ColliderML

ColliderML is a modern machine learning library designed specifically for high-energy physics (HEP) data analysis. It provides efficient tools for accessing, processing, and analyzing large-scale particle physics datasets.

## Why ColliderML?

High-energy physics data analysis presents unique challenges:
- Large-scale datasets distributed across multiple locations
- Complex data formats specific to particle physics
- Need for efficient parallel processing
- Requirements for data integrity and verification

ColliderML addresses these challenges by providing:
1. **Efficient Data Access**
   - Parallel downloading capabilities
   - Resume functionality for interrupted transfers
   - Automatic retries with exponential backoff
   - Progress tracking and detailed status reporting

2. **HEP Data Support**
   - Native support for ROOT files
   - Integration with XRootD for CERN data access
   - Unified interface for various HEP data formats

3. **Machine Learning Integration**
   - Specialized utilities for particle physics
   - Easy integration with popular ML frameworks
   - Tools for dataset preparation and preprocessing

4. **Visualization Tools**
   - Interactive data exploration
   - Physics-specific visualizations
   - Analysis result plotting

## Core Design Principles

ColliderML is built on several key principles:

- **Performance**: Optimized for handling large-scale physics data
- **Reliability**: Robust error handling and data integrity checks
- **Usability**: Clean, intuitive API design
- **Extensibility**: Easy to extend and customize

## Next Steps

- [Installation Guide](./installation.md) - Get ColliderML up and running
- [Quick Start](./quickstart.md) - Start using ColliderML in minutes
- [Core Concepts](./data-management.md) - Learn about the fundamental concepts 


========================================
# SOURCE: docs/guide/installation.md
========================================

# Installation

## Requirements

- Python 3.10 or newer
- pip or conda

## Install from PyPI

The easiest way to install ColliderML is via pip:

```bash
pip install colliderml
```

This will install ColliderML and its core dependencies:
- `datasets` - HuggingFace datasets library for data loading
- `numpy` - Numerical computing
- `h5py` - HDF5 file support (for local data inspection)

## Install from Source

For development or to get the latest features:

```bash
# Clone the repository
git clone https://github.com/OpenDataDetector/ColliderML.git
cd colliderml

# Install in development mode
pip install -e .
```

### Optional feature sets (`extras`)

ColliderML ships a handful of optional extras so that the base install
stays lean. Pick whichever you need:

```bash
# Run simulation pipelines locally (requires Docker or Podman separately)
pip install "colliderml[sim]"

# Submit simulation jobs to the SaaS backend (no Docker needed)
pip install "colliderml[remote]"

# Reference baselines for the benchmark task runner (brings scikit-learn)
pip install "colliderml[tasks]"

# Development tools (testing, formatting, linting, type-checking)
pip install "colliderml[dev]"
```

The `sim` extra pulls in only the Python dependencies required to drive
the container runtime — the actual container image and supporting data
are fetched on first use of `colliderml.simulate()`. See
[Local Simulation](./simulation.md) for the full story and disk-space
expectations.

The `remote` extra installs the `requests` library and gives you
`colliderml.remote.submit`, `status`, `balance`, and the integrated
`colliderml.simulate(remote=True)` path — see
[Remote Simulation](./remote-simulation.md). You will also need a
HuggingFace account and token, which is how the backend authenticates
you.

The `tasks` extra pulls in `scikit-learn` for the reference baselines
shipped with each benchmark task (GBDT for jets, IsolationForest for
anomaly detection). You only need it if you want to run the shipped
baselines verbatim — the task *registry* and local scoring work with
the base install. See [Benchmark Tasks](./tasks.md) for details.

The `dev` extra includes:
- `pytest` and `pytest-cov` for testing
- `black` for code formatting
- `ruff` for linting
- `mypy` for type checking

## Using Conda

If you prefer conda:

```bash
# Create a new environment
conda create -n colliderml python=3.11
conda activate colliderml

# Install ColliderML
pip install colliderml
```

## Verify Installation

Test your installation:

```python
import colliderml
print(f"ColliderML version: {colliderml.__version__}")

# Test loading a dataset
from datasets import load_dataset
dataset = load_dataset(
    "CERN/ColliderML-Release-1",
    "ttbar_pu0_particles",
    split="train"
)
print(f"Successfully loaded {len(dataset)} events")
```

## Troubleshooting

### HuggingFace Hub Access

ColliderML datasets are public and don't require authentication. However, if you experience connection issues:

1. Check your internet connection
2. Try setting a HuggingFace cache directory:
   ```bash
   export HF_HOME=/path/to/cache
   ```
3. Consult the [HuggingFace datasets documentation](https://huggingface.co/docs/datasets)

### Python Version Issues

ColliderML requires Python 3.10 or newer. Check your version:

```bash
python --version
```

If you have an older version, consider using conda or pyenv to manage multiple Python versions.

## Next Steps

- Continue to the [Quickstart Guide](./quickstart.md) to start using ColliderML
- Learn about [Data Structure](./data-structure.md) and available datasets



========================================
# SOURCE: docs/guide/quickstart.md
========================================

# Quickstart Guide

This guide will help you get started with ColliderML datasets from HuggingFace Hub.

## Installation

Install ColliderML using pip:

```bash
pip install colliderml
```

Or install from source:

```bash
git clone https://github.com/OpenDataDetector/ColliderML.git
cd colliderml
pip install -e .
```

## Loading Your First Dataset

The ColliderML dataset is hosted on HuggingFace Hub and can be loaded using the standard `datasets` library:

```python
from datasets import load_dataset

# Load the ttbar particles dataset (no pileup)
dataset = load_dataset(
    "CERN/ColliderML-Release-1",
    "ttbar_pu0_particles",
    split="train"
)

print(f"Loaded {len(dataset)} events")
```

## Prefer a local CLI + loader workflow?

The HuggingFace `datasets` approach is great for quick access. For analysis workflows, ColliderML also provides a more convenient pattern:

- `colliderml.load("ttbar_pu0")` — one-liner that downloads on first call, then caches
- explode event tables into flat tables with `colliderml.polars.explode_*`

```python
import colliderml
from colliderml.polars import explode_particles

# Downloads on first call, reads from the local cache afterwards.
frames = colliderml.load("ttbar_pu0", tables=["particles"], max_events=200)
particles_flat = explode_particles(frames["particles"])
```

See the [Library overview](/library/overview), then:

- [CLI download](/library/cli)
- [Local loading](/library/loading)
- [Exploding tables](/library/exploding)

## Generate events yourself

New in v0.4.0. Instead of downloading pre-generated data, you can
simulate your own events locally (inside the ODD software container
via Docker or Podman) or submit a job to the SaaS backend:

```python
import colliderml

# Local: needs `pip install "colliderml[sim]"` plus Docker or Podman.
result = colliderml.simulate(preset="ttbar-quick")
print(result.run_dir)                       # parquet outputs land here

# Remote: needs `pip install "colliderml[remote]"` and an HF token.
result = colliderml.simulate(preset="higgs-portal-quick", remote=True)
print(result.remote_request_id)
```

Full details in the [Local Simulation](./simulation.md) and
[Remote Simulation](./remote-simulation.md) guides.

## Score your model against a benchmark task

New in v0.4.0. Six built-in benchmark tasks (`tracking`, `jets`,
`anomaly`, and three systems tasks) let you compare any model on
equal footing:

```python
import colliderml.tasks

# What's available?
print(colliderml.tasks.list_tasks())

# Score local predictions
scores = colliderml.tasks.evaluate("tracking", "my_preds.parquet")
print(scores)

# Upload to the leaderboard (earns credits on new bests)
colliderml.tasks.submit("tracking", "my_preds.parquet")
```

See the [Benchmark Tasks guide](./tasks.md) for the full workflow.

## Understanding Dataset Structure

The ColliderML dataset is organized with configurations that combine:

- **Process**: The physics process being simulated (e.g., `ttbar`, `ggf`, `dihiggs`)
- **Pileup**: The pileup condition (e.g., `pu0` for no pileup, `pu200` for 200 pileup)
- **Object Type**: The detector data type or hierarchy level

### Available Configurations

Each configuration name follows the pattern `{process}_{pileup}_{object_type}`. For example:
- `ttbar_pu0_particles`
- `ggf_pu200_calo_hits`
- `dihiggs_pu0_tracks`

ColliderML provides multiple views of collision events:

- `particles`: Truth-level particle information (Monte Carlo truth)
- `tracker_hits`: Detector measurements in the tracking system
- `calo_hits`: Detector measurements in the calorimeter
- `tracks`: Reconstructed particle tracks

### Loading Different Configurations

```python
from datasets import load_dataset

# Load truth-level particles
particles = load_dataset(
    "CERN/ColliderML-Release-1",
    "ttbar_pu0_particles",
    split="train"
)

# Load tracker hits (detector measurements)
tracker_hits = load_dataset(
    "CERN/ColliderML-Release-1",
    "ttbar_pu0_tracker_hits",
    split="train"
)

# Load reconstructed tracks
tracks = load_dataset(
    "CERN/ColliderML-Release-1",
    "ttbar_pu0_tracks",
    split="train"
)
```

## Accessing Event Data

### Single Event

```python
# Get the first event
event = dataset[0]

# Inspect available fields
print("Event fields:", list(event.keys()))

# Access specific fields
for key, value in event.items():
    if hasattr(value, '__len__'):
        print(f"{key}: {len(value)} items")
    else:
        print(f"{key}: {value}")
```

### Batch Loading

Load multiple events at once for efficient processing:

```python
# Load first 10 events as a batch
batch = dataset[:10]

# batch is a dictionary where each value is a list
print("Batch keys:", list(batch.keys()))

# Process batch data
for key, values in batch.items():
    if hasattr(values, '__len__'):
        print(f"{key}: batch of {len(values)} events")
```

### Iteration

Iterate through the dataset:

```python
# Iterate over all events
for event in dataset:
    # Process each event
    print(f"Processing event with {len(event.keys())} fields")

    # Your analysis code here
    break  # Remove this to process all events
```

## Streaming Mode

For large datasets that don't fit in memory, use streaming mode:

```python
from datasets import load_dataset

# Load in streaming mode
dataset = load_dataset(
    "CERN/ColliderML-Release-1",
    "ttbar_pu0_particles",
    split="train",
    streaming=True  # Enable streaming
)

# Iterate without loading everything into memory
for i, event in enumerate(dataset):
    if i >= 10:  # Process first 10 events
        break
    print(f"Event {i}: {list(event.keys())}")
```

## Available Physics Processes

ColliderML includes multiple physics processes:

### Top Quark Pair Production (ttbar)

```python
dataset = load_dataset(
    "CERN/ColliderML-Release-1",
    "ttbar_pu0_particles",
    split="train"
)
```

### Gluon-Gluon Fusion / Higgs (ggf)

```python
dataset = load_dataset(
    "CERN/ColliderML-Release-1",
    "ggf_pu0_particles",
    split="train"
)
```

### Di-Higgs Production (dihiggs)

```python
dataset = load_dataset(
    "CERN/ColliderML-Release-1",
    "dihiggs_pu0_particles",
    split="train"
)
```

Check the [CERN/ColliderML-Release-1 dataset page](https://huggingface.co/datasets/CERN/ColliderML-Release-1) for a complete list of available configurations.

## Data Inspection Example

Here's a complete example of loading and inspecting ColliderML data:

```python
from datasets import load_dataset
import numpy as np

# Load dataset
dataset = load_dataset(
    "CERN/ColliderML-Release-1",
    "ttbar_pu0_particles",
    split="train"
)

print(f"\nDataset Information:")
print(f"  Total events: {len(dataset)}")
print(f"  Features: {dataset.features}")

# Inspect first event
event = dataset[0]
print(f"\nFirst Event Structure:")

for key, value in event.items():
    print(f"  {key}:")
    print(f"    Type: {type(value)}")

    if hasattr(value, 'shape'):
        print(f"    Shape: {value.shape}")
        print(f"    Dtype: {value.dtype}")

        # Print statistics for numeric arrays
        if np.issubdtype(value.dtype, np.number) and value.size > 0:
            print(f"    Range: [{np.min(value):.3f}, {np.max(value):.3f}]")
            print(f"    Mean: {np.mean(value):.3f}")
```

## Using with PyTorch or TensorFlow

The datasets library integrates seamlessly with popular ML frameworks:

### PyTorch

```python
from datasets import load_dataset

dataset = load_dataset(
    "CERN/ColliderML-Release-1",
    "ttbar_pu0_particles",
    split="train"
)

# Convert to PyTorch format
dataset.set_format(type='torch', columns=['your_feature_columns'])

# Use with PyTorch DataLoader
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### TensorFlow

```python
# Convert to TensorFlow format
tf_dataset = dataset.to_tf_dataset(
    columns=['your_feature_columns'],
    batch_size=32,
    shuffle=True
)
```

## Next Steps

- Explore the [Data Structure](./data-structure.md) documentation for detailed field descriptions
- Learn about [Data Management](./data-management.md) for caching and optimization
- Check out the [Examples](../examples/) for complete analysis workflows
- Read about the [Physics Processes](./processes.md) available in ColliderML

## Getting Help

If you encounter issues:

- Check the [FAQ](./faq.md)
- Visit our [GitHub Issues](https://github.com/OpenDataDetector/ColliderML/issues)
- Consult the [CERN/ColliderML-Release-1 dataset page](https://huggingface.co/datasets/CERN/ColliderML-Release-1)



========================================
# SOURCE: docs/guide/tutorial.md
========================================

# Platform tutorial

A from-zero walkthrough of the ColliderML platform from a
researcher's perspective. By the end you'll have **downloaded a
tracker-hits sample from HuggingFace**, **simulated events locally
with the official container**, **submitted a simulation request to
the SaaS backend**, **scored a tracking submission against the
leaderboard**, and **published your "model" as an HF repo that the
model-zoo Space indexes**.

There's an executable version of this tutorial in
[`notebooks/tutorial.ipynb`](https://github.com/OpenDataDetector/ColliderML/blob/main/notebooks/tutorial.ipynb).
Every code block below comes straight out of that notebook, so you
can read along here and run any piece you like without leaving the
docs.

Each chapter ends with a **"what just happened"** section that
unpacks the architecture under the hood, so the tutorial also
doubles as a conceptual map.

## Prerequisites

| | |
|---|---|
| **Python** | 3.10 or newer. |
| **Install** | `pip install --pre 'colliderml[all]'`  (pre-release until `0.4.0` final; drop `--pre` once it ships). |
| **Disk headroom** | ~2 GB for the Chapter 1 dataset slice. |
| **Chapter 2** | Docker or Podman on `$PATH`, plus ~12 GB for the pipeline container and Geant4 datasets (one-time download). |
| **Chapters 3–4** | The ColliderML backend must be reachable. In production this is `api.colliderml.com` (the default). For local development, see the [operator note](#running-the-backend-yourself) at the bottom of this page. |
| **Chapters 3–5** | A HuggingFace account and a token (`huggingface-cli login`). |

If you're on a machine with a tight `$HOME` quota (e.g. NERSC
login nodes), redirect caches before anything else:

```bash
export COLLIDERML_DATA_DIR=/scratch/$USER/colliderml-cache
export HF_HOME=/scratch/$USER/hf-cache
```

## How the pieces fit together

```
┌──────────────┐   HTTP    ┌────────────────┐   SFAPI   ┌───────────┐
│ your laptop  │──────────▶│ backend (FastAPI│──────────▶│ Perlmutter│
│ colliderml.* │           │  +  Postgres)  │           │  (Slurm)  │
└──────┬───────┘           └────────┬───────┘           └─────┬─────┘
       │                            │                         │
       │  HuggingFace Hub           │   uploads artefacts     │
       ▼                            ▼                         ▼
┌──────────────┐            ┌──────────────────┐      ┌───────────┐
│ datasets,    │            │ HF dataset repos │◀─────│ pipeline  │
│ model zoo,   │            │  (simulated      │      │  output   │
│ Spaces       │            │   events)        │      └───────────┘
└──────────────┘            └──────────────────┘
```

Chapters 1 and 5 use only the HuggingFace side (leftmost column).
Chapter 2 is your laptop alone. Chapters 3 and 4 exercise the full
stack.

---

## Chapter 1 — Loading data from HuggingFace

The canonical ColliderML datasets live at
[`CERN/ColliderML-Release-1`](https://huggingface.co/datasets/CERN/ColliderML-Release-1).
Each config is named `<channel>_<pileup>`, e.g. `ttbar_pu0` or
`higgs_portal_pu200`. The library's `load()` handles cache-aware
download and Polars loading in one call.

```python
from colliderml.core.hf_download import discover_remote_configs
import colliderml

configs = discover_remote_configs("CERN/ColliderML-Release-1")
print(f"{len(configs)} configs available; first six:", configs[:6])

frames = colliderml.load(
    "ttbar_pu0",
    tables=["tracker_hits", "particles"],
    max_events=200,
)
row = frames["tracker_hits"].row(0, named=True)
print("event_id:", row["event_id"], "number of hits:", len(row["hit_id"]))
```

### What just happened

- `discover_remote_configs` hit the HF dataset repo and extracted unique
  `data/<config>/...` prefixes — the config list is derived from
  what's on disk in the repo, not a hardcoded manifest.
- `colliderml.load(...)` resolved `"ttbar_pu0"` into the set of
  parquet shards for the requested tables, downloaded any missing
  shards into `$COLLIDERML_DATA_DIR` (default `~/.cache/colliderml`),
  and loaded them with the Polars-backed loader.
- `max_events=200` is an **in-memory slice**, not a download limit —
  the full shard has to be fetched once, then we take the first 200
  rows. For a tiny sample this is wasteful; for real workflows the
  cache pays for itself from the second call onward.

The nested schema (one row per event, columns as lists across hits)
is central to the library's performance story — Polars can lazily
scan these shards without exploding them into flat tables.

---

## Chapter 2 — Run the pipeline yourself: local simulation

ColliderML ships a container
(`ghcr.io/opendatadetector/sw`)
that bundles MadGraph, Pythia, Geant4 + ddsim, and ACTS. The
`colliderml.simulate` subpackage drives the full pipeline —
hard-scatter generation → parton shower → detector simulation →
track reconstruction — in a single call.

Three knobs control what you actually generate:

- **`channel`** — the physics process (`ttbar`, `higgs_portal`, …; see
  `colliderml.simulate.load_presets()` for the wired-up channels).
- **`events`** — how many hard-scatter events. Each one costs ~10–30 s
  of Geant4 stepping on a typical laptop CPU.
- **`pileup`** — average soft pp interactions overlaid per hard scatter.
  `0` is fastest (signal only); LHC reality is ~200. Cost scales
  roughly linearly with pileup.

For this chapter we set the three knobs explicitly to tiny values so
the whole pipeline finishes in ~2 minutes. Bump them once you have
time:

```python
import colliderml

CHANNEL = "higgs_portal"
EVENTS = 2
PILEUP = 5

result = colliderml.simulate(
    channel=CHANNEL,
    events=EVENTS,
    pileup=PILEUP,
    quiet=True,
)
print("run directory:", result.run_dir)
print("stages run:   ", [s.name for s in result.stages])
```

Presets are also available as named bundles of the same three knobs —
`colliderml.simulate(preset="ttbar-quick")` is equivalent to setting
`channel="ttbar", events=10, pileup=0`. List them with
`colliderml.simulate.load_presets()`. The tutorial uses explicit knobs
so the dimensions stay visible.

### What just happened

On first call, `colliderml.simulate` did three things before running
anything:

1. **Detected the runtime** — `docker` preferred, `podman` fallback.
2. **Cloned `colliderml-production@pipeline-v0.1.0`** into `.cache/`
   for the pipeline scripts. This auto-clone pattern mirrors the one
   already used for the ODD geometry and the MG5 ↔ Pythia8
   interface.
3. **Built the ODD detector geometry and downloaded Geant4
   datasets** into the cache (both one-time, ~10 minutes combined).

Then for each stage (Pythia → DDSim → Digi + Reco → Parquet) it
started a container with the pipeline script for that stage, mounted
the run directory, and ran to completion. The `SimulationResult` you
get back records which stages ran, how long each took, and where the
outputs are.

The **same** parquet output format that Chapter 1 downloaded from HF
is what this pipeline writes locally, so you can round-trip a
simulated sample through `colliderml.load()` without caring about
the source.

---

## Chapter 3 — Scale up: Simulation as a Service

Single-muon runs in minutes on your laptop. A full `ttbar + PU=200`
run takes a day of CPU time and tens of GB of output — not
something to babysit locally. The SaaS backend accepts a JSON
request, queues a Slurm job on NERSC Perlmutter via the SFAPI,
polls it to completion, and uploads the artefacts to an HF dataset
repo under your account.

From the user's side it's three calls: `submit`, `wait_for`, then
download the resulting HF repo like any other dataset.

```python
from colliderml import remote

print(f"You have {remote.balance():.0f} credits available.")

sub = remote.submit(channel="ttbar", events=10, pileup=10)
print(f"request_id: {sub.request_id}, state: {sub.state}, credits: {sub.credits_charged}")

final = remote.wait_for(sub.request_id, poll_interval=1, timeout=120)
print("final state:", final.state, "output repo:", final.output_hf_repo)
```

An identical second `submit` hits the backend's dedup cache:

```python
dup = remote.submit(channel="ttbar", events=10, pileup=10)
assert dup.request_id == sub.request_id  # same request
assert dup.credits_charged == 0          # no double-charge
```

### What just happened

Inside the backend, `POST /v1/simulate` did the following:

1. **Hashed the request config** and checked the dedup table — if a
   completed request with the same hash exists within the last 7
   days, returns it verbatim (`credits_charged=0`).
2. **Ran credit and abuse checks.** The kill switch
   (`POST /admin/freeze`) short-circuits here with a 503 if an
   operator has tripped it.
3. **Inserted a row** into `simulation_requests` with state `queued`.
4. **Handed off to `SFAPIRunner.submit`:**
   - **With real NERSC credentials** (`SFAPI_CLIENT_ID`,
     `SFAPI_CLIENT_SECRET` set on the backend): renders
     `app/sbatch_template.sh.j2`, uploads it to `/pscratch/sd/.../colliderml/`,
     POSTs to the SFAPI `/compute/jobs` endpoint, spawns a polling
     task that hits `/compute/jobs/{id}` every 60 s.
   - **In mock mode** (default for local dev): spawns a task that
     marks the request completed after 2 seconds.
5. **Returned** the `(request_id, state=queued, credits_charged, …)`
   tuple.

`remote.wait_for` polls `GET /v1/requests/{id}` every
`poll_interval` seconds until the state is terminal (`completed` /
`failed` / `cancelled`). On success you get back an `output_hf_repo`
URL — a fresh HF dataset repo under your account containing the
parquet shards, loadable via the same `colliderml.load()` you used
in Chapter 1.

::: tip Mock vs real SFAPI
The backend runs in mock mode unless `SFAPI_CLIENT_ID` and
`SFAPI_CLIENT_SECRET` are set. Mock mode reproduces the full
state-machine (queued → running → completed) in about 2 seconds
so the API shape is exact — you just don't get real pipeline
output. See the internal [SFAPI runner docs](../internal/sfapi.md)
for the production setup.
:::

---

## Chapter 4 — Benchmark a tracking algorithm

The `tracking` task asks you to reconstruct particle tracks from
detector hits in `ttbar_pu200` events. Your algorithm receives
**tracker hits only** — positions and measurement data — and must
output a grouping of hits into tracks. A submission is a parquet
file with three columns: `event_id`, `hit_id`, `track_id`.

Critically, the **truth** (which particle produced which hit) is
**only available on the server** for the held-out eval split
(events 90 000–99 999). You never see it. Your tracker must work
without it.

### Understanding the metric (training split, where you have truth)

Before submitting anything, let's build intuition for how the
scoring works. On the **training split** (events 0–89 999) truth
*is* available, so we can run oracle baselines that cheat — they
assign each hit to its true `particle_id`. These can't be submitted
(the eval split doesn't ship truth to clients), but they're perfect
for understanding the metric.

The primary metric is **TrackML weighted efficiency**, scored by the
double-majority rule:

> A reconstructed track is **correct** if
> **one truth particle owns ≥50% of the track's hits**
> AND
> **that particle contributes ≥50% of its own hits to the track**.
>
> Efficiency = sum of hit weights in correct tracks ÷ total hit weight.

Three companion metrics unpack the failure modes:

- **Fake rate** — fraction of tracks where no single particle
  dominates (fails the first half of the rule).
- **Duplicate rate** — fraction of truth particles matched to more
  than one reconstructed track.
- **Physics efficiency (pT > 1 GeV)** — fraction of high-pT primary
  particles with *any* reconstructed track.

```python
import colliderml
from colliderml.tasks.tracking.baselines import (
    noised_oracle_predictions,
    perfect_oracle_predictions,
)
from colliderml.tasks.tracking.metrics import trackml_weighted_efficiency

# Use training events (0–999) where truth is available locally.
tracking = colliderml.tasks.get("tracking")
truth = tracking.load(tables=["tracker_hits"], event_range=(0, 50))["tracker_hits"]

perfect = perfect_oracle_predictions(truth)
noised = noised_oracle_predictions(truth, split_fraction=0.1, merge_fraction=0.1, seed=42)

print("perfect oracle eff:", trackml_weighted_efficiency(perfect, truth))
print("noised oracle eff: ", trackml_weighted_efficiency(noised, truth))
```

The oracle baselines teach two things:

- `split_fraction` breaks tracks into asymmetric pieces. The small
  fragment's particle contribution falls below 50%, scoring as a
  fake. Efficiency drops roughly linearly.
- `merge_fraction` unifies two tracks into one. Neither particle
  owns a majority of the merged track, so **both** contributions
  are lost. Harsher — roughly quadratic near zero.

### Building a real (naive) tracker

A real submission operates on tracker hits **without truth**. The
simplest approach: cluster the hits spatially. Hits from the same
particle tend to lie along a helical trajectory, so even a crude
spatial clustering catches some of that structure.

```python
import numpy as np
import pyarrow as pa
from sklearn.cluster import DBSCAN

# Load eval-range tracker_hits (no truth here — only positions).
eval_hits = tracking.load(
    tables=["tracker_hits"],
    event_range=(90_000, 90_050),
)["tracker_hits"]

cols = eval_hits.to_pydict()
x = np.array(cols["tx"])       # global x position
y = np.array(cols["ty"])       # global y position
z = np.array(cols["tz"])       # global z position

# Cylindrical coords — better for helical tracks.
r = np.sqrt(x**2 + y**2)
phi = np.arctan2(y, x)
eta = np.arctanh(z / np.sqrt(x**2 + y**2 + z**2 + 1e-9))

# Cluster per-event (DBSCAN doesn't know about event boundaries,
# so we loop). This is deliberately naive — a real tracker would
# use a GNN or Kalman filter.
events = np.array(cols["event_id"])
track_ids = np.full(len(events), -1)

for eid in np.unique(events):
    mask = events == eid
    features = np.column_stack([phi[mask], eta[mask], r[mask] / r.max()])
    labels = DBSCAN(eps=0.15, min_samples=3).fit_predict(features)
    # Shift labels to avoid collisions across events.
    labels[labels >= 0] += track_ids.max() + 1
    track_ids[mask] = labels

preds = pa.table({
    "event_id": pa.array(cols["event_id"]),
    "hit_id":   pa.array(cols["hit_id"]),
    "track_id": pa.array(track_ids.tolist()),
})
print(f"DBSCAN found {len(set(track_ids)) - 1} track candidates")
```

::: warning Column names may vary
The exact position-column names (`tx`, `ty`, `tz`) depend on the
dataset release. Inspect `eval_hits.column_names` and adapt. The
tracking task definition (`tracking.inputs`) lists which tables are
available.
:::

### Submit to the leaderboard

```python
result = colliderml.tasks.submit("tracking", preds)
print("server scores: ", result["scores"])
print("credits earned:", result["credits_earned"])
```

### What just happened

The backend's `POST /v1/benchmark/tracking/submit` took your
parquet bytes, loaded them with pyarrow, and ran
`trackml_weighted_efficiency` / `fake_rate` / `duplicate_rate` /
`physics_eff_pt1` against the **server-held truth** for events
90 000–99 999 — data you never saw. If your result beats the
current leaderboard best on any metric with `higher_is_better=True`,
credits are written to your ledger. The leaderboard Space
(`spaces/leaderboard/`) polls `/v1/leaderboard/tracking` to render
the public table.

The DBSCAN baseline above is deliberately terrible — it ignores
curvature, momentum, and layer ordering. A real tracker (Kalman
filter, GNN, transformer) would improve dramatically. But the
submission flow is identical: produce `(event_id, hit_id, track_id)`
rows, call `colliderml.tasks.submit("tracking", preds)`, and let
the backend score it.

---

## Chapter 5 — Share your model: the model zoo

The ColliderML model zoo is a thin HuggingFace Hub filter: any model
on the Hub tagged `colliderml` shows up in
[`spaces/model-zoo`](https://huggingface.co/spaces/murnanedaniel/colliderml-model-zoo).
There's no backend registry, no approval process — the `tag` is the
contract. Publishing is a standard `huggingface_hub.create_repo` +
`upload_folder`.

```python
import pathlib, textwrap
from huggingface_hub import HfApi, create_repo, upload_folder

hf_user = HfApi().whoami()["name"]
model_dir = pathlib.Path("/tmp/colliderml-tutorial-model")
model_dir.mkdir(exist_ok=True)

(model_dir / "config.json").write_text(
    '{"kind": "noised-oracle", "split_fraction": 0.1, "merge_fraction": 0.1}\n'
)
(model_dir / "README.md").write_text(textwrap.dedent("""
    ---
    tags:
      - colliderml
      - tracking
    license: mit
    ---
    # Noised-oracle tracker (tutorial)

    Pedagogical baseline for the ColliderML tracking task. Not a real
    tracker — serves as a calibration point for evaluating real
    submissions.
"""))

repo_id = f"{hf_user}/colliderml-tutorial-tracker"
create_repo(repo_id, exist_ok=True)
upload_folder(folder_path=str(model_dir), repo_id=repo_id)
print("published:", f"https://huggingface.co/{repo_id}")
```

Refresh the model-zoo Space — your repo now appears.

### What just happened

You created an HF model repo under your namespace with a README that
has `tags: [colliderml, tracking]` in its YAML frontmatter. The
model-zoo Space runs

```python
huggingface_hub.list_models(filter="colliderml")
```

on every page load and renders whatever comes back. Your repo is now
one of the results.

**Tradeoff.** The simplicity is deliberate: no backend registry
means no moderation, no provenance linking, no guaranteed
compatibility with evaluation tooling. The contract is just the
tag. A future iteration could add a `/v1/models` backend endpoint
that stitches together a model card with its leaderboard score(s),
or runs the model's inference against the eval set to produce a
deterministic score — but that's a later conversation.

---

## Chapter 6 — Recap and next steps

You just drove every public surface of the ColliderML platform:

| Chapter | Surface |
|---|---|
| 1 | `colliderml.load()`, `colliderml.core.hf_download` |
| 2 | `colliderml.simulate()`, preset catalogue, container auto-clone |
| 3 | `colliderml.remote.{submit, wait_for, balance}`, dedup cache |
| 4 | `colliderml.tasks.{get, evaluate, submit}`, tracking metrics, oracle baselines |
| 5 | `huggingface_hub.create_repo`, the `tags: [colliderml]` contract |

### Reference card

```python
import colliderml
from colliderml import remote
from colliderml.tasks.tracking.baselines import noised_oracle_predictions
from huggingface_hub import create_repo, upload_folder

# 1. Load data
frames = colliderml.load("ttbar_pu0", tables=["tracker_hits"], max_events=200)

# 2. Run the pipeline locally
result = colliderml.simulate(channel="higgs_portal", events=2, pileup=5)

# 3. Submit a remote simulation
sub = remote.submit(channel="ttbar", events=10_000, pileup=200)
final = remote.wait_for(sub.request_id)

# 4. Score a tracking submission
preds = noised_oracle_predictions(truth_hits, split_fraction=0.05)
result = colliderml.tasks.submit("tracking", preds)

# 5. Publish a model (standard HF)
create_repo(f"{user}/my-tracker", exist_ok=True)
upload_folder(folder_path="...", repo_id=f"{user}/my-tracker")
```

### What's not covered

- **Other tasks.** `jets`, `anomaly`, `tracking_latency`,
  `tracking_small`, `data_loading` all ship with reference baselines
  and the same `colliderml.tasks.{get, evaluate, submit}` surface.
- **Writing your own task.** Subclass
  `colliderml.tasks.BenchmarkTask`, register with `@register`, ship
  reference baselines. An operator adds the new task to the
  backend's task list.
- **Running the platform yourself.** The
  [operator docs](../internal/overview.md) (unlisted, reachable by
  URL) cover backend deployment, SFAPI credentials, admin Space
  operations, and the container image build checklist.
- **Webhooks and multi-node.** The backend supports posting a
  webhook when a request completes and has multi-node sbatch
  templates for the `*-benchmark` presets — useful for CI/CD-style
  workflows.

Issues, questions, ideas:
[github.com/OpenDataDetector/ColliderML/issues](https://github.com/OpenDataDetector/ColliderML/issues).

---

## Running the backend yourself {#running-the-backend-yourself}

::: details For operators and contributors — not needed for most users

In production the backend runs at `api.colliderml.com` and the
library talks to it by default. If you're developing backend
features, running integration tests, or want to demo the full
stack locally, you can spin it up from the companion
[`colliderml-production`](https://github.com/OpenDataDetector/colliderml-production)
repo:

```bash
git clone git@github.com:OpenDataDetector/colliderml-production.git
cd colliderml-production/backend
docker-compose up -d          # or podman-compose up -d
export COLLIDERML_BACKEND=http://localhost:8000
```

Then use the provided setup helper to grant yourself tutorial
credits:

```bash
# from the public colliderml repo:
bash scripts/setup_tutorial_env.sh ../colliderml-production/backend your-hf-username
```

By default the backend runs in **mock SFAPI mode** — simulation
requests complete in ~2 seconds with no real pipeline execution.
Set `SFAPI_CLIENT_ID` and `SFAPI_CLIENT_SECRET` to switch to
real Perlmutter job submission. See the
[operator docs](../internal/sfapi.md) for details.
:::



========================================
# SOURCE: docs/guide/simulation.md
========================================

# Local Simulation

ColliderML can generate new events on your own machine using the full
physics pipeline: Pythia (or MadGraph + Pythia for NLO processes) →
Geant4 detector simulation via DDSim → digitisation and track
reconstruction via ACTS, which writes the parquet outputs directly. Everything runs inside the
official OpenDataDetector software container, so you do not need to
install any HEP tooling yourself — just Docker or Podman.

This page covers the **local** story: running the pipeline on your own
workstation or a shared node. If you would rather hand the work off to a
shared cluster, see [Remote Simulation](./remote-simulation.md).

## When to run locally

| Use local simulation when | Use remote simulation when |
| :-- | :-- |
| You want tight iteration on a few events | You need >100 events with pileup |
| You have Docker / Podman + ~20 GB free disk | You do not have a container runtime |
| You want offline reproducibility | You want someone else to pay the compute bill |
| You are debugging a pipeline change | You are running a benchmark |

A typical iteration loop — `simulate → inspect → tweak → simulate again`
— fits comfortably within 10–15 minutes for the `*-quick` presets.

## Prerequisites

1. **Python 3.10 or newer** — see [Installation](./installation.md).
2. **Either Docker or Podman** on your `$PATH`. The library auto-detects
   whichever is installed; Docker is preferred if both are present.
3. **Disk headroom**:
   - ~10 GB for the OpenDataDetector software image (pulled once)
   - ~2 GB for the Geant4 physics data tables (downloaded on first run)
   - 50 MB to a few hundred MB per run, depending on event count
4. The simulate extras installed:
   ```bash
   pip install "colliderml[sim]"
   ```

## The first run, annotated

```python
import colliderml

result = colliderml.simulate(preset="ttbar-quick")
print(result.run_dir)
print([p.name for p in result.list_files()])
```

On first invocation, `simulate()` will:

1. **Detect the container runtime** (`docker` or `podman`). If neither is
   available you get a clear error pointing at
   [Remote Simulation](./remote-simulation.md) instead.
2. **Prompt before pulling** the ~10 GB container image (skipped in
   non-interactive environments — pass `image="..."` to override the
   pinned tag).
3. **Clone `colliderml-production`** into
   `~/.cache/colliderml/simulate/colliderml-production/`. This repo holds
   the stage scripts, YAML config templates, and container bootstrap
   script. It is pinned to a known-good git ref for reproducibility.
4. **Clone OpenDataDetector v4.0.4** (CERN GitLab) and
   **MG5aMC_PY8_interface** (GitHub) into the same cache directory. These
   provide the detector geometry and MadGraph/Pythia bridge that the
   container expects at `/cache/odd-v4` and `/cache/MG5aMC_PY8_interface`.
5. **Run each pipeline stage** in order. Progress is printed to stdout;
   the first stage typically takes a few seconds, detector sim is the
   long pole (~10 minutes for 10 events).
6. **Return a `SimulationResult`** with paths to the outputs.

Everything cached in step 2–4 persists between runs, so the *second*
invocation skips straight to step 5 and starts producing events
immediately.

## Presets versus explicit parameters

Presets are named shorthand for common workloads:

```python
# Preset: bundled (channel, events, pileup) triple.
colliderml.simulate(preset="ttbar-quick")

# Explicit parameters.
colliderml.simulate(channel="higgs_portal", events=10, pileup=10)

# Preset as a starting point, but bump pileup:
colliderml.simulate(preset="ttbar-quick", pileup=40)
```

Explicit arguments always win over the preset's defaults. List the full
preset catalogue on the CLI:

```bash
colliderml list-presets
```

## What comes out

`SimulationResult.run_dir` points at a per-run directory (e.g.
`colliderml_output/ttbar_pu0_10evt/runs/0/`) containing:

| File | Stage | Format |
| :-- | :-- | :-- |
| `events.hepmc.gz` | MadGraph (ttbar only) | HepMC2 |
| `merged_events.hepmc3` | Pythia + pileup | HepMC3 |
| `edm4hep.root` | DDSim / Geant4 | EDM4hep ROOT |
| `measurements.root` | Digitisation | ACTS ROOT |
| `particles.root` | Truth particles | ACTS ROOT |
| `tracksummary_ambi.root` | Reconstruction | ACTS ROOT |
| `*.parquet` | Digitisation / reconstruction | Parquet |

The parquet outputs can be loaded with the same
[`colliderml.load()`](../library/loading.md) helper used for the released
dataset, so downstream analysis code does not care whether the events
came from HuggingFace or from your own machine.

## Cache layout and disk hygiene

Everything is rooted at `$COLLIDERML_CACHE` (defaults to
`~/.cache/colliderml/`):

```
~/.cache/colliderml/simulate/
├── colliderml-production/          # pinned production-repo clone
│   └── .cache/
│       ├── odd-v4/                 # OpenDataDetector geometry
│       ├── g4data/                 # Geant4 physics tables (~2 GB)
│       └── MG5aMC_PY8_interface/   # MG5 → Pythia bridge
```

Wipe the whole tree at any time — `simulate()` will re-populate it on
the next run. If a stage complains that a cache is corrupt, call
`colliderml.simulate.docker.clone_colliderml_production(force_refresh=True)`
to re-fetch just the production-repo clone.

## Overriding the pinned pipeline ref

The pinned `colliderml-production` ref is a module-level constant in
[`colliderml.simulate.docker`](../library/simulate.md#auto-cloning-the-production-repo).
To point at an in-flight branch of the production repo without editing
the library:

```bash
export COLLIDERML_PRODUCTION_REF=my-feature-branch
python -c "import colliderml; colliderml.simulate(preset='ttbar-quick')"
```

This is the supported escape hatch for pipeline developers working on
both repos at once; end users should stick with the pinned default.

## Troubleshooting

**"Neither docker nor podman is installed."** — install one of them, or
run remotely with `simulate(..., remote=True)`.

**"daemon is not responding"** — Docker Desktop needs to be running (on
macOS/Windows) or the `dockerd` service needs to be up (on Linux). Try
`docker info` from a shell to confirm.

**First run gets stuck on "Pulling …"** — the OpenDataDetector image is
~10 GB. Check your connection and let it finish; subsequent runs reuse
the cached image.

**Stage fails with `libOpenDataDetector.so not found`** — the production
checkout in the cache is corrupt. Re-clone with
`force_refresh=True`, or delete `~/.cache/colliderml/simulate/` and let
`simulate()` start fresh.

**"No stages were generated"** — the pinned production-repo ref is
missing config YAML files for the requested channel. Bump
`COLLIDERML_PRODUCTION_REF` to a newer ref, or file an issue.

## Next steps

- Run the CLI equivalent: [`colliderml simulate`](../library/cli.md#simulate)
- Look up the `simulate()` signature: [Simulate API](../library/simulate.md)
- Pipe the outputs into a benchmark: [Benchmark Tasks](./tasks.md)
- Offload heavy runs: [Remote Simulation](./remote-simulation.md)



========================================
# SOURCE: docs/guide/remote-simulation.md
========================================

# Remote Simulation

The ColliderML SaaS backend runs the same pipeline as
[Local Simulation](./simulation.md) — but on NERSC Perlmutter, dispatched
via a small FastAPI service. You submit a job, get a request ID, and
collect the results from a per-request HuggingFace dataset when the run
completes. No container runtime needed on your side.

This page covers the operator-facing story. For the API reference (every
function signature, error type, env var), see
[`colliderml.remote`](../library/remote.md).

## When to use remote simulation

| Use remote when | Use local instead when |
| :-- | :-- |
| You need more than a few hundred events | You want tight iteration cycles |
| You want realistic pileup (mu ≥ 10) | You have Docker/Podman already |
| You are on a laptop / CI runner without Docker | You need offline reproducibility |
| You want the job's output hosted on HuggingFace automatically | You are debugging the pipeline scripts themselves |
| You want to earn credits by submitting benchmark models | |

Remote runs are idempotent and *deduplicated*: if a request with the
same parameters has already succeeded, the backend reuses the existing
dataset and charges zero credits.

## Prerequisites

1. **Python 3.10 or newer**.
2. **The remote extra**:
   ```bash
   pip install "colliderml[remote]"
   ```
   This pulls in `requests`. Everything else (`huggingface_hub`,
   `pyyaml`) is already in the base install.
3. **A HuggingFace account and token** with *read* permissions:
   ```bash
   huggingface-cli login     # interactive
   # or
   export HF_TOKEN=hf_xxxxxxxx
   ```
   The token is used both to authenticate with the backend *and* to
   pull the resulting dataset from HuggingFace.

## Credits

The backend uses a simple credit model:

- **1 credit ≈ 1 node-hour ≈ 100 events at `pu0`** (roughly; the backend
  prices each request based on the pipeline stages it will run)
- **New users get a 10-credit seed grant** on first login
- **Benchmark wins and PRs to the model zoo earn additional credits** —
  see [Benchmark Tasks](./tasks.md)

Check your balance any time:

```python
import colliderml
print(colliderml.balance(), "credits")
```

Or on the CLI:

```bash
colliderml balance
```

## Submitting a job

There are two entry points:

### `colliderml.simulate(remote=True)` — integrated

The drop-in counterpart to [local simulation](./simulation.md). Returns a
`SimulationResult` whose `remote_request_id` is set so you can poll it
later.

```python
import colliderml

result = colliderml.simulate(
    preset="higgs-portal-quick",
    remote=True,
)
print(result.remote_request_id)
```

`simulate()` returns **immediately** after the backend acknowledges the
request. Use the request ID with `colliderml.remote.wait_for()` or the
`colliderml status` CLI to track progress.

### `colliderml.remote.submit(...)` — direct

Lower-level than `simulate()`. You get the full :class:`RemoteSubmission`
object back, with credits charged, estimated node-hours, and the initial
state. Use this when you want tight control over the submit → poll →
download loop.

```python
from colliderml.remote import submit, wait_for

submission = submit(channel="ttbar", events=1000, pileup=200)
print(submission.request_id, submission.credits_charged, "credits charged")

final = wait_for(
    submission.request_id,
    poll_interval=30,
    on_poll=lambda s: print(f"  state={s.state}"),
)
print("done:", final.output_hf_repo)
```

## Polling, status, cancellation

```python
from colliderml.remote import status, wait_for

# One-shot status check
snap = status("req-abc123")
print(snap.state, snap.output_hf_repo)

# Block until done, with progress callback
final = wait_for("req-abc123", poll_interval=60)
```

Terminal states are `completed`, `failed`, and `cancelled`. `wait_for`
raises `RuntimeError` on `failed` / `cancelled` and `TimeoutError` when
a supplied `timeout` elapses.

**Cancellation** — contact the backend admin (see the internal
[admin](../internal/admin.md) runbook) for now; a user-facing
cancellation endpoint is planned.

## Collecting outputs

Completed requests expose their output dataset at:

```
hf.co/datasets/ColliderML/req-<request_id>
```

You can load it with the same convenience you use for the canonical
release:

```python
import colliderml
from colliderml.core.hf_download import DownloadSpec, download_config
from colliderml.core.loader import load_tables
from colliderml.core.data.loader_config import LoaderConfig

# Pull outputs for the completed request into the local cache
download_config(DownloadSpec(
    dataset_id="ColliderML/req-abc123",
    config="ttbar_pu200_tracker_hits",
))

# Then load locally
tables = load_tables(LoaderConfig(
    dataset_id="ColliderML/req-abc123",
    channels="ttbar",
    pileup="pu200",
    objects=["tracker_hits"],
    split="train",
    lazy=False,
))
```

(After commit B4 lands, the one-liner `colliderml.load("ttbar_pu200",
dataset_id="ColliderML/req-abc123")` will subsume all of the above.)

## Configuration

| Env var | Meaning | Default |
| :-- | :-- | :-- |
| `COLLIDERML_BACKEND` | Backend base URL | `https://api.colliderml.com` |
| `HF_TOKEN` | Explicit HuggingFace token (overrides hub login) | — |
| `HUGGING_FACE_HUB_TOKEN` | Fallback env var | — |

Pass `backend_url=` to any client function to override per-call.

## Troubleshooting

**`RuntimeError: Authentication failed`** — the backend rejected your
token. Re-run `huggingface-cli login` and try again. Make sure your
token has at least *read* permissions.

**`RuntimeError: Insufficient credits`** — the backend priced your
request higher than your current balance. Reduce `events`, lower
`pileup`, or earn more credits by submitting a benchmark entry. Run
`colliderml balance` to see what you have.

**`RuntimeError: Duplicate request`** — someone (possibly you) already
ran exactly this request. The backend typically returns the existing
dataset URL in the error detail — use that instead of re-submitting.

**`Could not reach ColliderML backend`** — network, DNS, or the backend
is down. Check the status page linked from the docs home.

**`ImportError: The colliderml.remote client requires the 'remote' extra`**
— run `pip install 'colliderml[remote]'`.

## Next steps

- [Remote API reference](../library/remote.md)
- [Local simulation guide](./simulation.md) — when to stay on your own machine
- [Benchmark tasks](./tasks.md) — earn more credits



========================================
# SOURCE: docs/guide/tasks.md
========================================

# Benchmark Tasks

ColliderML ships a small catalogue of benchmark tasks so that every
model can be compared on an equal footing. Each task defines:

- a **dataset** to draw events from (e.g. ``ttbar_pu200``)
- a **held-out eval range** (event IDs) that participants must not train on
- the **inputs** a model is allowed to see
- a set of **metrics** with directions (higher-is-better or lower-is-better)

This page is the conceptual tour. For the complete API, see
[`colliderml.tasks`](../library/tasks.md).

## The catalogue at a glance

| Task name | Dataset | Inputs | Primary metric | What it measures |
| :-- | :-- | :-- | :-- | :-- |
| `tracking` | `ttbar_pu200` | `tracker_hits` | `trackml_eff` | Track reconstruction quality |
| `jets` | `ttbar_pu0` | `tracks`, `calo_hits` | `btag_auc` | Jet flavour tagging (b / c / light) |
| `anomaly` | mixed SM + BSM | `tracks`, `calo_hits` | `auroc` | BSM event detection against SM background |
| `tracking_latency` | `ttbar_pu200` | `tracker_hits` | `events_per_sec` | Inference throughput for tracking |
| `tracking_small` | `ttbar_pu200` | `tracker_hits` | `trackml_eff` per tier | Tracking under a parameter budget |
| `data_loading` | `ttbar_pu200` | `tracker_hits` | `events_per_sec_local` | Raw data loading throughput |

Get the list programmatically at any time:

```python
import colliderml.tasks
print(colliderml.tasks.list_tasks())
```

## Installing the task runner

The task registry is in the base install, but the reference baselines
(BDT for jets, IsolationForest for anomaly detection) use scikit-learn.
Install the optional extra to get them:

```bash
pip install "colliderml[tasks]"
```

You can evaluate your own predictions against a task without the
`tasks` extra — you only need it if you want to run the shipped
baselines verbatim.

## The scoring contract

Every task exposes three methods on its instance:

1. **`load_eval_inputs()`** — returns the held-out eval data as a dict
   of pyarrow Tables. Users call this to build their inference
   pipeline; the backend calls it to run baselines on demand.
2. **`validate_predictions(preds)`** — raises `ValueError` if your
   submission has the wrong columns, sums, coverage, etc. Run this
   locally before uploading anything.
3. **`score(preds)`** — returns ``{metric_name: value}``.

The top-level helper `colliderml.tasks.evaluate` chains the last two:

```python
import colliderml.tasks

scores = colliderml.tasks.evaluate("tracking", "my_tracks.parquet")
print(scores)
# {'trackml_eff': 0.87, 'fake_rate': 0.03, 'dup_rate': 0.04, 'physics_eff_pt1': 0.92}
```

## Eval splits — "what can I train on?"

Each task withholds a range of event IDs (``eval_event_range``) for
scoring. A valid submission covers *at least half* of that range —
partial submissions are allowed so users can iterate quickly, but full
coverage is required to win a credit award on the leaderboard.

Use the task's own `load()` helper to pull just the eval events:

```python
task = colliderml.tasks.get("tracking")
hits = task.load(tables=["tracker_hits"], event_range=task.eval_event_range)
# hits["tracker_hits"] is a pyarrow Table with events 90_000..100_000
```

For training, load the complement — events *outside* the eval range.
The loader is explicit enough that "this data is safe to train on" is
easy to enforce in your own pipeline.

## Submitting to the leaderboard

Local evaluation gives you a preview. To earn credits and appear on
the public leaderboard, submit to the backend:

```python
import colliderml.tasks

result = colliderml.tasks.submit("tracking", "my_tracks.parquet")
print(result["scores"])         # backend's canonical scores
print(result["credits_earned"]) # non-zero if you beat a previous best
```

Submission requires the `remote` extra (for `requests`) and a
HuggingFace token — see
[Remote Simulation](./remote-simulation.md#prerequisites) for auth
setup. The backend re-scores your submission against its held-out
truth set (which is *not* bundled with the library, so you can't
overfit on it) and awards credits on new bests.

## Running the reference baselines

Each task ships a simple reference baseline you can run as a module:

```bash
# CKF baseline for the tracking task — converts a pipeline run's
# tracksummary_ambi.root into the task's expected parquet schema.
python -m colliderml.tasks.tracking.baselines.ckf \
    --run-dir ./colliderml_output/ttbar_pu200_100evt/runs/0 \
    --output ckf_preds.parquet

# GBDT baseline for jet classification
python -m colliderml.tasks.jets.baselines.bdt \
    --channel ttbar_pu0 --max-events 1000 --output bdt_preds.parquet

# IsolationForest baseline for anomaly detection
python -m colliderml.tasks.anomaly.baselines.isoforest \
    --output iso_preds.parquet
```

The BDT and IsolationForest baselines use scikit-learn; the CKF
baseline only needs a completed pipeline run, which you can produce
with `colliderml simulate` (see [Local Simulation](./simulation.md)).

## Systems tasks vs. physics tasks

The first three tasks (`tracking`, `jets`, `anomaly`) are **physics
tasks** — they measure the quality of a model's output against truth.

The last three (`tracking_latency`, `tracking_small`, `data_loading`)
are **systems tasks** — they measure operational properties (wall-clock
time, parameter count, I/O throughput). Systems tasks exist because
the best scientific model is not always the most useful: a 10 GB
transformer that wins the tracking leaderboard is worse than a 10 kB
network that hits 95% of its score if you actually have to deploy it
at collider trigger latencies.

## What's next

- Browse the [API reference](../library/tasks.md) for every function signature
- Read about [remote simulation](./remote-simulation.md) to generate training data
- Check the live [Leaderboard](https://huggingface.co/spaces/murnanedaniel/colliderml-leaderboard)
  (deployed from this repo's `spaces/leaderboard/`)
- Open a PR to add a new task or baseline — contributors earn credits!



========================================
# SOURCE: docs/library/overview.md
========================================

# Library Overview

ColliderML ships a small Python library (`colliderml`) to make working with the dataset easier and more reproducible.

This section documents the **library features** (CLI, local loading, and utilities). If you only need a quick download, the HuggingFace `datasets` approach in the main docs still works — but the library workflow is often more convenient for analysis.

## Recommended workflow

Most analysis workflows follow this pattern:

- **Load** Parquet shards with the one-liner `colliderml.load(...)` — downloads on first call, then caches.
- **Explode** event-as-row tables into flat (object-as-row) tables.
- **Apply utilities** (pileup subsampling, decay traversal labels, calibration).
- *(optional)* **Simulate** new events yourself when you need more data or a different channel.
- *(optional)* **Score** your model against a built-in benchmark task.

```python
import colliderml
from colliderml.polars import explode_particles

# Download-and-load in one call; frames is dict[str, pl.DataFrame]
frames = colliderml.load("ttbar_pu0", tables=["particles"], max_events=200)
particles_flat = explode_particles(frames["particles"])   # particle-as-row
```

Power users can still reach the low-level config-driven loader directly:

```python
from colliderml.core import load_tables, collect_tables

tables = load_tables(
    {
        "channels": "ttbar",
        "pileup": "pu0",
        "objects": ["particles"],
        "split": "train",
        "lazy": True,
        "max_events": 200,
    }
)
frames = collect_tables(tables)
```

And when you need more events than the released dataset provides,
generate your own:

```python
result = colliderml.simulate(preset="ttbar-quick")
frames = colliderml.load("ttbar_pu0")   # still works — or read result.run_dir directly
```

## What you'll find here

- **CLI**: unified `colliderml` command (download, simulate, balance, …)
- **Loading**: one-line `colliderml.load()` + the config-driven `load_tables` under the hood
- **Exploding**: event-table → flat tables (`explode_*`)
- **Physics utilities**: pileup subsampling, decay traversal labels, calibration
- **Simulate**: run the full simulation pipeline (local or remote)
- **Remote (SaaS)**: HTTP client for the ColliderML backend
- **Tasks (benchmarks)**: built-in benchmark task registry + reference baselines
- **Benchmarks (timing)**: CI warn-only download-speed harness





========================================
# SOURCE: docs/library/cli.md
========================================

# CLI reference

The unified `colliderml` command groups every supported operation
under one entry point. Each subcommand has its own `--help`.

```bash
colliderml --help
colliderml <subcommand> --help
```

## `download` — fetch dataset shards

Download a subset (heuristic shard selection) for a process / pileup / object set:

```bash
colliderml download \
  --channels ttbar \
  --pileup pu0 \
  --objects particles,tracker_hits,calo_hits,tracks \
  --split train \
  --max-events 200
```

`--max-events` is a best-effort way to limit how many shards are
downloaded — the Polars loader then enforces exact event selection
locally. Data goes into `$COLLIDERML_DATA_DIR` (default
`~/.cache/colliderml`) or wherever you point `--out`.

## `list-configs` — inspect the cache or the remote repo

```bash
colliderml list-configs                       # locally cached configs
colliderml list-configs --remote              # query HuggingFace for what exists
```

## `simulate` — run the pipeline

New in v0.4.0. Runs the full Pythia / MadGraph / Geant4 / ACTS
pipeline, either locally inside a Docker or Podman container or via
the SaaS backend.

```bash
# Named preset, local container run (needs Docker/Podman):
colliderml simulate --preset ttbar-quick --local

# Explicit parameters, writes into ./runs/hp10/
colliderml simulate \
  --channel higgs_portal --events 10 --pileup 10 \
  --local --output ./runs/hp10

# Submit to the SaaS backend (needs an HF token):
colliderml simulate --preset ttbar-benchmark --remote
```

Options:

| Flag | Meaning |
| :-- | :-- |
| `--preset <name>` | Named preset; sets channel/events/pileup if not supplied. |
| `--channel <name>` | Physics channel (`ttbar`, `higgs_portal`, `single_muon`, …). |
| `--events <N>` | Number of events to generate. |
| `--pileup <N>` | Pileup level (0 for no pileup, 200 for HL-LHC). |
| `--seed <N>` | Random seed forwarded to every stage. Default: 42. |
| `--output <dir>` | Host output directory. Defaults to `./colliderml_output/<channel>_pu<pileup>_<events>evt`. |
| `--image <tag>` | Override the container image. Defaults to the pinned ODD sw image. |
| `--local` / `--remote` | Mutually exclusive; default is local if a runtime is available. |
| `--run-id <name>` | Subdirectory under `runs/`. Default: `0`. |
| `--quiet` | Suppress per-stage progress messages. |

See the [Local Simulation](../guide/simulation.md) and
[Remote Simulation](../guide/remote-simulation.md) guides for the full
story.

## `list-presets` — print the bundled preset catalogue

```bash
colliderml list-presets
```

Outputs one line per preset with its name, channel, event count,
pileup, and description. See
[`colliderml.simulate.load_presets`](./simulate.md#list_presets) for
the API equivalent.

## `balance` — show your credit balance

New in v0.4.0. Queries the SaaS backend's `/v1/me` endpoint and prints
your credit balance. Requires the `[remote]` extra and a HuggingFace
token.

```bash
colliderml balance
# user:    alice
# credits: 42.00

colliderml balance --json    # also print the raw /v1/me body
```

## `status` — check a remote request

New in v0.4.0. Fetches a remote simulation request's state from
`/v1/requests/<request-id>`.

```bash
colliderml status req-abc123
# request_id:  req-abc123
# state:       running
# channel:     ttbar
# events:      1000
# pileup:      200

colliderml status req-abc123 --json   # full backend response body
```

## Cache location

- By default, data is stored in `~/.cache/colliderml`.
- Override via `$COLLIDERML_DATA_DIR` or `--out /path/to/cache` on the
  `download` / `list-configs` subcommands.
- The `simulate` subsystem uses a separate subdirectory
  (`$COLLIDERML_CACHE/simulate/`, default
  `~/.cache/colliderml/simulate/`) for its container cache, the
  auto-cloned production repo, and the Geant4 physics tables.





========================================
# SOURCE: docs/library/loading.md
========================================

# Loading: Config-driven Local Parquet

The ColliderML loader reads **local Parquet shards** (downloaded via the CLI) using Polars.

## Load tables

Use `colliderml.core.load_tables` to read a set of object tables from the local cache.

```python
from colliderml.core import load_tables, collect_tables

cfg = {
    "dataset_id": "CERN/ColliderML-Release-1",
    "channels": "ttbar",
    "pileup": "pu0",
    "objects": ["particles", "tracker_hits", "calo_hits", "tracks"],
    "split": "train",
    "lazy": True,
    "max_events": 200,
}

tables = load_tables(cfg)      # dict[str, pl.DataFrame | pl.LazyFrame]
frames = collect_tables(tables)  # dict[str, pl.DataFrame]
```

## Key config fields

- `channels`: string or list (or `"all"`)
- `pileup`: `"pu0"`, `"pu200"`, etc.
- `objects`: `["particles", "tracker_hits", "calo_hits", "tracks"]`
- `split`: typically `"train"`
- `lazy`: `True` to `scan_parquet`, `False` to `read_parquet`
- `max_events`: exact event limit enforced locally across all tables
- `data_dir`: optional override for cache directory (otherwise uses defaults / env var)

## Event selection (`max_events`)

The loader enforces a consistent event selection across objects:

- Select `event_id`s from a reference table (by default `particles`)
- Filter all other tables to those `event_id`s

This means you can download full shards but still work with a deterministic, small event slice.





========================================
# SOURCE: docs/library/exploding.md
========================================

# Exploding: Event Tables → Flat Tables

ColliderML stores each object collection as an **event table**:

- one row per `event_id`
- many columns are lists (one entry per particle/hit/cell in that event)

The `colliderml.polars` helpers “explode” these list columns into a flat, object-as-row table.

## Recommended pattern

Filter to the event(s) you care about **before** exploding:

```python
import polars as pl
from colliderml.polars import explode_particles

evt = frames["particles"].filter(pl.col("event_id") == event_id)
particles_flat = explode_particles(evt)
```

## Particles

```python
from colliderml.polars import explode_particles

particles_flat = explode_particles(frames["particles"])
```

Output includes:

- `event_id`
- `particle_index` (per-event row index)
- particle columns (scalar per row, previously list elements)

## Tracker hits

```python
from colliderml.polars import explode_tracker_hits

hits_flat = explode_tracker_hits(frames["tracker_hits"])
```

## Calorimeter hits (cells + contributions)

Calorimeter hits have nested contribution structure. Use:

```python
from colliderml.polars import explode_calo_cells_and_contribs

cells, contribs = explode_calo_cells_and_contribs(frames["calo_hits"])
```

- `cells`: one row per calorimeter cell hit
- `contribs`: one row per contribution (cell position repeated per contrib)






========================================
# SOURCE: docs/library/physics.md
========================================

# Physics utilities

These utilities operate on ColliderML tables (Polars DataFrame/LazyFrame) and are designed to preserve physical consistency.

## Pileup subsampling

Subsample full-pileup events down to a target number of vertices.

Semantics:

- `vertex_primary` labels vertices as integers `1..N` per event
- To target pileup `K`, remove vertices with `vertex_primary > K`
- Remove particles and calo contributions from removed vertices

```python
from colliderml.physics import subsample_pileup

tables_sub = subsample_pileup(tables, target_vertices=50)
```

## Decay traversal labels

Assign a per-particle label identifying the primary ancestor in the decay chain:

```python
from colliderml.physics import assign_primary_ancestor

particles_labeled = assign_primary_ancestor(frames["particles"])
```

## Calorimeter calibration

Calibration can be applied to calo hit energies (and optionally contribution energies).

### Default ODD calibration

```python
from colliderml.physics.calibration import apply_calo_calibration, odd_default_calo_calibration

calo_cal = apply_calo_calibration(frames["calo_hits"], odd_default_calo_calibration())
```

### YAML-based calibration

You can also provide a YAML mapping keyed by detector IDs or region names:

```yaml
detector_scale:
  ecal_barrel: 37.5
  ecal_endcap: 38.7
  hcal_barrel: 45.0
  hcal_endcap: 46.9
default_scale: 1.0
apply_to_contrib: true
```

```python
from colliderml.physics.calibration import apply_calo_calibration

calo_cal = apply_calo_calibration(frames["calo_hits"], "calibration.yaml")
```





========================================
# SOURCE: docs/library/simulate.md
========================================

# `colliderml.simulate`

API reference for the simulation subsystem. For the conceptual overview
(container runtimes, caching, the pipeline stages) see
[Local Simulation](../guide/simulation.md); for the SaaS variant see
[Remote Simulation](../guide/remote-simulation.md).

## Top-level entry point

### `simulate(...)` → `SimulationResult`

```python
colliderml.simulate(
    *,
    preset: str | Preset | None = None,
    channel: str | None = None,
    events: int | None = None,
    pileup: int | None = None,
    seed: int = 42,
    output_dir: str | Path | None = None,
    image: str = DEFAULT_IMAGE,
    remote: bool = False,
    run_id: str = "0",
    prod_root: str | Path | None = None,
    runtime: str | None = None,
    quiet: bool = False,
) -> SimulationResult
```

Runs the full pipeline for one (channel, events, pileup) configuration.

**Parameter semantics**

- `preset` — name of a bundled preset (see
  [`list_presets()`](#list_presets)), or a [`Preset`](#preset-dataclass)
  instance. Supplies defaults for `channel`, `events`, and `pileup`.
- Explicit `channel` / `events` / `pileup` **always win** over the preset's
  values. Pass just one (e.g. `pileup=40`) to tweak a preset without
  defining a new one.
- `output_dir` defaults to
  `./colliderml_output/<channel>_pu<pileup>_<events>evt/`.
- `seed` is forwarded to every stage so the full run is reproducible.
- `image` pins the OpenDataDetector software image. Leave at the default
  unless you are rebuilding the image yourself.
- `remote=True` submits to the SaaS backend instead of running locally —
  requires `pip install "colliderml[remote]"` and a valid HuggingFace
  token. See [`colliderml.remote`](./remote.md) for the SaaS client.
- `run_id` names the subdirectory under `output_dir/runs/`. The default
  `"0"` is fine for single-run workflows; bump it for parameter sweeps.
- `prod_root` lets advanced callers point at a pre-existing
  `colliderml-production` checkout. Leave `None` for the default
  auto-clone behaviour.
- `runtime` forces a container runtime (`"docker"` or `"podman"`);
  auto-detected otherwise.
- `quiet=True` suppresses per-stage progress messages.

**Errors**

- `ValueError` — neither preset nor explicit channel is supplied, or the
  channel is unknown.
- `ContainerRuntimeError` — no container runtime is available and
  `remote=False`.
- `RuntimeError` — a pipeline stage exited non-zero.

**Example**

```python
import colliderml

result = colliderml.simulate(
    preset="higgs-portal-quick",
    output_dir="runs/hp10",
    seed=1234,
)
print(result.run_dir)
for fp in result.list_files():
    print(" ", fp.name)
```

## `SimulationResult` dataclass

```python
@dataclass
class SimulationResult:
    channel: str
    events: int
    pileup: int
    output_dir: Path
    run_dir: Path
    stages: list[StageRun]
    remote_request_id: str | None = None
```

| Field | Meaning |
| :-- | :-- |
| `channel` | Physics channel that was simulated. |
| `events` | Requested event count (actual may differ slightly for some channels). |
| `pileup` | Pileup level. |
| `output_dir` | The directory you passed (or the computed default). |
| `run_dir` | `<output_dir>/runs/<run_id>/` — the per-run artefact directory. |
| `stages` | One `StageRun(name, stage, returncode)` per executed stage. |
| `remote_request_id` | Only set when `remote=True`; the backend's request ID. |

Methods:

- `list_files()` → sorted `list[Path]` of every file under `run_dir`.

## `Preset` dataclass

```python
@dataclass(frozen=True)
class Preset:
    name: str
    channel: str
    events: int
    pileup: int
    description: str = ""
```

Helper methods:

- `as_dict()` — plain dict view for JSON/YAML serialisation.

## `load_presets() -> dict[str, Preset]` { #list_presets }

Loads the bundled preset catalogue from the package's `presets.yaml`. The
file ships as package data, so it works in editable installs, wheels, and
sdists alike.

```python
from colliderml.simulate import load_presets
catalogue = load_presets()
for name, preset in sorted(catalogue.items()):
    print(f"{name:25s} {preset.channel:15s} events={preset.events:>6}  pileup={preset.pileup}")
```

## `resolve_preset(name, presets=None) -> Preset`

Looks up a single preset by name. Raises `ValueError` with the full list
of available names on a miss — so typos surface immediately.

## Auto-cloning the production repo

The pipeline scripts, YAML stage configs, and container bootstrap script
live in [`OpenDataDetector/colliderml-production`](https://github.com/OpenDataDetector/colliderml-production).
The public library clones that repository on first use into a cache
directory and mounts it inside the container as `/workspace`.

```python
from colliderml.simulate.docker import (
    clone_colliderml_production,
    default_cache_root,
    COLLIDERML_PRODUCTION_REF,
)

print("cache root:", default_cache_root())
print("pinned ref:", COLLIDERML_PRODUCTION_REF)

# Idempotent; just fetches if the clone already exists.
prod = clone_colliderml_production()
print("cloned to:", prod)
```

**Overriding the pinned ref** — set the `COLLIDERML_PRODUCTION_REF`
environment variable before calling `simulate()`. This is the supported
way for pipeline developers to test an in-flight branch without editing
the library.

**Forcing a re-clone** — pass `force_refresh=True` to
`clone_colliderml_production`, or delete the cache directory and let
`simulate()` rebuild it.

## Container runtime helpers

All exported from `colliderml.simulate.docker`:

| Function | Purpose |
| :-- | :-- |
| `get_container_runtime()` | Return `"docker"` or `"podman"`; raise `ContainerRuntimeError` if neither is installed. |
| `check_runtime_available(runtime=None)` | Verify the runtime's daemon is actually reachable (`runtime info`). |
| `check_image_available(image, *, runtime=None)` | Return `True` iff the image is already pulled locally. |
| `pull_image(image, *, interactive=True, runtime=None)` | Pull the image, prompting before the ~10 GB download when running on a tty. |
| `default_cache_root()` | `$COLLIDERML_CACHE`/`simulate` or `~/.cache/colliderml/simulate` if unset. |
| `clone_colliderml_production(cache_root=None, *, ref=None, force_refresh=False)` | Clone or refresh the production repo at the pinned ref. |
| `run_pipeline(...)` | Low-level multi-stage runner used by `simulate()`. |

You rarely need to call these directly — they are documented here so
that users debugging CI failures or writing their own orchestrators
can see the primitives.

## Pipeline introspection

`colliderml.simulate.pipeline` exposes the channel→stages mapping that
`simulate()` uses internally. Useful when you want to reason about what
would run *before* spinning up a container:

```python
from colliderml.simulate.pipeline import (
    CHANNEL_STAGES,
    get_channel_stages,
    list_channels,
)

print(list_channels())
for stage in get_channel_stages("ttbar"):
    print(f"  {stage.name:35s} ({stage.script})")
```

And to preview the manifest that will be handed to the container runner:

```python
from pathlib import Path
from colliderml.simulate.docker import clone_colliderml_production
from colliderml.simulate.pipeline import generate_stage_manifest

prod = clone_colliderml_production()
for step in generate_stage_manifest("higgs_portal", prod_root=prod):
    print(step["name"], "→", step["config_path"])
```

## See also

- [Local Simulation guide](../guide/simulation.md) — conceptual overview
- [Remote Simulation guide](../guide/remote-simulation.md) — SaaS variant
- [`colliderml.load()`](./loading.md) — load simulation output
- [Benchmark tasks](./tasks.md) — score simulated events against a task



========================================
# SOURCE: docs/library/remote.md
========================================

# `colliderml.remote`

API reference for the ColliderML SaaS backend client. For the conceptual
overview — credits, auth, when to use remote vs. local — see
[Remote Simulation](../guide/remote-simulation.md).

This module lives behind the **`remote` extra**:

```bash
pip install "colliderml[remote]"
```

The only additional runtime dependency is `requests`; everything else is
already in the base install.

## Public surface

All of these are importable directly from `colliderml.remote`:

| Symbol | Kind | Purpose |
| :-- | :-- | :-- |
| [`submit`](#submit) | function | `POST /v1/simulate` |
| [`status`](#status) | function | `GET /v1/requests/{id}` |
| [`wait_for`](#wait_for) | function | submit-and-poll loop |
| [`get_me`](#get_me) | function | `GET /v1/me` |
| [`balance`](#balance) | function | convenience wrapper around `get_me()` |
| [`RemoteSubmission`](#remotesubmission-dataclass) | dataclass | snapshot of a request |
| `DEFAULT_BACKEND_URL` | constant | base URL (overridable) |
| `POLL_INTERVAL_SECONDS` | constant | default poll cadence for `wait_for` |

## `submit(...)` → `RemoteSubmission` { #submit }

```python
colliderml.remote.submit(
    *,
    channel: str,
    events: int,
    pileup: int,
    seed: int = 42,
    backend_url: str | None = None,
) -> RemoteSubmission
```

Submits a new simulation request. Returns immediately after the backend
acknowledges the submission — to block until the job is done, follow up
with [`wait_for`](#wait_for).

**Behaviour**

- `HF_TOKEN` (or the `huggingface-cli` saved token) is used as the
  bearer credential.
- If the backend recognises the request as an exact duplicate of a
  completed run, the returned `RemoteSubmission` is populated from the
  cached result and no new compute is scheduled. A reuse notice is
  printed to stderr.
- Non-2xx responses are translated into `RuntimeError` with an
  actionable message: 401 points users at `huggingface-cli login`, 402
  surfaces the backend's "insufficient credits" detail verbatim, 409 is
  reported as a duplicate, and everything else is a generic
  "Backend error {code}".

**Raises**

- `RuntimeError` — auth, credits, or backend error.
- `ImportError` — the `remote` extra is not installed.

## `status(request_id, *, backend_url=None)` → `RemoteSubmission` { #status }

Fetches the current state of an existing request. The returned object is
a **new snapshot**; it does not share identity with any previous
`RemoteSubmission` for the same request.

```python
from colliderml.remote import status

snap = status("req-abc123")
if snap.is_terminal:
    print(snap.output_hf_repo)
```

## `wait_for(...)` { #wait_for }

```python
colliderml.remote.wait_for(
    request_id: str,
    *,
    backend_url: str | None = None,
    timeout: float | None = None,
    poll_interval: float = POLL_INTERVAL_SECONDS,
    on_poll: Callable[[RemoteSubmission], None] | None = None,
) -> RemoteSubmission
```

Polls `request_id` until the backend reports a terminal state
(`completed`, `failed`, or `cancelled`). Returns the final snapshot, or
raises:

- `TimeoutError` if `timeout` seconds elapse before completion.
- `RuntimeError` if the request ends in `failed` or `cancelled`.

**Progress reporting** — pass `on_poll=` to receive a callback with the
latest `RemoteSubmission` on every non-terminal poll. Useful for CLI
progress lines:

```python
wait_for(rid, on_poll=lambda s: print(f"  state={s.state}"))
```

## `get_me(*, backend_url=None)` → `dict` { #get_me }

Returns the authenticated user's profile. Shape (defined by the backend):

```json
{
  "hf_username": "alice",
  "credits": 42.5,
  "created_at": "2025-04-01T12:34:56Z",
  "total_requests": 7,
  "completed_requests": 5
}
```

## `balance(*, backend_url=None)` → `float` { #balance }

Thin wrapper around `get_me()["credits"]`. Used by the CLI
`colliderml balance` command and re-exported as `colliderml.balance`
(after commit B4).

```python
import colliderml
print(colliderml.balance(), "credits remaining")
```

## `RemoteSubmission` dataclass { #remotesubmission-dataclass }

```python
@dataclass
class RemoteSubmission:
    request_id: str
    state: str                       # submitted | queued | running | completed | failed | cancelled
    channel: str
    events: int
    pileup: int
    credits_charged: float = 0.0
    estimated_node_hours: float = 0.0
    estimated_completion_seconds: int = 0
    output_hf_repo: str | None = None
    backend_url: str = DEFAULT_BACKEND_URL
```

Properties and methods:

- `.is_terminal` → `bool`. `True` when `state` ∈ {`completed`,
  `failed`, `cancelled`}.
- `.refresh()` → mutates in-place with the latest `status()` and
  returns `self`.

The `raw` field (hidden from `repr`) stores the backend's full response
body for forward-compatibility — new fields the backend starts sending
show up there before they're promoted to dedicated attributes.

## Environment variables

| Name | Meaning | Default |
| :-- | :-- | :-- |
| `COLLIDERML_BACKEND` | Backend base URL (overridden by `backend_url=` keyword) | `https://api.colliderml.com` |
| `HF_TOKEN` | HuggingFace API token | — |
| `HUGGING_FACE_HUB_TOKEN` | Fallback HF token env var | — |

## Authentication

The client never asks for a password. Token resolution follows this
order, first hit wins:

1. `HF_TOKEN` env var
2. `HUGGING_FACE_HUB_TOKEN` env var
3. [`huggingface_hub.get_token()`](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/authentication#huggingface_hub.get_token)
   — the modern helper (hub ≥ 0.20)
4. [`HfFolder.get_token()`](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/authentication#huggingface_hub.HfFolder.get_token)
   — legacy fallback for older hub versions

No token → `RuntimeError` with a one-line recipe for fixing it.

## See also

- [Remote Simulation guide](../guide/remote-simulation.md)
- [`colliderml.simulate`](./simulate.md) — integrated entry point that
  drives `submit()` under the hood when `remote=True`.
- [`colliderml.load`](./loading.md) — load per-request HF datasets
  produced by completed runs.



========================================
# SOURCE: docs/library/tasks.md
========================================

# `colliderml.tasks`

API reference for the benchmark task registry. For a conceptual tour
(what each task measures, eval splits, leaderboard workflow), see
[Benchmark Tasks](../guide/tasks.md).

Install the optional extra if you want the reference baselines:

```bash
pip install "colliderml[tasks]"
```

The registry itself and local scoring work with the base install —
only the BDT and IsolationForest baselines require scikit-learn.

## Registry

### `list_tasks() -> list[str]`

Returns every registered task name, sorted alphabetically.

```python
>>> import colliderml.tasks
>>> colliderml.tasks.list_tasks()
['anomaly', 'data_loading', 'jets', 'tracking', 'tracking_latency', 'tracking_small']
```

### `get(name: str) -> BenchmarkTask`

Returns a fresh instance of the task registered under ``name``.
Raises `ValueError` with the full list of available names on a miss.

### `register(cls: type[BenchmarkTask]) -> type[BenchmarkTask]`

Class decorator that registers a subclass of
[`BenchmarkTask`](#benchmarktask-abc). Used internally by each task
module; external code can also use it to register custom tasks in a
plugin. Rejects duplicate `name` values so accidental shadowing
surfaces as an `ValueError` at import time.

```python
from colliderml.tasks import BenchmarkTask, register

@register
class MyTask(BenchmarkTask):
    name = "my_task"
    dataset = "ttbar_pu0"
    eval_event_range = (95_000, 100_000)
    inputs = ["tracks"]
    metrics = ["my_metric"]
    higher_is_better = {"my_metric": True}

    def load_eval_inputs(self):
        return self.load(tables=self.inputs, event_range=self.eval_event_range)

    def validate_predictions(self, preds):
        ...

    def score(self, preds):
        return {"my_metric": 0.42}
```

## Evaluation

### `evaluate(task_name, predictions) -> dict[str, float]`

Runs `validate_predictions()` and `score()` on the registered task.

```python
scores = colliderml.tasks.evaluate("tracking", "my_tracks.parquet")
```

`predictions` accepts:

- a path (`str` or `pathlib.Path`) to a parquet file
- an existing `pyarrow.Table`
- a `pandas.DataFrame`

Anything else raises `TypeError`.

### `submit(task_name, predictions, *, backend_url=None) -> dict`

Uploads predictions to the backend leaderboard and returns the
backend's response (typically `{'scores': ..., 'credits_earned': ...}`).

Requires the `remote` extra (`pip install 'colliderml[remote]'`) and a
HuggingFace token. See [`colliderml.remote`](./remote.md) for auth
details. On a non-2xx response the function raises `RuntimeError` with
the backend's error message verbatim.

The submission flow:

1. Coerce `predictions` to a pyarrow Table.
2. Validate locally (raises `ValueError` on bad submissions so you
   don't waste bandwidth).
3. Score locally and include the preview in the multipart body.
4. POST the parquet bytes to
   `POST /v1/benchmark/<task_name>/submit` with the HF bearer token.
5. Return the backend's JSON body.

## `BenchmarkTask` ABC { #benchmarktask-abc }

Base class for every task. Subclasses set the following class
attributes:

| Attribute | Type | Meaning |
| :-- | :-- | :-- |
| `name` | `str` | Identifier used in URLs, the CLI, the leaderboard, and the registry. |
| `dataset` | `str` | `<channel>_<pileup>` key like `"ttbar_pu200"`. |
| `eval_event_range` | `tuple[int, int]` | Half-open range of event IDs withheld for scoring. |
| `inputs` | `list[str]` | Table names the task exposes to the model. |
| `metrics` | `list[str]` | Metric names in display order (first is primary). |
| `higher_is_better` | `dict[str, bool]` | Per-metric direction. Defaults to `True`. |

Subclasses must implement:

- `load_eval_inputs() -> dict[str, pa.Table]` — return the held-out
  eval data as pyarrow Tables.
- `validate_predictions(preds: pa.Table) -> None` — raise `ValueError`
  if the submission is malformed.
- `score(preds: pa.Table) -> dict[str, float]` — compute every metric.

The base class provides a `load()` helper that subclasses use inside
`load_eval_inputs()` and `score()`:

```python
def load(
    self,
    *,
    tables: list[str],
    dataset: str | None = None,
    max_events: int | None = None,
    event_range: tuple[int, int] | None = None,
) -> dict[str, pa.Table]: ...
```

`dataset` defaults to `self.dataset` — override it only for tasks
like anomaly detection that pull from multiple datasets. The helper
auto-downloads missing parquets via
[`colliderml.core.hf_download`](./loading.md) before delegating to
[`colliderml.core.loader.load_tables`](./loading.md).

`is_better(metric, new, current) -> bool` on the base class honours
`higher_is_better` so the leaderboard can decide whether a submission
beats a prior best.

## Per-task details

### `tracking` (`TrackingTask`)

- **Dataset**: `ttbar_pu200`
- **Eval range**: `(90_000, 100_000)`
- **Inputs**: `tracker_hits`
- **Metrics**: `trackml_eff`, `fake_rate`, `dup_rate`, `physics_eff_pt1`
- **Higher is better**: `trackml_eff`, `physics_eff_pt1`
- **Schema required on preds**: `event_id`, `hit_id`, `track_id`

The primary metric is TrackML weighted efficiency: the sum of hit
weights across all correctly reconstructed tracks over the total hit
weight. A track is "correct" when (a) a majority of its hits come
from one particle, and (b) that particle contributes ≥ 50% of its own
hits to the track.

### `jets` (`JetClassificationTask`)

- **Dataset**: `ttbar_pu0`
- **Eval range**: `(90_000, 100_000)`
- **Inputs**: `tracks`, `calo_hits`
- **Metrics**: `btag_auc`, `light_rej_70`, `c_rej_70`
- **Schema required on preds**: `event_id`, `jet_id`, `prob_b`,
  `prob_c`, `prob_light` (probabilities must sum to 1 per row, ±0.02)

The local `score()` currently synthesises a deterministic truth label
for preview purposes. The backend scorer overrides this with the
canonical truth file when you submit via `submit()`.

### `anomaly` (`AnomalyDetectionTask`)

- **Dataset**: virtual `mixed_sm_bsm` (draws from 3 SM + 4 BSM channels)
- **Eval range**: `(0, 10_000)` per channel
- **Inputs**: `tracks`, `calo_hits`
- **Metrics**: `auroc`, `sig_eff_1fpr`
- **Schema required on preds**: `event_id`, `channel`, `anomaly_score`

SM channels: `ttbar_pu0`, `zmumu_pu0`, `zee_pu0`.
BSM channels: `higgs_portal_pu0`, `susy_gmsb_pu0`, `hidden_valley_pu0`,
`zprime_pu0`.

`load_eval_inputs()` silently skips channels that aren't cached
locally — on a developer laptop that has only downloaded some of the
BSM datasets, the task still works with whatever is present.

### `tracking_latency` (`TrackingLatencyTask`)

- **Dataset**: `ttbar_pu200`
- **Eval range**: `(99_000, 100_000)` (1000 events)
- **Metrics**: `wallclock_s`, `events_per_sec`
- **Schema required**: 1-row table with `wallclock_s`, `n_events`

Users report the time they measured; the backend re-runs the code in
a standardised container for canonical numbers.

### `tracking_small` (`TrackingSmallModelTask`)

- **Dataset**: `ttbar_pu200`
- **Eval range**: `(90_000, 100_000)`
- **Metrics**: `trackml_eff`, `n_params`, `tier`
- **Schema required**: tracking predictions + a constant `n_params` column

Parameter tiers: tier 1 < 10k, tier 2 < 100k, tier 3 < 1M, tier 4 is
"does not qualify". The leaderboard sorts by tier first (smaller
tiers first), then by TrackML efficiency within a tier.

### `data_loading` (`DataLoadingTask`)

- **Dataset**: `ttbar_pu200`
- **Eval range**: `(0, 10_000)`
- **Metrics**: `events_per_sec_local`, `events_per_sec_streaming`
- **Schema required**: 1-row table with `local_seconds`,
  `streaming_seconds`, `n_events`

## Reference baselines

Three reference baselines ship with the library, runnable as Python
modules:

| Baseline | Module | Dependency | Purpose |
| :-- | :-- | :-- | :-- |
| CKF (tracking) | `colliderml.tasks.tracking.baselines.ckf` | — | Converts a pipeline run's tracksummary into the task's schema |
| GBDT (jets) | `colliderml.tasks.jets.baselines.bdt` | scikit-learn | Shallow gradient-boosted trees on per-event kinematics |
| IsoForest (anomaly) | `colliderml.tasks.anomaly.baselines.isoforest` | scikit-learn | Unsupervised anomaly detection on per-event features |

All three expose a `main()` entry point and a `--help` flag. The
scikit-learn dependency is gated inside the baseline scripts — the
main `tasks` package stays importable even if scikit-learn is not
installed.

## Internal helpers (rarely needed)

- `colliderml.tasks.parse_dataset_name(name)` — split
  `"ttbar_pu200"` into `("ttbar", "pu200")`.
- `colliderml.tasks.load_task_data(dataset, *, tables, max_events=None, event_range=None)`
  — the free-function that `BenchmarkTask.load()` calls under the hood.

## See also

- [Benchmark Tasks guide](../guide/tasks.md) — conceptual overview
- [`colliderml.load`](./loading.md) — the low-level loader used by tasks
- [`colliderml.remote`](./remote.md) — the SaaS client used by `submit()`
- [Leaderboard Space](https://huggingface.co/spaces/murnanedaniel/colliderml-leaderboard) — live rankings



========================================
# SOURCE: docs/library/benchmarks.md
========================================

# Benchmarks

Benchmarks live under the top-level `benchmarks/` directory and focus on **download speed**.

## Download benchmark runner

Run locally:

```bash
python benchmarks/download_benchmark.py --help
```

Typical usage benchmarks a small, deterministic subset (e.g. a fixed set of configs and one shard each) and writes machine-readable JSON output.

## CI integration

GitHub Actions runs a lightweight benchmark in **warn-only** mode to detect large regressions without flaking CI due to network variability.

See:

- `benchmarks/README.md`
- `.github/workflows/tests.yml`





========================================
# SOURCE: docs/spaces/overview.md
========================================

# HuggingFace Spaces

ColliderML ships four public HuggingFace Spaces, each a lightweight
Gradio frontend for a different slice of the platform. The code lives
under [`spaces/`](https://github.com/OpenDataDetector/ColliderML/tree/main/spaces)
in this repository and is auto-synced to HuggingFace on every push to
`main` (staging previews sync on every push to `staging`) via the
`sync-spaces.yml` workflow.

## The four Spaces

| Space | What it does | Needs login? |
| :-- | :-- | :-- |
| [Event Display](./event-display.md) | Interactive 3D viewer for single events | No |
| [Simulation Form](./simulation-form.md) | Submit custom simulations via form or chat agent | HuggingFace OAuth |
| [Leaderboard](./leaderboard.md) | Browse + submit to benchmark task leaderboards | HuggingFace OAuth for submission |
| [Model Zoo](./model-zoo.md) | Browse HF models tagged `colliderml` | No |

A fifth Space — `colliderml-admin` — is **private** and lives in the
[`colliderml-production`](https://github.com/OpenDataDetector/colliderml-production)
repository instead. It exposes operator controls (credit grants, user
bans, kill switch) and is gated behind a separate admin token.

## How the Spaces fit the platform

```
┌──────────────────────┐         ┌──────────────────────┐
│  colliderml pip pkg  │         │  HuggingFace Spaces  │
│  (this repo)         │         │  (this repo, synced) │
└──────────┬───────────┘         └──────────┬───────────┘
           │                                │
           │                                │  HF OAuth → bearer token
           │                                │
           ▼                                ▼
       ┌──────────────────────────────────────┐
       │   ColliderML Backend (FastAPI)       │
       │   (colliderml-production repo)       │
       │                                      │
       │   POST /v1/simulate                  │
       │   GET  /v1/me, /v1/requests/{id}     │
       │   POST /v1/benchmark/{task}/submit   │
       │   GET  /v1/leaderboard/{task}        │
       └──────────────────┬───────────────────┘
                          │
                          ▼
                ┌─────────────────┐
                │  NERSC SFAPI    │
                │  Perlmutter     │
                └─────────────────┘
```

The Spaces **never talk to NERSC directly** — every button in every Space
is a thin wrapper around a backend HTTP endpoint. The backend does all
the authentication, quota enforcement, credit accounting, and actual
SFAPI dispatch.

This means the same features are available whether you reach for the
pip package or a Space: `colliderml simulate --remote` and the
Simulation Form Space both POST to `/v1/simulate`; `colliderml.tasks.submit`
and the Leaderboard Space both POST to `/v1/benchmark/<task>/submit`.

## Embedding a Space in your own site

HuggingFace Spaces have a built-in iframe embed:

```html
<iframe
    src="https://huggingface.co/spaces/murnanedaniel/colliderml-event-display"
    frameborder="0"
    width="850"
    height="450">
</iframe>
```

The four public Spaces carry `hf_oauth: true` where needed and an
Apache-2.0 license, so they're safe to embed in papers, blog posts, or
lab pages. Please keep the attribution link visible.

## Running a Space locally

Each Space is a self-contained Gradio app:

```bash
cd spaces/event-display
pip install -r requirements.txt
python app.py
# Open http://localhost:7860
```

For Spaces that need backend auth (`simulation-form`, `leaderboard`), set
`COLLIDERML_BACKEND=http://localhost:8000` to point at a local backend
checkout, and supply a HuggingFace token via env var instead of OAuth.

## Deployment workflow

`.github/workflows/sync-spaces.yml` (landing in commit B7 of the v0.4.0
migration) matrix-builds every directory under `spaces/` and pushes it
to the corresponding HuggingFace Space via `huggingface_hub.upload_folder`.
On `staging` pushes the Spaces are synced to suffixed variants
(`-staging`) so you can preview changes without clobbering the
production Space.

See the individual Space pages for each one's specifics.



========================================
# SOURCE: docs/spaces/event-display.md
========================================

# Event Display

**Live:** [huggingface.co/spaces/murnanedaniel/colliderml-event-display](https://huggingface.co/spaces/murnanedaniel/colliderml-event-display)
**Source:** [`spaces/event-display/`](https://github.com/OpenDataDetector/ColliderML/tree/main/spaces/event-display)

Interactive 3D viewer for single events from the ColliderML datasets.
Pick a process and an event ID to see the tracker hits, reconstructed
tracks, and truth particles inside the OpenDataDetector geometry —
rendered with Plotly so you can rotate, zoom, and toggle layers.

## What you can see

- **Blue points** — tracker hits, coloured by detector layer (or
  `volume_id` / `particle_id` if layer is missing).
- **Red lines** — reconstructed tracks, approximated as helical
  polylines from the perigee parameters (`d0`, `z0`, `phi`, `theta`,
  `qop`). This is a visualisation aid, not a physics-accurate
  reconstruction.
- **Yellow dashes** — truth-particle momentum vectors drawn from the
  primary vertex, restricted to primary particles and capped at the 20
  highest-momentum entries so the plot stays readable.

## Datasets

The dropdown ships with a curated subset of the catalogue so the UI
stays responsive on the Space's modest resources:

```
ttbar_pu0
ttbar_pu200
higgs_portal_pu10
zmumu_pu0
diphoton_pu0
single_muon_pu0
```

Each selection loads at most 50 events into memory. If you want a
different process, open an issue — or run the Space locally and edit
`DATASETS` in `app.py`.

## Data loading

The Space prefers its own on-disk cache
(`_cached_events/<dataset>/*.parquet`) if present; `cache_events.py`
populates it by calling `colliderml.load()` once for each dataset and
writing the results to parquet. On a fresh Space with an empty cache
the app falls back to calling `colliderml.load()` live from the
dataset's HuggingFace repo — slower on first hit, then cached in
memory via `functools.lru_cache`.

## Running it locally

```bash
cd spaces/event-display
pip install -r requirements.txt
python app.py
# Open http://localhost:7860
```

## Customisation

- Add a dataset → append to `DATASETS` in `app.py` and re-run
  `cache_events.py`.
- Add a visualisation layer (e.g. calo cells) → follow the pattern for
  `tracker_hits` in `render_event()` and add a new `colliderml.load`
  table to the list in `_load_dataset()`.
- Change the colour scheme → the Plotly `Scatter3d` traces carry all
  the knobs; search for `Viridis` and `crimson` in `app.py`.



========================================
# SOURCE: docs/spaces/simulation-form.md
========================================

# Simulation Form

**Live:** [huggingface.co/spaces/murnanedaniel/colliderml-simulation-form](https://huggingface.co/spaces/murnanedaniel/colliderml-simulation-form)
**Source:** [`spaces/simulation-form/`](https://github.com/OpenDataDetector/ColliderML/tree/main/spaces/simulation-form)

A graphical front door for the SaaS simulation backend. Two tabs:

1. **Simulate** — a form with dropdowns for channel, events, and pileup.
   Submit, get a request ID, then hit **Refresh status** to poll until
   the job completes.
2. **Chat** — a natural-language interface backed by Claude that can
   estimate compute costs, check your credit balance, and submit jobs
   on your behalf after you confirm.

Both tabs talk to the same backend that the
[`colliderml.simulate(remote=True)`](../library/simulate.md) Python API
uses. Whichever entry point you pick, you end up in the same queue.

## Authentication

Authentication is HuggingFace OAuth. Click **Sign in with HuggingFace**
and the Space receives a scoped token that it forwards as a bearer
credential to the backend. The backend verifies the token, checks
quotas, and charges your credit balance.

Every user starts with 10 credits on first sign-in. See
[Remote Simulation](../guide/remote-simulation.md#credits) for the
credit economics.

## The form tab

| Field | Meaning |
| :-- | :-- |
| Physics channel | `ttbar`, `higgs_portal`, `zmumu`, `zee`, `diphoton`, `jets`, `susy_gmsb`, `hidden_valley`, `zprime`, `single_muon` |
| Events | Number of events to simulate (1–100 000) |
| Pileup | Pileup level in steps of 10, up to 200 (HL-LHC) |
| Seed | Random seed forwarded to every stage (default 42) |

After submission the UI shows:

- the backend's assigned request ID,
- an estimated cost in credits,
- an estimated wall-clock completion time,
- a note if the request was deduplicated against an existing run
  (zero credits in that case).

## The chat tab

The chat agent is Claude (`claude-sonnet-4-6`) with three tools:

- `check_balance` — queries `/v1/me` and reports credits remaining.
- `estimate_compute` — dry-runs the backend's capacity estimate and
  returns credits + estimated minutes.
- `submit_simulation` — posts to `/v1/simulate`. The system prompt
  requires the agent to confirm cost + parameters with the user before
  calling this tool.

The chat tab only works if the Space has `ANTHROPIC_API_KEY` set as a
Space-level secret. Without it the chat tab shows a "not configured"
message but the rest of the Space still works.

## Required Space secrets

| Secret | Purpose |
| :-- | :-- |
| `COLLIDERML_BACKEND` | Backend URL; defaults to `https://api.colliderml.com` |
| `ANTHROPIC_API_KEY` | Needed for the Chat tab; without it the tab degrades gracefully |

See the
[sync-spaces workflow docs](../internal/deployment.md#space-secrets)
for how to set these after the first automated sync.

## Running it locally

```bash
cd spaces/simulation-form
pip install -r requirements.txt
export COLLIDERML_BACKEND=http://localhost:8000   # point at your local backend
export ANTHROPIC_API_KEY=sk-ant-...               # optional, only for the Chat tab
python app.py
```

Use a HuggingFace token instead of OAuth for local testing — see the
`fetch_me` call in `app.py` for where to inject one.

## See also

- [Remote Simulation guide](../guide/remote-simulation.md) — the CLI
  equivalent of this Space
- [`colliderml.simulate`](../library/simulate.md) — the Python API it
  mirrors
- [`colliderml.remote`](../library/remote.md) — the client library
  the backend speaks to



========================================
# SOURCE: docs/spaces/leaderboard.md
========================================

# Leaderboard

**Live:** [huggingface.co/spaces/murnanedaniel/colliderml-leaderboard](https://huggingface.co/spaces/murnanedaniel/colliderml-leaderboard)
**Source:** [`spaces/leaderboard/`](https://github.com/OpenDataDetector/ColliderML/tree/main/spaces/leaderboard)

Public scoreboard for the six benchmark tasks shipped in
[`colliderml.tasks`](../library/tasks.md). Browse rankings, submit your
own predictions, or reproduce someone else's to earn credits.

## Layout

One tab per task. Inside each tab you get:

1. A table of the top 100 submissions (sortable, read-only). Columns
   include submitter HF username, all task metrics, credits earned for
   that submission, and whether it's a shipped baseline.
2. A **Submit predictions** accordion — upload a parquet file, the
   backend re-scores it against held-out truth, and any metric that
   beats the current best earns credits.
3. A **Reproduce a submission** accordion — enter a submission ID and
   upload your own predictions. If every metric lands within 2% of the
   original you earn 20 credits. This is the self-policing mechanism
   that keeps the board honest.

## Task discovery

The task list is loaded from the installed
[`colliderml.tasks`](../library/tasks.md) package at startup:

```python
import colliderml.tasks as _tasks
TASKS = _tasks.list_tasks()
```

Adding a new task in the library is enough to surface it in the UI —
no Space redeploy beyond a pin-bump on the `colliderml` package in
`requirements.txt`.

The Space falls back to a hardcoded task list if the `colliderml`
import fails (e.g. during a first-boot partial install).

## Authentication

Submission and reproduction both require a HuggingFace OAuth sign-in.
The received token is forwarded as a bearer credential to the
`/v1/benchmark/<task>/submit` and `/v1/benchmark/<task>/reproduce/<id>`
backend endpoints. Browsing the rankings does **not** require a
sign-in.

## Credit economics

| Action | Reward |
| :-- | :-- |
| Submit a new best on any metric | 30–50 credits (task-dependent) |
| Add a new baseline that becomes the bar to beat | 100 credits |
| Reproduce a submission within 2% tolerance on every metric | 20 credits |
| Contribute a new task or channel | 200 credits |

Credit balances are shared across every entry point in the platform
(the pip package, the Simulation Form Space, and this Space all draw
from the same backend ledger).

## Required Space secrets

| Secret | Purpose |
| :-- | :-- |
| `COLLIDERML_BACKEND` | Backend URL; defaults to `https://api.colliderml.com` |

## Running it locally

```bash
cd spaces/leaderboard
pip install -r requirements.txt
export COLLIDERML_BACKEND=http://localhost:8000
python app.py
```

## See also

- [Benchmark Tasks guide](../guide/tasks.md) — the conceptual tour
- [`colliderml.tasks`](../library/tasks.md) — the registry that
  populates the task tabs
- [Simulation Form](./simulation-form.md) — for generating training data



========================================
# SOURCE: docs/spaces/model-zoo.md
========================================

# Model Zoo

**Live:** [huggingface.co/spaces/murnanedaniel/colliderml-model-zoo](https://huggingface.co/spaces/murnanedaniel/colliderml-model-zoo)
**Source:** [`spaces/model-zoo/`](https://github.com/OpenDataDetector/ColliderML/tree/main/spaces/model-zoo)

A browser for HuggingFace models tagged `colliderml`. Discovery is
entirely tag-based: the Space calls
`huggingface_hub.list_models(filter="colliderml", limit=500, full=True)`
and lays the results out in a sortable table.

## Getting your model listed

Add the `colliderml` tag to your model card and (optionally) an
appropriate task tag:

```yaml
---
tags:
- colliderml
- tracking        # or jets, anomaly, or omit for "general"
task_category: object-detection
---
```

Re-push the model — the Space picks up the new tag on its next refresh
(tag queries go through HuggingFace's live API; no re-deploy needed).

## Columns

| Column | Source |
| :-- | :-- |
| `model` | `m.modelId` (repo name) |
| `task` | First of `tracking` / `jets` / `anomaly` in the tag list, else `general` |
| `downloads` | HuggingFace download count |
| `likes` | HuggingFace like count |
| `pipeline` | `m.pipeline_tag` |
| `url` | Direct link to the model card |

The default sort is by download count, descending. The **Filter by
task** dropdown lets you narrow to one of the three physics tasks or
to `general` (everything else).

## Authentication

Public models only — no HuggingFace sign-in needed.

## Running it locally

```bash
cd spaces/model-zoo
pip install -r requirements.txt
python app.py
```

## Limitations

- The Space caps the query at 500 models per refresh. If you have more
  than 500 models tagged `colliderml` (a nice problem to have), bump
  the `limit=` argument in `fetch_models()`.
- Task detection is naïve — it looks for a literal tag match. If you
  use different tags (e.g. `track-reco` instead of `tracking`) your
  model shows up under `general`.

## See also

- [Benchmark Tasks guide](../guide/tasks.md) — the tasks users train
  against when they publish a model
- [Leaderboard](./leaderboard.md) — where shipped models go to battle
