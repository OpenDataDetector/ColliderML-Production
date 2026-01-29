# EDM4HEP to Parquet Conversion Scripts

This directory contains scripts for converting EDM4HEP ROOT files into Parquet (and legacy HDF5) format for easier analysis and machine learning applications.

## Overview

The scripts convert different components of EDM4HEP data:
- Tracker hits
- Digitized tracker measurements (measurements.root)
- Reconstructed tracks (with states and hit associations)
- Particle information (including parent/daughter relationships)
- Calorimeter data

The output is organized into HDF5 files with an event-based hierarchy, making it easy to access specific events or ranges of events.

## Usage

### Converting All Data Types (config-driven)

To convert all data types at once and write Parquet outputs, use the `convert_all.py` script with a YAML config:

```bash
python convert_all.py --config /path/to/config.yaml
```

Key config fields:
- `campaign`, `dataset`, `version`
- `common.output_base_dir`: base directory for inputs/outputs
- `chunk_size`: number of events per output file (default: 1000)
- `run_size`: number of events per run (default: 10)
- `output_format: parquet` to enable Parquet (HDF5 is deprecated)

### Converting Individual Components

You can also convert individual components using their specific scripts:

1. Tracker Hits:
```bash
python convert_hits.py /path/to/edm4hep/files /path/to/output dataset_name
```

2. Digitized Measurements (tracker hits in reco space):
```bash
python convert_digihits.py --config /path/to/config.yaml
```
The config should provide at least: campaign, dataset, version, `common.output_base_dir`, `chunk_size`, `run_size`, and `output_format: parquet`.

2. Reconstructed Tracks:
```bash
python convert_tracks.py /path/to/edm4hep/files /path/to/output dataset_name
```

3. Particles:
```bash
python convert_particles.py /path/to/edm4hep/files /path/to/output dataset_name
```

4. Calorimeter Data:
```bash
python convert_calorimeter.py /path/to/edm4hep/files /path/to/output dataset_name
```

Each script accepts the same optional arguments as `convert_all.py`.

### Converting Efficiency Graphs

Efficiency histograms (TEfficiency objects) need to be converted to TGraphAsymmErrors + TTrees so they can be read by uproot.

#### Single File Conversion

```bash
root -l -b -q 'convert_eff_to_graphs.C+("/pscratch/sd/d/danieltm/ColliderML/simulation/full_pileup_mini_pilot/ttbar/v6/runs/0/performance_finding_ckf.root","/pscratch/sd/d/danieltm/ColliderML/simulation/full_pileup_mini_pilot/ttbar/v6/runs/0/performance_finding_ckf_graphs.root","trackeff_vs_pT,trackeff_vs_eta")'
```

#### Batch Conversion

To convert all efficiency ROOT files matching a pattern across multiple run directories:

```bash
# Convert all performance_finding_ckf.root files in a dataset
python batch_convert_efficiency_graphs.py /path/to/dataset

# Dry run to preview what would be processed
python batch_convert_efficiency_graphs.py /path/to/dataset --dry-run

# Convert with custom pattern
python batch_convert_efficiency_graphs.py /path/to/dataset --pattern "performance_*.root"

# Specify additional efficiency objects to convert
python batch_convert_efficiency_graphs.py /path/to/dataset --keys "trackeff_vs_pT,trackeff_vs_eta,trackeff_vs_phi"

# Control parallelism (default: auto-detect, max 16)
python batch_convert_efficiency_graphs.py /path/to/dataset --workers 8
```

The batch script will:
- Recursively find all matching ROOT files
- Create output files with `_graphs` suffix in the same directory
- Skip files where output already exists (use `--no-skip-existing` to reconvert)
- Process files in parallel for speed
- Provide progress tracking and error reporting

## Output Structure (legacy HDF5)

Legacy HDF5 outputs are stored with an `/events/event_#/` hierarchy where each group contains a structured `data` dataset per event. This path is retained for backwards compatibility.

## Output Structure (Parquet v1 schema)

Parquet outputs are written one file per event range (e.g. `...events0-999.parquet`). Each row corresponds to a single event and contains list-valued columns for per-object data.

At a high level, the content is:

- **Truth particles (`truth/particles`)**:
  - `event_id` (int64): global event index
  - `particle_id` (uint32): EDM4hep particle index (stable across files), **not** the Parquet row index
  - `pdg_id`, `mass`, `energy`, `charge`
  - `vx`, `vy`, `vz`, `time`, `px`, `py`, `pz`
  - `num_tracker_hits`, `num_calo_hits`
  - `primary` (bool): `True` if `created_in_simulation == False` (generator primary)
  - `vertex_primary` (when available), `parent_id` (first parent `particle_id`, when available)

- **Tracker hits (`reco/tracker_hits`)**:
  - `event_id` (int64): global event index
  - Geometry identifiers:
    - `volume_id` (uint8)
    - `layer_id` (uint16)
    - `surface_id` (uint32)
  - Hit positions:
    - `x`, `y`, `z` (reconstructed global coordinates)
    - `true_x`, `true_y`, `true_z` (SimHit coordinates)
  - Kinematics: `time`, `px`, `py`, `pz` (when available)
  - `particle_id` (uint32): EDM4hep particle index of the associated truth particle
  - `detector` (uint8): tracker detector enum
    - Pixel: negative endcap = 0, barrel = 1, positive endcap = 2
    - Short strip: negative endcap = 3, barrel = 4, positive endcap = 5
    - Long strip: negative endcap = 6, barrel = 7, positive endcap = 8
  - **Note**: `cell_id` is not stored for tracker hits; it is redundant given `(volume_id, layer_id, surface_id)` and the underlying geometry.

- **Tracks (`reco/tracks`)**:
  - `event_id` (int64)
  - `track_id` (int32): per-event track index
  - Quality: `num_hits`, `num_outliers`, `num_holes`, `num_shared_hits`, `chi2`
  - Fit parameters: `d0`, `z0`, `phi`, `theta`, `qop`, `time`
  - Truth parameters: `d0_truth`, `z0_truth`, `phi_truth`, `theta_truth`,
    `charge_truth`, `p_truth`, `pT_truth`, `time_truth`
  - `majority_particle_id` (uint32 or NaN): most frequent `particle_id` among associated hits
  - `hit_ids`: list of indices into the corresponding event’s tracker hits list

- **Calorimeter hits (`reco/calo_hits`)**:
  - `event_id` (int64)
  - `cell_id` (string): calorimeter cell identifier (bitfield encoded)
  - Positions: `x`, `y`, `z`
  - `detector` (uint8): calorimeter detector enum
    - ECal: negative endcap = 9, barrel = 10, positive endcap = 11
    - HCal: negative endcap = 12, barrel = 13, positive endcap = 14
  - `total_energy`: total calibrated energy per cell
  - Nested contribution lists:
    - `contrib_particle_ids`: list of `particle_id` values per cell
    - `contrib_energies`: list of energies per contributing particle
    - `contrib_times`: list of (energy-weighted) times per contributing particle

## Requirements

- Python 3.6+
- numpy
- pandas
- h5py
- uproot
- tqdm

## Example

Convert a ttbar dataset with default settings:

```bash
python convert_all.py \
    /eos/home-d/dmurnane/www/ColliderML/simulation/gg2ttbar/v1 \
    /eos/home-d/dmurnane/ColliderML/staging \
    pileup-10/ttbar/v1
```

This will create separate HDF5 files for hits, tracks, particles, and calorimeter data under the specified output directory. 

## Particle ↔ Hit Linkage

- The `particle_id` column in all tables is the EDM4hep particle index. It is **stable across files** within a dataset and is **not** the Parquet row index.
- To join particles to hits or tracks, you can:
  - Set the index on the particles table: `particles = particles_df.set_index("particle_id")`
  - Use standard joins on `particle_id` between tables (e.g. tracker hits, calo contributions, tracks via `majority_particle_id`).

## Geometry and Acts Compatibility

- The triplet `(volume_id, layer_id, surface_id)` is aligned with Acts geometry identifiers.
- Given the appropriate Acts/DD4hep geometry description, these identifiers can be mapped back to the corresponding detector elements.