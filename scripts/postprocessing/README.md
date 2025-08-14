# EDM4HEP to HDF5 Conversion Scripts

This directory contains scripts for converting EDM4HEP ROOT files into HDF5 format for easier analysis and machine learning applications.

## Overview

The scripts convert different components of EDM4HEP data:
- Tracker hits
- Digitized tracker measurements (measurements.root)
- Reconstructed tracks (with states and hit associations)
- Particle information (including parent/daughter relationships)
- Calorimeter data

The output is organized into HDF5 files with an event-based hierarchy, making it easy to access specific events or ranges of events.

## Usage

### Converting All Data Types

To convert all data types at once, use the `convert_all.py` script:

```bash
python convert_all.py /path/to/edm4hep/files /path/to/output dataset_name
```

Optional arguments:
- `--chunk-size`: Number of events per output file (default: 1000)
- `--run-size`: Number of events per run (default: 10)

### Converting Individual Components

You can also convert individual components using their specific scripts:

1. Tracker Hits:
```bash
python convert_hits.py /path/to/edm4hep/files /path/to/output dataset_name
```

2. Digitized Measurements:
```bash
python convert_digihits.py --config /path/to/config.yaml
```
The config should provide at least: base_dir, output_dir, dataset_name, chunk_size, run_size.

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

## Output Structure

The converted data is stored in HDF5 files with the following structure:

```
/events/
    /event_0/
        /data    # Dataset containing properties
    /event_1/
        /data
    ...
```

Each event's data is stored as a structured array containing all relevant properties for that component:

- **Hits**: cellID, energy deposit, time, position, momentum, etc.
- **Tracks**: track parameters, quality metrics, states (IP, first/last hit), hit associations
- **Particles**: PDG code, charge, mass, vertex, momentum, parent/daughter relations
- **Calorimeter**: cellID, energy, time, position, etc.

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