# Calorimeter Performance Plotting

This directory contains scripts for generating publication-quality plots of calorimeter digitization performance.

## Files

- `plot_calo_performance.py`: Main plotting script
- `calo_performance_config.yaml`: Example configuration file
- `plots/`: Output directory for generated plots

## Usage

### Basic Usage

```bash
python plot_calo_performance.py calo_performance_config.yaml
```

### Custom Configuration

Create a new YAML config file (e.g., `my_config.yaml`) with your desired settings:

```yaml
digi_file: "/path/to/your/edm4hep_digitized.root"
events_range: [0, 5000]
pdg_code: 11  # electron
output_dir: "./my_plots"
```

Then run:

```bash
python plot_calo_performance.py my_config.yaml
```

## Generated Plots

The script generates three sets of plots:

### 1. Energy Distribution (`energy_distribution.pdf/png`)
- Shows the generator particle energy distribution
- Log-log scale with error bars
- Useful for verifying the input particle spectrum

### 2. Residuals and Pulls (`residuals_pulls.pdf/png`)
Three-panel plot showing:
- **Top**: Absolute residual |E_gen - E_calo|
- **Middle**: Relative residual |E_gen - E_calo| / E_gen
- **Bottom**: Pull distribution (E_gen - E_calo) / √E_calo

### 3. Profile Plots (`profiles_eta_pt.pdf/png`)
Two-panel plot showing mean relative residual as a function of:
- **Left**: Pseudorapidity (η)
- **Right**: Transverse momentum (pT)

These profiles reveal systematic variations in calorimeter response with kinematics.

## Configuration Parameters

### Data Selection
- `digi_file`: Path to digitized EDM4hep ROOT file
- `events_range`: [start, stop] event indices to process
- `pdg_code`: PDG code to filter (11=e⁻, 22=γ, 211=π⁺, etc.)

### Detector Parameters
- `tracking_radius`: Tracker radius in mm
- `tracking_z_max`: Tracker maximum z in mm
- `energy_threshold`: Energy threshold in GeV

### Plot Parameters
- `energy_nbins`: Number of bins for energy distribution
- `profile_nbins`: Number of bins for profile plots
- Custom labels and figure sizes

## Dependencies

- numpy
- pandas
- matplotlib
- pyyaml
- atlasify
- pyedm4hep (custom library)

## Notes

- All plots are saved in both PDF (vector) and PNG (raster) formats
- The script automatically creates the output directory if it doesn't exist
- Error bars represent Poisson uncertainties (√N) on bin contents
- Profile plot errors represent standard error on the mean (σ/√N)

