# Single Particle Simulation Feature

This document describes the implementation and usage of the single particle simulation feature in the ColliderML pipeline.

## Overview

The single particle simulation feature allows users to bypass the Pythia8 event generation step and directly use DD4hep's particle gun to generate and simulate single particles or multiple particles with specific properties. This is useful for detector studies, calibration, and performance evaluation.

## Implementation Details

The implementation adds support for DD4hep's particle gun mode in the simulation stage of the pipeline. The key components are:

1. **Modified ddsim_run.py**: Added support for enabling the particle gun and configuring its parameters based on YAML configuration.
   - Modular design with separate functions for different configuration aspects
   - Robust parameter handling with `getattr()` for default values
   - Comprehensive logging of particle gun configuration

2. **Configuration Options**: Added new configuration parameters in YAML files to control particle gun behavior.

3. **Job Submission**: The existing job submission system works with the new feature without modifications.

## Usage

To use the single particle simulation feature:

1. Create a YAML configuration file with `single_particle: true` and the desired particle gun parameters.
2. Run the simulation stage directly using the job submission script.

### Example Command

```bash
python colliderml_dev/batch/job_submission.py colliderml_dev/configs_development/testing_and_validation/single_particle_test.yaml
```

## Configuration Options

The following configuration options are available for the particle gun:

| Parameter | Description | Example Value | Default |
|-----------|-------------|---------------|---------|
| `single_particle` | Enable single particle mode | `true` | `false` |
| `gun_particle` | Particle type (PDG name) | `"e-"`, `"mu+"`, `"pi-"` | `"e-"` |
| `gun_energy` | Fixed energy in GeV | `10.0` | `None` |
| `gun_momentum_min` | Minimum momentum in GeV | `1.0` | `0.0` |
| `gun_momentum_max` | Maximum momentum in GeV | `50.0` | `10.0` |
| `gun_direction` | Direction vector [x,y,z] | `[0, 0, 1]` | `[0, 0, 1]` |
| `gun_position` | Position vector [x,y,z] in mm | `[0.0, 0.0, 0.0]` | `[0.0, 0.0, 0.0]` |
| `gun_distribution` | Angular distribution type | `"uniform"`, `"cos(theta)"`, `"eta"`, `"ffbar"` | `None` |
| `gun_theta_min` | Minimum theta angle | `"10*deg"` | `None` |
| `gun_theta_max` | Maximum theta angle | `"170*deg"` | `None` |
| `gun_phi_min` | Minimum phi angle | `"0*deg"` | `None` |
| `gun_phi_max` | Maximum phi angle | `"360*deg"` | `None` |
| `gun_multiplicity` | Number of particles per event | `1` | `1` |
| `vertexOffset` | Mean vertex offset [x,y,z,t] | `[0.0, 0.0, 0.0, 0.0]` | `[0.0, 0.0, 0.0, 0.0]` |
| `vertexSigma` | Vertex smearing sigma [x,y,z,t] | `[0.1, 0.1, 30.0, 0.0]` | `[0.0, 0.0, 0.0, 0.0]` |

### Units in Configuration

- **Energy and Momentum**: Values are specified as numeric values in GeV (e.g., `gun_energy: 10.0` for 10 GeV)
- **Angles**: Must include the unit as a string (e.g., `gun_theta_min: "10*deg"`)
- **Positions**: Values are specified in mm (e.g., `gun_position: [0.0, 0.0, 0.0]` for origin)

## Example Configurations

Three example configuration files are provided:

1. **single_particle_test.yaml**: Basic configuration with a single electron.
2. **single_particle_random.yaml**: Configuration with random momentum and direction distributions.
3. **multi_particle_test.yaml**: Configuration with multiple particles per event.

## Code Structure

The implementation follows a modular design with separate functions for different aspects of configuration:

- `configure_particle_gun()`: Sets up the particle gun parameters
- `log_particle_gun_config()`: Logs the particle gun configuration
- `configure_detector()`: Sets up the detector geometry
- `configure_physics()`: Configures physics settings
- `run_ddsim()`: Main function that orchestrates the simulation

This modular approach makes the code more maintainable and easier to extend with new features.

## Notes and Limitations

- When using `single_particle: true`, the generation and merge_smear stages are bypassed.
- The particle gun generates particles at the event vertex, which can be smeared using the `vertexOffset` and `vertexSigma` parameters.
- For complex event topologies, the standard Pythia8 generation is still recommended.

## Future Improvements

Potential future improvements include:

1. Support for multiple particle types in a single event.
2. Integration with the digitization and reconstruction stages for automated workflows.
3. Visualization tools for particle gun events.
4. More complex particle gun configurations (e.g., particle decays, custom distributions). 