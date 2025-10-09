# Particle Gun - Log-Uniform Energy Distribution

Generate single particle HepMC3 events with log-uniform energy distribution using ACTS `ParametricParticleGenerator`.

## Usage

```bash
# Via CLI (production)
cd scripts/cli
python run_stage.py ../configs_production/full_pileup/diphoton/particlegun_config.yaml

# Interactive test
python run_stage.py ../configs_production/full_pileup/diphoton/particlegun_config.yaml \
    --execution-mode interactive
```

## Configuration

```yaml
stage: "particlegun_generation"
events: 1000
particle: 22                   # PDG code (22=photon, 11=electron, 13=muon)
energy_min: 1.0               # GeV
energy_max: 1000.0            # GeV
log_uniform: true             # Use log-uniform energy distribution
eta_min: -2.5
eta_max: 2.5
phi_min: 0.0
phi_max: 6.28318530718        # 2*pi
```

## Energy Distributions

**log_uniform: true** - Samples uniformly in log(E), equal events per decade
- Good for wide energy ranges (e.g., 1-1000 GeV)
- Uses ACTS `pLogUniform=True`

**log_uniform: false** - Linear uniform sampling
- Good for narrow energy ranges

## Output

Generates `events.hepmc3` compatible with `ddsim_run.py`.

## Implementation

Uses ACTS `ParametricParticleGenerator` with `pLogUniform` parameter:
- Native ACTS particle gun (no external Python libraries needed)
- Writes directly to HepMC3 via `HepMC3Writer`
- Runs in ACTS environment (no shifter container)

## Common Particle PDG Codes

- 22 = photon
- 11 = electron, -11 = positron
- 13 = muon-, -13 = muon+
- 211 = pi+, -211 = pi-, 111 = pi0
- 2212 = proton, -2212 = antiproton

