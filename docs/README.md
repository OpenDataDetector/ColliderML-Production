# ColliderML Documentation Index

Welcome to the ColliderML comprehensive documentation. This directory contains detailed technical documentation for the event generation and simulation pipeline, with emphasis on Pythia8, MadGraph, and detector simulation for Beyond Standard Model (BSM) physics studies.

## Quick Navigation

### 📋 Core Documentation

1. **[PYTHIA_SIMULATION_REVIEW.md](./PYTHIA_SIMULATION_REVIEW.md)** — **START HERE**
   - Complete architectural overview of the pipeline
   - Pythia8 generation system (hard scatter + pileup)
   - MadGraph generation (init + parallel generation)
   - ACTS-based event merging
   - Configuration system and batch submission
   - Data directory structure
   - BSM simulation recommendations
   - Troubleshooting guide

2. **[ADVANCED_SIMULATION_TOPICS.md](./ADVANCED_SIMULATION_TOPICS.md)** — Deep Technical Details
   - Event merging algorithm specifics
   - Poisson sampling mathematics
   - Vertex smearing mechanics
   - Seed management and reproducibility
   - Multi-node job coordination
   - HepMC splitting algorithms
   - Card customization patterns
   - Batch job architecture
   - Performance analysis and tuning
   - Troubleshooting complex scenarios

## 📚 Topics by Use Case

### Getting Started with BSM Simulations

**If you want to...**

- **Generate Pythia8 events**: Read PYTHIA_SIMULATION_REVIEW.md §2 (Pythia8 Generation)
- **Run MadGraph for complex processes**: Read PYTHIA_SIMULATION_REVIEW.md §3 (MadGraph Generation)
- **Merge signal + pileup**: Read PYTHIA_SIMULATION_REVIEW.md §4 (Event Merging)
- **Submit batch jobs**: Read PYTHIA_SIMULATION_REVIEW.md §6 (Batch Submission)
- **Understand reproducibility**: Read ADVANCED_SIMULATION_TOPICS.md §2 (Seed Management)

### Production Workflows

- **High-statistics signal generation**: §12.2 in PYTHIA_SIMULATION_REVIEW.md
- **Multi-node parallel generation**: §3 in ADVANCED_SIMULATION_TOPICS.md
- **Custom pileup studies**: §10.1 in ADVANCED_SIMULATION_TOPICS.md
- **Cross-section calculations**: §10.2 in ADVANCED_SIMULATION_TOPICS.md

### Troubleshooting & Optimization

- **Common errors**: PYTHIA_SIMULATION_REVIEW.md §13
- **Complex failures**: ADVANCED_SIMULATION_TOPICS.md §8
- **Performance tuning**: ADVANCED_SIMULATION_TOPICS.md §7
- **Configuration validation**: ADVANCED_SIMULATION_TOPICS.md §9

## 🔗 Related Documentation

**In the repository:**
- `README.md` (top-level) — Project overview and quickstart
- `scripts/cli/README.md` — Job submission details
- `scripts/simulation/example_usage.md` — Practical examples
- `configs_production/README.md` — Configuration file guidelines

**External references:**
- [Pythia8 Documentation](https://pythia.org)
- [MadGraph5_aMC@NLO](https://launchpad.net/madgraph5)
- [ACTS Framework](https://acts.readthedocs.io)
- [DD4hep Simulation](https://dd4hep.cern.ch)

## 🎯 Key Concepts Overview

### Pipeline Architecture

```
Pythia8/MadGraph           DD4hep                Reconstruction
Generation              Simulation              & Analysis
    ↓                        ↓                        ↓
Hard Scatter   ─┐       Detector                Hit
  + Pileup      ├→ Merge → Simulation → Digitization → Reco
                ↓                                       ↓
             Vertices                           Final Objects
             Smeared                        (Tracks, Clusters)
```

### Configuration Hierarchy

```
env_setup.yaml (environment variables)
    ↓
Stage Config (pythia_config.yaml)
    ↓
CLI Overrides (--events, --seed, etc.)
    ↓
Resolved Configuration
    ↓
Execution Scripts
```

### Execution Models

| Mode | Best For | Scaling |
|------|----------|---------|
| **interactive** | Development, debugging | Serial (1 job) |
| **monolithic_slurm** | Small studies | Single HPC node |
| **distributed_slurm** | Large productions | Array job (N jobs) |
| **multi_node_slurm** | Massive jobs | Task farm (multiple nodes) |

## 📊 Documentation Statistics

| Document | Sections | Topics | Focus |
|----------|----------|--------|-------|
| PYTHIA_SIMULATION_REVIEW.md | 16 | Pythia8, MadGraph, ACTS, Config, Batch | Production architecture |
| ADVANCED_SIMULATION_TOPICS.md | 10 | Algorithms, Coordination, Tuning | Technical depth |

## 🔐 Version & Reproducibility

**These docs describe:**
- Code state: ColliderML development repository
- Configuration: Full YAML-based system
- Versioning: Git-tracked with commit snapshots
- Reproducibility: Seed management and config snapshots

## 💡 Tips & Best Practices

### Before Running Production Samples

1. **Test your configuration** on a small sample (10-100 events)
2. **Validate inputs** - check that all paths are accessible
3. **Set appropriate time limits** - add 50% buffer to estimates
4. **Use meaningful seed patterns** - enables reproducibility
5. **Archive your configs** - save to version control

### For BSM Physics Studies

1. **Understand your process** - know the physics (σ, decay chains)
2. **Use appropriate pileup** - match your physics channel
3. **Tune vertex smearing** - set to detector specifications
4. **Monitor job status** - watch for systematic failures
5. **Validate outputs** - spot-check simulation quality

## 📞 Getting Help

**For documentation issues:**
- File an issue in the repository
- Contact the simulation working group

**For physics questions:**
- Consult PYTHIA_SIMULATION_REVIEW.md §9 (Physics Processes)
- Review MadGraph/Pythia8 official documentation

**For technical debugging:**
- See troubleshooting sections in both documents
- Check output logs for error patterns
- Validate configurations before submission

## 📝 Document Metadata

| Item | Details |
|------|---------|
| **Created** | 2025-01-19 |
| **Version** | 1.0 |
| **Scope** | Pythia8/MadGraph-based BSM Simulation |
| **Target Audience** | Physics researchers, simulation experts, BSM analysts |
| **Repository** | ColliderML Development |

---

**Next Steps:**
1. Read [PYTHIA_SIMULATION_REVIEW.md](./PYTHIA_SIMULATION_REVIEW.md) for architecture overview
2. Check specific sections for your use case
3. Reference [ADVANCED_SIMULATION_TOPICS.md](./ADVANCED_SIMULATION_TOPICS.md) for technical details
4. Try a small test run using examples from `scripts/simulation/example_usage.md`

Happy simulating! 🚀

