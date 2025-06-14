# Plan for the Full-Pileup Pilot Runs

## Pilot Run A

**Run date**: 13th June 2025

The pilot will run the following stages:

1. MadGraph generation
2. Pythia generation
3. Merge smearing
4. Simulation
5. Digitization

We will produce 32,000 events across each of the following channels:

1. ttbar
3. dihiggs
4. GGF
5. drell-yan
7. SUSY

- We will use multi-threading for the simulation stage, with 8 threads per run.
- We can run 8 processes per node. 
- Each simulation (the bottleneck) takes around 20 minutes.
- So, we could run 8 threads, across 8 processes=8 files, each with 128 events.
- This should take around 5.3 hours, so we could allocate 6 hours per node.
- 128 events per run, 8 runs per node, 32 nodes = 32,768 events
- 5.3 hours per node, 32 nodes = 169 hours per channel
- 10 channels = 848 hours

## Pilot Run B

**Run date**: 19th June 2025

The pilot will run the following stages:

1. MadGraph generation
2. Pythia generation
3. Merge smearing
4. Simulation
5. Digitization

We will produce 100,000 events across the following channels:

1. ttbar
2. diboson
3. dihiggs
4. GGF
5. drell-yan
6. dijet (pileup)
7. SUSY
8. Z prime
9. HNL
10. Hidden Valley

- We will use the NERSC offer of 50% off for runs of 256 nodes.
- We will use multi-threading for the simulation stage, with 8 threads per run.
- We can run 8 processes per node. 
- Each simulation (the bottleneck) takes around 20 minutes.
- So, we could run 8 threads, across 8 processes=8 files, each with 64 events.
- This should take around 2.7 hours, so we could allocate 3 hours per node.
- 64 events per run, 8 runs per node, 256 nodes = 131,072 events
- 2.7 hours per node, 256 nodes = 691 hours per channel
- 10 channels = 6,910 hours @ 50% discount = 3,455 charged hours

