# Plan for the Full-Pileup Pilot Run

## Intro

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

- We will try to use the NERSC offer of 50% off for runs of 256 nodes.
- We will use multi-threading for the simulation stage, with 8 threads per run.
- We can run 8 processes per node. 
- Each simulation (the bottleneck) takes around 20 minutes.
- So, we could run 8 threads, across 8 processes=8 files, each with 128 events.
- This should take around 5.3 hours, so we could allocate 6 hours per node.

Ah, but the 256 node offer requires a job to use 256 nodes. Our current system submits each node as a job.

