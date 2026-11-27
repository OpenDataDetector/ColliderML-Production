# Validation results

| Image | Variant | ECAL gap correction | Image ID |
|---|---|---|---|
| `ghcr.io/opendatadetector/colliderml-reco:20261127` (published) | optimised | off | `a61ba905a0c43f48418f0375b3bf50698338cf5356504629819c4b2c85f81e78` |
| first build of `colliderml/reco:20261127` (not published) | optimised | on | `7390034fab5d043ceeac63557881b61e080b92348bd67add9af807e4a2e6fe08` |
| `colliderml/reco:20261127-reference` (not published) | reference | on | `16b113ed56bef1f5191ceecb414d3f306acc8d9a2fd5f2f8df93650ca663d2d9` |

The published image differs from the first build in two ways. The optimised Pandora sources
are fetched from the tag `colliderml-reco-20261127` of the OpenDataDetector forks instead of
being patched in at build time; the trees are the same, and the four Pandora libraries came
out with the same SHA-256 (`results/library_checksums.txt`). And the ECAL gap correction is
switched off (`patches/k4odd_ecal_gap_correction_off.patch`). Every set below was rerun on
the published image; the records of the first build are in `results/20261127-gap-on/`.

The reference variant was built and validated before the last change sets to the recipe (the
LCContent fix of the optimised variant, five small script changes, the gap correction
switch). Its Pandora libraries do not depend on those changes.

Reference output: the earlier image `reco:20261106` (unoptimised Pandora sources, CMake
`RelWithDebInfo`, ECAL gap correction on), which produced the ColliderML v23 pilot runs.

All runs are on one NERSC Perlmutter CPU node (two AMD EPYC 7763 processors, 512 GB), one
`k4run` process per slice, each pinned to its own core, jemalloc preloaded.

## What is compared

`compare_reco.py` compares a candidate file with a reference file at zero tolerance, in
collection order:

- every stored field of the PFOs, clusters and start vertices, and the targets of their
  relations (the hits of each cluster, the clusters, tracks and vertex of each PFO);
- every digitised calorimeter hit: cell ID, energy, energy error, time, type and position.

Not compared: the links from digitised hits to simulated hits, and the collections copied
from the input.

## Event sets

| Set | Content | Slices | Events |
|---|---|---|---|
| `suite345` | 15 PU200 ttbar events, 30 PU0 ttbar events, 300 single particles (photon, pion, K0L at 10 and 100 GeV) | 22 | 345 |
| `pu200_192` | all events of the PU200 ttbar pilot runs 0, 16 and 1562, one event per process | 192 | 192 |
| `pu0_run0` | PU0 ttbar pilot run 0, in the 16 slices of 80 events used for the reference | 16 | 1280 |

The 15 PU200 events of `suite345` are also in `pu200_192`, and its 30 PU0 events are the start
of `pu0_run0`.

## Optimised variant against the reference output

Published image, `results/20261127/suite345.json`, `pu200_192.json`, `pu0_run0.json` (the
first build gave the same counts, `results/20261127-gap-on/`):

| Set | Events | Events equal to the reference | Pandora objects compared | Calorimeter hits compared |
|---|---|---|---|---|
| `suite345` | 345 | 345 | 298075 | 12980793 |
| `pu200_192` | 192 | 192 | 3447033 | 148331116 |
| `pu0_run0` | 1280 | 1280 | 1033851 | 40274359 |

Every event of the three sets is equal to the reference output, in the Pandora collections
and in the digitised calorimeter hits, with the gap correction off in the published image and
on in the reference. These are 1817 event comparisons of 1772 distinct
events (the 45 ttbar events of `suite345` are also in the other two sets).

### Earlier build of the same day

A first build of the optimised variant (image `2a2c506f`, no longer distributed) differed from
the reference in six of these events:

| Sample | Run | Entries |
|---|---|---|
| PU200 | 0 | 49, 57 |
| PU200 | 16 | 31, 56 |
| PU200 | 1562 | 38 |
| PU0 | 0 | 207 |

In five of them a few clusters were grouped
differently and the event had one PFO fewer; in the sixth, one cluster differed in the
seventh digit. Two changes in the optimised LCContent caused this:

- the fast copy of the helix layer-crossing count in MainFragmentRemoval sampled the helix at
  50 points where the original uses 100;
- in several optimised algorithms the cached photon id or track comparison energy of a
  cluster was first requested at another moment than in the original code. These caches are
  not reset when the track associations of the cluster change, so the cached value depends
  on the moment of the first request.

`patches/lccontent_optimised.patch` now keeps the original behaviour in both respects, and the
six events are equal to the reference. The records of the earlier build are in
`results/earlier_runs.tar.gz` (`before_fix/`, `differing/`), together with the runs made before
the code review (`before_review/`). The same archive holds the comparison of that earlier
build with a `-march=znver3` build of the same sources: the two agreed on all 1472 ttbar
events, so the compiler target played no part.

## Reference variant against the reference output

The reference variant is built from the sources and flags of `reco:20261106`
(`--build-arg PANDORA_VARIANT=reference`). `results/20261127-reference/reference_variant.json`:

| Set | Events | Events equal to the reference | Pandora objects compared | Calorimeter hits compared | Hits equal |
|---|---|---|---|---|---|
| `suite345` | 345 | 345 | 298075 | 12980793 | all |
| the six slices with an event that differed in the earlier optimised build (5 PU200 events, 80 PU0 events) | 85 | 85 | 155175 | 6595758 | all |

We did not run the reference variant on the full `pu200_192` and `pu0_run0` sets.

## Timing

One process on an otherwise idle node, pinned to one core (task list `timing`: the 5 PU200
events of run 0 in `suite345`, and the first 80-event slice of `pu0_run0`). "Pandora" is the
summed algorithm time reported with `PANDORA_ALG_TIMING=1`; "wall" is the whole `k4run`
process, including about 12 s of start-up and geometry loading.

Optimised variant, published image (`results/20261127/timing.json`) and first build with the
gap correction on (`results/20261127-gap-on/timing.json`):

| Sample | Events | Gap correction | Wall per event | Pandora per event |
|---|---|---|---|---|
| ttbar, 200 pileup | 5 | off | 21.6 s (18.3 to 27.1 s) | 8.6 s (5.9 to 12.7 s) |
| ttbar, 200 pileup | 5 | on | 21.9 s (18.8 to 26.6 s) | 8.5 s (5.8 to 12.5 s) |
| ttbar, no pileup | 80 | off | 0.51 s | 0.20 s |
| ttbar, no pileup | 80 | on | 0.51 s | 0.19 s |

In the optimised variant the switch saves little: its `fillECALGaps` already skips the pair
search. The 20 to 30 s per PU200 event that the correction costs are in the reference
variant (ECAL gap correction switch, below).

The earlier build without the LCContent fix, measured the same way on an idle node, took
21.4 s per PU200 event with 8.4 s in Pandora (`results/earlier_runs.tar.gz`,
`before_review/timing.json`). 

Entry 38 of PU200 run 0, compared with a `-march=znver3` build of the unfixed sources that
was measured on 2026-11-23:

| Build | Wall | Pandora |
|---|---|---|
| this image | 21.6 s | 7.9 s |
| `-march=znver3`, unfixed | 22.1 s | 7.8 s |

The two rows differ in compiler target and in the LCContent fix, so they do not isolate
either.

With many processes on the node, as in production. The two variants were run with
different numbers of processes, so the ratio of their rows is indicative only:

| Variant | Sample | Processes on the node | Events | Wall per event |
|---|---|---|---|---|
| optimised, published | ttbar, 200 pileup | 64 | 192 | 37 s |
| optimised, published | ttbar, no pileup | 16 | 1280 | 0.72 s |
| optimised, first build | ttbar, 200 pileup | 64 | 192 | 36 s |
| optimised, first build | ttbar, no pileup | 16 | 1280 | 0.74 s |
| reference | ttbar, 200 pileup | 28, with other runs on the node | 20 | 1067 s |
| reference | ttbar, no pileup | 28, with other runs on the node | 110 | 5.0 s |

### ECAL gap correction switch

`reco:20261106` and the reference variant run `DDCaloDigi` with `ECALGapCorrection = 1`. As a control we
ran the reference variant on two PU200 events with the correction on and off
(`--ECalBarrelDigi.ECALGapCorrection=0 --ECalEndcapDigi.ECALGapCorrection=0`), at the same time
on the same node (`results/20261127-reference/gap_timing_*.json`):

| PU200 event | Correction | Wall | Pandora | Outside Pandora |
|---|---|---|---|---|
| run 0, entry 0 | on | 614 s | 577 s | 38 s |
| run 0, entry 0 | off | 595 s | 578 s | 17 s |
| run 0, entry 38 | on | 864 s | 815 s | 50 s |
| run 0, entry 38 | off | 837 s | 819 s | 18 s |

The output is equal to the reference in all four runs. The published image runs with the
correction off and is equal to the reference in all three sets above.

Largest resident set size of a process:

| Variant | PU200 event | PU0 slice of 80 events |
|---|---|---|
| optimised | 6.3 GiB | 2.2 GiB |
| reference | 6.6 GiB | 1.8 GiB |

## Other checks

`check_image.sh` (log in `results/<tag>/check_image.log`):

- `colliderml-reco selftest` passes as root, with `--user 12345:12345`, and with
  `--read-only --tmpfs /tmp --user 12345:12345`, without network and without any bind mount.
- The image configuration has no environment variable, label or annotation from the build
  host, and the recipe files, install prefixes and built libraries contain no NERSC path.
  (The k4ODD source tree has a CI workflow that names a NERSC web URL; that file is part of
  the public k4ODD repository.)

Build reproducibility: repeated builds gave the same SHA-256 for the Pandora libraries of
each variant (`results/library_checksums.txt`). The four Pandora libraries of the reference variant are
bit-identical to those in `reco:20261106`.

## Rerunning

```bash
python3 nersc/make_tasks.py nersc/tasks        # task lists with the NERSC paths of inputs and references
srun -N1 -n1 nersc/run_validation_perlmutter.sh localhost/colliderml/reco:20261127
srun -N1 -n1 nersc/run_validation_perlmutter.sh localhost/colliderml/reco:20261127 timing
srun -N1 -n1 nersc/run_validation_perlmutter.sh localhost/colliderml/reco:20261127-reference reference_variant
validation/check_image.sh localhost/colliderml/reco:20261127 podman-hpc | tee validation/results/20261127/check_image.log
```

On another site, write a task list for your own files (format in `run_validation.py`) and call
`run_validation.py` directly.
