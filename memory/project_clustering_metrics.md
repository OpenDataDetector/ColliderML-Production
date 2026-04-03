---
name: Clustering metrics project
description: Calorimeter clustering metrics suite in colliderml_lib — soft truth clusters, EIOU matching, energy-weighted metrics
type: project
---

Implemented `colliderml.clustering` subpackage in colliderml_lib with three modules:
- `truth.py`: builds soft truth clusters from primary ancestor decay chains + calo contributions
- `matching.py`: energy overlap matrix, EIOU, greedy matching
- `metrics.py`: efficiency, purity, energy resolution (sigma_eff), splitting/merging/fake rates, weighted V-score

**Why:** Existing notebook metrics used hard truth assignment (highest-energy particle per cell) and fragile dominant-particle matching. New suite uses soft energy fractions from simulation truth, EIOU-based object matching, and energy weighting throughout.

**How to apply:** All metrics are energy-weighted. V-score must be computed per-event then averaged. The validation notebook in colliderml_lib/notebooks/clustering_metrics_validation.ipynb demonstrates all concepts. Next steps are: (1) clean production scripts in colliderml_dev, (2) apply to real clustering algorithms beyond KMeans.
