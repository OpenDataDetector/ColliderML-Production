# Beamspot-constrained refit study — reproduction recipe

Produces the paper's d0/z0 residual + resolution figures comparing:
1. **unconstrained** truth-seeded CKF + Kalman refit,
2. **beamspot-constrained** refit (beamspot as an extra 2D perigee measurement at the
   origin — ACTS PR #5342, merged upstream 2026-05),
3. the **naive predict-0 baseline** (assume every track originates at the beamspot
   centre: residual = −t_d0 / −t_z0; width = the beamspot itself).

## Inputs (read-only)
`/global/cfs/cdirs/m4958/data/ColliderML/simulation/full_pileup/ttbar/v1/runs/<N>/edm4hep.root`
— ttbar, ⟨μ⟩ = 200, ~10 events/file. Generation beamspot (pythia_config.yaml):
σ_xy = 0.0125 mm, σ_z = 55.5 mm.

## Environment
Everything runs in the public production container `ghcr.io/opendatadetector/sw:pr-8`
(pinned tarball: `.../stress_mu200/sw_pr8_image.tar`) — its ACTS has the
`RefittingAlgorithm.beamSpotConstraint` python binding and its ODD v6 at `/opt/odd`
is the 3 T geometry. **No custom ACTS build.**

Inside the container:
```bash
source colliderml-prod-container/scripts/cli/setup_container_env.sh
# strip the runtime-built ODD-v4 factory (tracker-only) that would shadow the full
# factory set and break loading the /opt/odd v6 xml:
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '^/cache' | paste -sd:)
```

NERSC note: `podman-hpc` needs a **login node or an exclusive compute node**
(interactive/debug/regular qos) — it fails silently on shared-qos nodes.

## Run
One job per edm4hep file; each job runs the CKF pass once and all three constraint
configs off it (`--constraint all`). The driver hard-fails unless B(0,0,0) = 3 T
(the 2 T/3 T mismatch is what stalled the original study).

```bash
python3 run_ckf_refit.py --input .../runs/0/edm4hep.root --output out/0 \
    --constraint all --events 10
```
Batch over N files (inside one container launch): see the xargs pattern in the
paper-production batch script; jobs are resumable (skips runs whose three
tracksummary files exist). ~6.5 min / file (10 μ=200 events, single-threaded;
CKF dominates — the three refits add ~7% each).

Outputs per run dir: `tracksummary_ckf_refit_{none,corrected,doga}.root`,
`performance_ckf_refit_*.root`, `tracksummary_ckf_base.root`.

## Constraint configs
| tag | covariance diag (mm²) | meaning |
|---|---|---|
| none | — | unconstrained refit |
| corrected | (0.0125², 55.5²) | true beamspot σ² — **use this** |
| doga | (0.0125, 55.5) | original study's matrix: σ values passed as covariance (d0 ~9× loose, z0 ~7.4× tight) — cross-check only |

## Analysis / figures
```bash
conda run -p /pscratch/sd/d/danieltm/envs/hep4m2 python \
    ../evaluation/refit_beamspot_plots.py --prod-dir <out>/prod
```
Row-concatenates the tracksummary TTrees (never merge histogram files), dedups CKF
duplicates per (run, event, majority particle), and writes the figures + a
`refit_summary.json` with the widths into `../paper/figures/`.
