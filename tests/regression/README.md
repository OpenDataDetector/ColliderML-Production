# ACTS-native parquet vs legacy convert_all regression

Compares the parquets produced by **two pipelines that take the same EDM4hep input** and verifies that downstream consumers see equivalent data:

| Path | Stages | Tracker hits row count |
|---|---|---|
| **v1** (legacy) | `digi_and_reco.py` → `*.root` → `scripts/postprocessing/convert_all.py` → parquet | one row per measurement (~16k/event) |
| **ACTS-native** | `digi_and_reco.py` w/ `output_parquet_arrow: True` → parquet directly via the Arrow plugin | one row per simhit (~16.6k/event) |

The two should be **semantically equivalent under deduplication**:
`acts_native.tracker_hits.unique(['x','y','z'])` should give a row count
close to (≤) v1's per-event tracker hit count, and for single-contributor
measurements (~98%) the per-hit fields should agree row-for-row.

## Running

```bash
# 1. Produce v1 parquet (the existing path, already works):
#    See scripts/cli/run_pipeline_docker.sh --channel higgs_portal
#    Outputs under <output>/<campaign>/<dataset>/<version>/parquet/

# 2. Produce ACTS-native parquet using the Arrow-enabled image:
#    docker build -f docker/acts-arrow/Dockerfile -t colliderml/acts-arrow:0.2.2-arrow-dev docker/acts-arrow
tests/regression/run_actsnative.sh /tmp/sim-verify/colliderml_output/higgs_portal_pu10_10evt

# 3. Run the diff:
COLLIDERML_V1_PARQUET_DIR=/tmp/sim-verify/colliderml_output/higgs_portal_pu10_10evt/docker_test/higgs_portal/v1/parquet \
COLLIDERML_ACTSNATIVE_PARQUET_DIR=/tmp/sim-verify/colliderml_output/higgs_portal_pu10_10evt/actsnative_parquet \
pytest tests/regression/test_actsnative_vs_v1.py -v
```

If the two env-vars are unset the suite skips cleanly so it can live in CI without blocking on a built image.

## Running on NERSC (podman-hpc, sw:pr-8) — the 2026-07 recipe

`run_actsnative.sh` above targets local Docker + the old arrow-dev image. On
Perlmutter use the podman-hpc pair instead:

```bash
# 1. one same-seed digi+reco emitting BOTH ROOT (for convert_all) and native
#    Arrow parquet, then convert_all on the ROOT — defaults: 100 events of a
#    drift_beamspot muon edm4hep, sw:pr-8; override via EDM/WORK/IMAGE/EVENTS:
tests/regression/run_backcompat_nersc.sh

# 2. the pytest comparison, inside the same image (it has polars):
tests/regression/run_compare_nersc.sh
```

2026-07 result on 100 muon events: **10 passed / 6 skipped** — fitted track
parameters (d0/z0/phi/theta/qop) bit-identical between the two writers;
convert_all drops zero-track events while native keeps them (asserted as a
subset relationship). The calo tests skip: no available image has the native
calo converter (tracker-hits-v2 dropped #5441), so calo parquet is
convert_all-only for now.

Tier-0 (no data needed, run in any image): `python3 -m pytest
tests/regression/test_postprocessing_deps.py` — hard-fails if any convert_all
dependency is missing (guards the pip-guard failure mode).

NERSC gotcha: if you source `setup_container_env.sh` manually, strip `/cache`
from `LD_LIBRARY_PATH` afterwards (ODD-v4 factory shadow) — both drivers do
this for you.
