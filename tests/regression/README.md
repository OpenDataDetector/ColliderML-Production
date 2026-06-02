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
