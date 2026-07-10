#!/usr/bin/env bash
set -uo pipefail
REPO=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software/colliderml-prod-container
SETUP=$REPO/scripts/cli/setup_container_env.sh
WORK="${WORK:-/pscratch/sd/d/danieltm/backcompat}"

podman-hpc run --rm \
  -v /global/cfs/cdirs/m4958:/global/cfs/cdirs/m4958 -v /pscratch/sd/d/danieltm:/pscratch/sd/d/danieltm \
  --entrypoint /bin/bash ghcr.io/opendatadetector/sw:pr-8 -c "
    source $SETUP >/tmp/setup.log 2>&1
    export LD_LIBRARY_PATH=\$(printf '%s' \"\$LD_LIBRARY_PATH\" | tr ':' '\n' | grep -v '^/cache' | paste -sd:)
    # pytest into a temp target if the image lacks it
    if ! python3 -c 'import pytest' 2>/dev/null; then
        python3 -m pip install --quiet --target /tmp/pt --trusted-host pypi.org --trusted-host files.pythonhosted.org pytest 2>/dev/null || echo 'PYTEST-INSTALL-FAILED'
        export PYTHONPATH=/tmp/pt:\${PYTHONPATH:-}
    fi
    export COLLIDERML_V1_PARQUET_DIR=$WORK/backcompat/muon/v1/parquet
    export COLLIDERML_ACTSNATIVE_PARQUET_DIR=$WORK/runs/0
    echo 'python:' \$(command -v python3) '| polars:' \$(python3 -c 'import polars;print(polars.__version__)' 2>&1)
    cd $REPO
    python3 -m pytest tests/regression/test_actsnative_vs_v1.py -v --no-header -p no:cacheprovider 2>&1
"
