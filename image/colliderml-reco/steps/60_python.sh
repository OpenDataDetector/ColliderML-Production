#!/bin/bash
# Python packages for the table converters, in a fixed directory on PYTHONPATH.
# (The base python is a spack virtualenv that refuses "pip install --user".)
source /opt/colliderml-reco/steps/common.sh
load_key4hep

python3 -m pip install --no-cache-dir --no-deps --require-hashes \
  --target /opt/reco-pylibs -r "$RECO_HOME/requirements.txt"
PYTHONPATH=/opt/reco-pylibs:$PYTHONPATH python3 -c 'import pyarrow, polars, tqdm, yaml, uproot'

# jemalloc from the base image at a fixed path, for LD_PRELOAD into k4run.
set +u; source /opt/setup_spack.sh; set -u
jemalloc_prefix="$(spack location --install-dir jemalloc)"
test -f "$jemalloc_prefix/lib/libjemalloc.so.2"
ln -s "$jemalloc_prefix/lib/libjemalloc.so.2" /opt/libjemalloc.so.2
