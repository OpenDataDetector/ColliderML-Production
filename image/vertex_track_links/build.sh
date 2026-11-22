#!/bin/bash
# Build colliderml_vertex_links.cpython-313-x86_64-linux-gnu.so inside ghcr.io/opendatadetector/sw:pr-8, against the
# image's own ACTS (acts-44.99.99-colliderml-arrow), Python 3.13, pybind11 3.0.0, Eigen and Boost.
#   podman-hpc run --rm -v <repo>:<repo> --entrypoint /bin/bash ghcr.io/opendatadetector/sw:pr-8 -c "bash <repo>/image/vertex_track_links/build.sh <outdir>"
set -euo pipefail
OUT=${1:?output dir}
mkdir -p "$OUT"
SRC=$(cd "$(dirname "$0")" && pwd)/VertexTrackLinkWriter.cpp
SP=/spack/opt/spack/linux-x86_64
A=$(ls -d $SP/acts-44.99.99-colliderml-arrow-*)
PYB=$(ls -d $SP/py-pybind11-3.0.0-*)/include
PY=$(ls -d $SP/python-3.13.13-*)
EIG=$(ls -d $SP/eigen-3.4.0-*)/include/eigen3
BST=$(ls -d $SP/boost-1.88.0-*)/include
INC=""
for d in $SP/*/include; do INC="$INC -isystem $d"; done   # ACTS public headers pull in dependencies (e.g. nlohmann, GSL)
g++ -std=c++20 -O2 -fPIC -shared -fvisibility=hidden \
    -I$A/include -I$PYB -I$PY/include/python3.13 -isystem $EIG -isystem $BST $INC \
    "$SRC" -o "$OUT/colliderml_vertex_links.cpython-313-x86_64-linux-gnu.so" \
    -L$A/lib64 -L$A/lib -lActsExamplesFramework -lActsCore -Wl,-rpath,$A/lib64 -Wl,-rpath,$A/lib
echo "built $OUT/colliderml_vertex_links.cpython-313-x86_64-linux-gnu.so"
