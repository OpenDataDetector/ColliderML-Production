#!/bin/bash
# Quick environment test inside the ODD container.
# Run with: apptainer exec ... bash container/test_container_env.sh
set -euo pipefail

SPACK=/spack/opt/spack/linux-x86_64

# Discover paths
PYTHON_DIR=$(find $SPACK -maxdepth 1 -name "python-3.13*" -type d | head -1)
HEPMC3_DIR=$(find $SPACK -maxdepth 1 -name "hepmc3-3.3.0-*" -type d | head -1)
ROOT_DIR=$(find $SPACK -maxdepth 1 -name "root-*" -type d | head -1)
ACTS_DIR=$(find $SPACK -maxdepth 1 -name "acts-main-*" -type d | head -1)
DD4HEP_DIR=$(find $SPACK -maxdepth 1 -name "dd4hep-*" -type d | head -1)

echo "Spack packages found:"
echo "  PYTHON:  $PYTHON_DIR"
echo "  ROOT:    $ROOT_DIR"
echo "  HEPMC3:  $HEPMC3_DIR"
echo "  ACTS:    $ACTS_DIR"
echo "  DD4HEP:  $DD4HEP_DIR"
echo ""

# Python 3.13 first
export PATH=$PYTHON_DIR/bin:$PATH

# All spack bin/lib
for d in $SPACK/*/bin; do [ -d "$d" ] && export PATH=$d:$PATH; done
for d in $SPACK/*/lib; do [ -d "$d" ] && export LD_LIBRARY_PATH=$d:${LD_LIBRARY_PATH:-}; done
for d in $SPACK/*/lib64; do [ -d "$d" ] && export LD_LIBRARY_PATH=$d:${LD_LIBRARY_PATH:-}; done

# ROOT Python bindings (spack puts them in lib/root/, not lib/python*/site-packages/)
for d in $SPACK/root-*/lib/root; do [ -d "$d" ] && export PYTHONPATH=$d:${PYTHONPATH:-}; done
for d in $SPACK/root-*/lib; do [ -d "$d" ] && export PYTHONPATH=$d:${PYTHONPATH:-}; done
# DD4hep Python bindings
for d in $SPACK/dd4hep-*/lib/python*/site-packages; do [ -d "$d" ] && export PYTHONPATH=$d:${PYTHONPATH:-}; done

# ACTS Python (symlink)
ACTS_SHIM=/tmp/colliderml_acts_py_$$
mkdir -p $ACTS_SHIM
ln -sf $ACTS_DIR/python $ACTS_SHIM/acts
export PYTHONPATH=$ACTS_SHIM:$PYTHONPATH

# THE FIX: ROOT_INCLUDE_PATH for HepMC3 dictionary resolution
for d in $SPACK/*/include; do [ -d "$d" ] && export ROOT_INCLUDE_PATH=$d:${ROOT_INCLUDE_PATH:-}; done

echo "ROOT_INCLUDE_PATH HepMC3 entry: $(echo $ROOT_INCLUDE_PATH | tr : '\n' | grep hepmc3)"
echo ""

# --- Test 1: ROOT/HepMC3 dictionary ---
echo "=== Test 1: ROOT/HepMC3 dictionary ==="
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
python3 "$SCRIPT_DIR/test_root_hepmc3.py"
echo ""

# --- Test 2: ACTS imports ---
echo "=== Test 2: ACTS Python imports ==="
python3 -c "
import acts
import acts.examples
import acts.examples.pythia8
import acts.examples.hepmc3
print(f'[PASS] ACTS {acts.__version__} with pythia8 + hepmc3 plugins')
"
echo ""

# --- Test 3: DDSim ---
echo "=== Test 3: DDSim import ==="
python3 -c "
from DDSim.DD4hepSimulation import DD4hepSimulation
print('[PASS] DDSim imported')
"
echo ""

echo "=== All environment tests passed ==="
