#!/usr/bin/env bash
# =============================================================================
# ColliderML Container Environment Setup
# =============================================================================
# Sets up the environment for running ColliderML pipeline stages inside the
# OpenDataDetector software container:
#   ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0
#
# Usage:
#   source scripts/cli/setup_container_env.sh
#
# This script configures PATH, LD_LIBRARY_PATH, PYTHONPATH, and other
# environment variables needed to use the full HEP software stack
# (ACTS, Pythia8, ROOT, DD4hep, Geant4, HepMC3, etc.) installed via Spack.
# =============================================================================

SPACK_BASE="/spack/opt/spack/linux-x86_64"

if [ ! -d "$SPACK_BASE" ]; then
    echo "ERROR: Spack installation not found at $SPACK_BASE"
    echo "This script must be run inside the ODD software container."
    return 1 2>/dev/null || exit 1
fi

# --- 1. Source .bashrc for PATH and CMAKE_PREFIX_PATH ---
# The container's .bashrc sets up PATH and CMAKE_PREFIX_PATH for all spack
# packages, but guards behind an interactive-shell check. We bypass it.
if [ -z "${PS1:-}" ]; then
    export PS1="colliderml"
fi
# shellcheck disable=SC1090
source /root/.bashrc 2>/dev/null || true

# --- 2. LD_LIBRARY_PATH: all spack lib/ directories ---
_ld_paths=""
for dir in "$SPACK_BASE"/*/lib; do
    [ -d "$dir" ] && _ld_paths="$dir:$_ld_paths"
done
export LD_LIBRARY_PATH="${_ld_paths}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# --- 3. PYTHONPATH: Python site-packages + ROOT + ACTS ---

# 3a. Collect all site-packages directories
_py_paths=""
for dir in "$SPACK_BASE"/*/lib/python*/site-packages; do
    [ -d "$dir" ] && _py_paths="$dir:$_py_paths"
done

# 3b. ROOT Python bindings (installed under lib/root/)
for dir in "$SPACK_BASE"/root-*/lib/root; do
    [ -d "$dir" ] && _py_paths="$dir:$_py_paths"
done

# 3c. ACTS Python bindings
# ACTS installs its Python package under <prefix>/python/ with __init__.py
# directly in that directory, but it must be importable as "acts". We create
# a symlink so Python can find it as a package named "acts".
_acts_python_dir=$(find "$SPACK_BASE"/acts-main-*/python -maxdepth 0 -type d 2>/dev/null | head -1)
if [ -n "$_acts_python_dir" ]; then
    ln -sf "$_acts_python_dir" /tmp/acts
    _py_paths="/tmp:$_py_paths"
else
    echo "WARNING: ACTS Python bindings not found"
fi

export PYTHONPATH="${_py_paths}${PYTHONPATH:+:$PYTHONPATH}"

# --- 4. PYTHIA8DATA: Pythia8 XML data directory ---
_pythia8_data=$(find "$SPACK_BASE"/pythia8-*/share/Pythia8/xmldoc -maxdepth 0 -type d 2>/dev/null | head -1)
if [ -n "$_pythia8_data" ]; then
    export PYTHIA8DATA="$_pythia8_data"
fi

# --- 5. Geant4 datasets (if downloaded) ---
if [ -d "/g4data" ]; then
    export GEANT4_DATA_DIR="/g4data"
fi

# --- 6. Clean up temporary variables ---
unset _ld_paths _py_paths _acts_python_dir _pythia8_data

echo "ColliderML container environment configured successfully."
echo "  Python:  $(which python3) ($(python3 --version 2>&1 | cut -d' ' -f2))"
echo "  ACTS:    $(python3 -c 'import acts; print(acts.__version__)' 2>/dev/null || echo 'not available')"
