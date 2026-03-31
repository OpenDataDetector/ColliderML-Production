#!/bin/bash
# =============================================================
# ColliderML Container Environment Setup
# Sources spack paths and configures the full pipeline environment.
# Baked into the container image — no runtime discovery needed.
# =============================================================

# Load resolved spack package paths (includes ROOT_INCLUDE_PATH fix)
source /opt/colliderml/spack_paths.sh

SPACK=/spack/opt/spack/linux-x86_64

# --- Spack Python 3.13 first in PATH ---
export PATH=$PYTHON_DIR/bin:$PATH

# --- All spack bin/lib directories ---
for d in $SPACK/*/bin; do [ -d "$d" ] && export PATH=$d:$PATH; done
for d in $SPACK/*/lib; do [ -d "$d" ] && export LD_LIBRARY_PATH=$d:${LD_LIBRARY_PATH:-}; done
for d in $SPACK/*/lib64; do [ -d "$d" ] && export LD_LIBRARY_PATH=$d:${LD_LIBRARY_PATH:-}; done

# --- ACTS Python (symlinked at build time) ---
export PYTHONPATH=/opt/colliderml/python:${PYTHONPATH:-}

# --- ROOT Python (spack puts bindings in lib/root/) ---
for d in $SPACK/root-*/lib/root; do [ -d "$d" ] && export PYTHONPATH=$d:$PYTHONPATH; done

# --- Base Python site-packages (pip installs here, spack venv has separate site) ---
for d in $SPACK/python-3.*/lib/python*/site-packages; do [ -d "$d" ] && export PYTHONPATH=$d:$PYTHONPATH; done

# --- DD4hep Python ---
for d in $SPACK/dd4hep-*/lib/python*/site-packages; do [ -d "$d" ] && export PYTHONPATH=$d:$PYTHONPATH; done

# --- ODD factory plugins ---
if [ -d "${ODD_INSTALL:-}/lib" ]; then
    export LD_LIBRARY_PATH=$ODD_INSTALL/lib:$LD_LIBRARY_PATH
fi

# --- Pythia8 data ---
export PYTHIA8DATA=$PYTHIA8_DIR/share/Pythia8/xmldoc

# --- LHAPDF data (PDF sets for shower) ---
for d in $SPACK/lhapdf-*/share/LHAPDF; do [ -d "$d" ] && export LHAPDF_DATA_PATH=$d:${LHAPDF_DATA_PATH:-}; done
for d in $SPACK/lhapdfsets-*/share/lhapdfsets; do [ -d "$d" ] && export LHAPDF_DATA_PATH=$d:${LHAPDF_DATA_PATH:-}; done

# --- Geant4 data from CVMFS (bind-mount /cvmfs when running on CERN infra) ---
G4DATA=/cvmfs/geant4.cern.ch/share/data
if [ -d "$G4DATA" ]; then
    export G4NEUTRONHPDATA=$G4DATA/G4NDL4.7.1
    export G4LEDATA=$G4DATA/G4EMLOW8.6.1
    export G4LEVELGAMMADATA=$G4DATA/PhotonEvaporation6.1
    export G4RADIOACTIVEDATA=$G4DATA/RadioactiveDecay6.1.2
    export G4PARTICLEXSDATA=$G4DATA/G4PARTICLEXS4.1
    export G4PIIDATA=$G4DATA/G4PII1.3
    export G4REALSURFACEDATA=$G4DATA/RealSurface2.2
    export G4SAIDXSDATA=$G4DATA/G4SAIDDATA2.0
    export G4ABLADATA=$G4DATA/G4ABLA3.3
    export G4INCLDATA=$G4DATA/G4INCL1.2
    export G4ENSDFSTATEDATA=$G4DATA/G4ENSDFSTATE3.0
fi

# --- Pipeline env_setup — use baked-in default unless user overrides ---
export COLLIDERML_ENV_SETUP=${COLLIDERML_ENV_SETUP:-/opt/colliderml/env_setup.yaml}

echo "ColliderML container environment ready"
echo "  Python:  $(python3 --version 2>&1)"
echo "  ACTS:    $(python3 -c 'import acts; print(acts.__version__)' 2>/dev/null || echo 'import failed')"
echo "  ROOT:    $(python3 -c 'import ROOT; print(ROOT.gROOT.GetVersion())' 2>/dev/null || echo 'import failed')"
echo "  MG5:     $MG5_DIR"
echo "  ODD:     $ODD_PATH"
