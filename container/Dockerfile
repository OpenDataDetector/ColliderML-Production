# ColliderML Full-Pipeline Container
# Extends the ODD spack-built image with all fixes for end-to-end running.
#
# Build:
#   docker build -t ghcr.io/opendatadetector/colliderml-production:latest container/
#
# Use on lxplus (after pushing to registry):
#   apptainer pull docker://ghcr.io/opendatadetector/colliderml-production:latest
#   apptainer exec --bind /eos:/eos --bind /afs:/afs --bind /tmp:/tmp \
#     --bind /run/user --env KRB5CCNAME=$KRB5CCNAME \
#     colliderml-production_latest.sif bash

FROM ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0

LABEL maintainer="ColliderML <colliderml@cern.ch>"
LABEL description="Full ColliderML simulation pipeline: MadGraph -> Pythia8 -> DDSim -> ACTS -> Parquet"

ENV SPACK=/spack/opt/spack/linux-x86_64

# =============================================================
# Stage 1: Discover spack paths and write them to a static file.
# =============================================================
RUN bash -c '\
    SPACK=/spack/opt/spack/linux-x86_64 && \
    mkdir -p /opt/colliderml && \
    \
    find_pkg() { find $SPACK -maxdepth 1 -name "$1" -type d | head -1; } && \
    \
    ACTS_DIR=$(find_pkg "acts-main-*") && \
    PYTHIA8_DIR=$(find_pkg "pythia8-8.313-*") && \
    HEPMC3_DIR=$(find_pkg "hepmc3-3.3.0-*") && \
    ROOT_DIR=$(find_pkg "root-*") && \
    DD4HEP_DIR=$(find_pkg "dd4hep-*") && \
    PYTHON_DIR=$(find_pkg "python-3.13*") && \
    MG5_DIR=$(find_pkg "madgraph5amc-*") && \
    GEANT4_DIR=$(find_pkg "geant4-*") && \
    \
    cat > /opt/colliderml/spack_paths.sh << EOF
export SPACK=$SPACK
export ACTS_DIR=$ACTS_DIR
export PYTHIA8_DIR=$PYTHIA8_DIR
export HEPMC3_DIR=$HEPMC3_DIR
export ROOT_DIR=$ROOT_DIR
export DD4HEP_DIR=$DD4HEP_DIR
export PYTHON_DIR=$PYTHON_DIR
export MG5_DIR=$MG5_DIR
export GEANT4_DIR=$GEANT4_DIR
EOF
    \
    echo "Discovered paths:" && cat /opt/colliderml/spack_paths.sh \
'

# =============================================================
# Stage 2: Fix ACTS Python import (symlink python/ -> acts/)
# =============================================================
RUN bash -c '\
    source /opt/colliderml/spack_paths.sh && \
    mkdir -p /opt/colliderml/python && \
    ln -sf $ACTS_DIR/python /opt/colliderml/python/acts \
'

# =============================================================
# Stage 3: Fix ROOT/HepMC3 dictionary integration
#           Set ROOT_INCLUDE_PATH for all spack packages.
# =============================================================
RUN bash -c '\
    source /opt/colliderml/spack_paths.sh && \
    SPACK=/spack/opt/spack/linux-x86_64 && \
    INCLUDE_DIRS="" && \
    for d in $SPACK/*/include; do \
        [ -d "$d" ] && INCLUDE_DIRS="$d:$INCLUDE_DIRS"; \
    done && \
    echo "export ROOT_INCLUDE_PATH=$INCLUDE_DIRS\$ROOT_INCLUDE_PATH" >> /opt/colliderml/spack_paths.sh && \
    ls $HEPMC3_DIR/include/HepMC3/Data/GenRunInfoData.h && \
    echo "ROOT/HepMC3 dictionary fix applied" \
'

# =============================================================
# Stage 4: Clone ODD v4.0.4 from GitLab CERN and compile plugins
#           GitHub repo is a stale redirect; real repo on GitLab.
#           v4.0.4 = ACTS main pin. Factory code uses dd4hep
#           VariantParameters, no ACTS dependency.
# =============================================================
RUN git clone --depth 1 --branch v4.0.4 https://gitlab.cern.ch/acts/OpenDataDetector.git /opt/colliderml/ODD && \
    echo 'export ODD_PATH=/opt/colliderml/ODD_install/share/OpenDataDetector' >> /opt/colliderml/spack_paths.sh

RUN bash -c '\
    source /opt/colliderml/spack_paths.sh && \
    SPACK=/spack/opt/spack/linux-x86_64 && \
    export PATH=$PYTHON_DIR/bin:$PATH && \
    for d in $SPACK/*/bin; do export PATH=$d:$PATH; done && \
    for d in $SPACK/*/lib; do export LD_LIBRARY_PATH=$d:${LD_LIBRARY_PATH:-}; done && \
    PREFIX_PATH="" && \
    for d in $SPACK/*/; do PREFIX_PATH="${d%/};$PREFIX_PATH"; done && \
    mkdir -p /tmp/odd_build && cd /tmp/odd_build && \
    cmake /opt/colliderml/ODD \
        -DCMAKE_PREFIX_PATH="$PREFIX_PATH" \
        -DCMAKE_INSTALL_PREFIX=/opt/colliderml/ODD_install \
        -DCMAKE_CXX_STANDARD=20 \
        -Wno-dev 2>&1 | tail -5 && \
    make -j$(nproc) 2>&1 | tail -10 && \
    make install 2>&1 | tail -5 && \
    echo "export ODD_INSTALL=/opt/colliderml/ODD_install" >> /opt/colliderml/spack_paths.sh && \
    cd / && rm -rf /tmp/odd_build \
'

# =============================================================
# Stage 5: Install Python packages
# =============================================================
RUN bash -c '\
    source /opt/colliderml/spack_paths.sh && \
    $PYTHON_DIR/bin/python3 -m ensurepip 2>/dev/null || true && \
    $PYTHON_DIR/bin/python3 -m pip install --no-cache-dir \
        pyedm4hep pyhepmc uproot awkward pyarrow pandas \
        pyyaml simple-slurm six tqdm h5py \
'

# =============================================================
# Stage 6: Configure MadGraph — Pythia8 path + PY8 interface
# =============================================================
RUN bash -c '\
    source /opt/colliderml/spack_paths.sh && \
    sed -i "s|# pythia8_path.*|pythia8_path = $PYTHIA8_DIR|" $MG5_DIR/input/mg5_configuration.txt && \
    LHAPDF_DIR=$(find $SPACK -name "lhapdf-*" -maxdepth 1 -type d | head -1) && \
    [ -n "$LHAPDF_DIR" ] && sed -i "s|# lhapdf_py3.*|lhapdf_py3 = $LHAPDF_DIR/bin/lhapdf|" $MG5_DIR/input/mg5_configuration.txt && \
    export PATH=$PYTHON_DIR/bin:$PATH && \
    for d in $SPACK/*/bin; do export PATH=$d:$PATH; done && \
    for d in $SPACK/*/lib; do export LD_LIBRARY_PATH=$d:${LD_LIBRARY_PATH:-}; done && \
    echo "install mg5amc_py8_interface" | $MG5_DIR/bin/mg5_aMC 2>&1 || echo "PY8 interface install attempted" \
'

# =============================================================
# Stage 7: Install the heft model for gg->H processes
# =============================================================
RUN bash -c '\
    source /opt/colliderml/spack_paths.sh && \
    export PATH=$PYTHON_DIR/bin:$PATH && \
    for d in $SPACK/*/bin; do export PATH=$d:$PATH; done && \
    for d in $SPACK/*/lib; do export LD_LIBRARY_PATH=$d:${LD_LIBRARY_PATH:-}; done && \
    echo "install heft" | $MG5_DIR/bin/mg5_aMC 2>&1 || echo "heft install attempted" \
'

# =============================================================
# Stage 8: Copy setup scripts and pipeline config
# =============================================================
COPY setup_env.sh /opt/colliderml/setup_env.sh
COPY env_setup.yaml /opt/colliderml/env_setup.yaml
RUN chmod +x /opt/colliderml/setup_env.sh

# Default: source the environment and drop into bash
ENTRYPOINT ["/bin/bash", "-c", "source /opt/colliderml/setup_env.sh && exec \"$@\"", "--"]
CMD ["bash"]
