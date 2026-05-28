# =============================================================================
# ACTS with Apache Arrow / Parquet plugin
# =============================================================================
# Layers Paul Gessinger's stacked PRs onto the existing ColliderML container:
#   - acts-project/acts#5410 (feat/arrow-plugin+conversion) — initial Arrow
#     conversion + Parquet output for particles + simhits + tracks
#   - acts-project/acts#5441 (feat/calo-hit-conversion) — calo hit reading
#     from EDM4hep + Arrow/Parquet writing. Stacked on #5410.
#
# Reuses every spack-built dependency from the base image (DD4hep, Geant4,
# EDM4hep, Pythia8, HepMC3, ROOT, Podio, Boost, etc.). Only adds Apache
# Arrow C++ from Ubuntu's repo and rebuilds ACTS itself with the Arrow
# plugin and the Parquet examples turned on.
#
# Outputs:
#   - /opt/acts-arrow            ACTS install prefix (binaries, libs, python)
#   - /opt/acts-arrow/setup.sh   sourceable activation; prepends PATH +
#                                LD_LIBRARY_PATH + PYTHONPATH ahead of the
#                                spack-shipped ACTS so this build wins.
# =============================================================================
FROM ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0

ARG ACTS_REPO=https://github.com/paulgessinger/acts.git
ARG ACTS_REF=feat/calo-hit-conversion
ARG ACTS_SHA=e485e4cdfd41fb9aa3a9098efbfb229baaa48a36
# Keep parallelism modest — the host running this build has ~7 GB and the
# LTO-heavy pybind11 link steps each consume 2-4 GB. JOBS=2 fits two
# concurrent link processes; higher values trip the kernel OOM killer
# which terminates cc1plus mid-link, leaving make with no usable error.
ARG JOBS=2

# Apache Arrow / Parquet C++ headers + shared libs. Ubuntu 24.04 doesn't
# ship these in the base archive, so we pull them from Apache's own apt repo
# (apache-arrow-apt-source-latest-noble.deb). We deliberately use the
# shared variant and DISABLE ACTS_ARROW_ISOLATED below, since isolated mode
# expects static Arrow. Downstream pyarrow compatibility is verified by
# ACTS's own test_coexist_with_pyarrow.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget gnupg lsb-release git \
 && rm -rf /var/lib/apt/lists/*
# ACTS (Paul's branch) hard-pins ``set(_acts_arrow_version 23.0.0)`` at
# CMakeLists.txt:325. Arrow's CMake config uses SameMajorVersion
# compatibility — 24.x is rejected — so we install 23.0.0 explicitly. The
# Apache noble apt repo keeps a rolling window of recent releases, with the
# 23.x packages tagged as ``23.0.0-1``.
# Arrow 23.0.1 is the latest 23.x in the Apache noble repo; ACTS's
# find_package(Arrow 23.0.0) accepts any 23.x (SameMajorVersion compat).
ARG ARROW_VERSION=23.0.1-1
RUN wget -q "https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb" \
      -O /tmp/apache-arrow-apt-source.deb \
 && apt-get install -y --no-install-recommends /tmp/apache-arrow-apt-source.deb \
 && rm /tmp/apache-arrow-apt-source.deb \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      libarrow-dev=${ARROW_VERSION} \
      libarrow-acero-dev=${ARROW_VERSION} \
      libarrow-compute-dev=${ARROW_VERSION} \
      libparquet-dev=${ARROW_VERSION} \
      libarrow-dataset-dev=${ARROW_VERSION} \
      libthrift-dev \
      libsnappy-dev \
      liblz4-dev \
      libzstd-dev \
      libbz2-dev \
      libutf8proc-dev \
      libre2-dev \
 && rm -rf /var/lib/apt/lists/*

# Use the spack-installed packages as the dependency view. ACTS_USE_SYSTEM_LIBS
# is ON in the original build, so most deps come from CMake's find_package
# walking CMAKE_PREFIX_PATH. We point it at the spack tree.
ENV SPACK_VIEW=/spack/opt/spack/linux-x86_64

# Podio's CMake macros generate C++ from yaml at configure time using the
# spack-shipped python; pyyaml + jinja2 aren't in that python by default.
# The spack layout keeps each package in its own prefix, so the bare
# python-venv-*/bin/python3 doesn't see py-pip's site-packages. Bootstrap
# pip into it via ensurepip and install into the venv directly.
RUN PY=$(ls -d /spack/opt/spack/linux-x86_64/python-venv-*/bin/python3 | head -1) \
 && [ -x "$PY" ] \
 && $PY -m ensurepip --default-pip \
 && $PY -m pip install --no-cache-dir \
      pyyaml \
      jinja2 \
      numpy \
      sympy \
      particle

# Build & install. Configure mirrors the existing spack build args (extracted
# from the container's spack-configure-args.txt), with three additions:
#   ACTS_BUILD_PLUGIN_ARROW=ON       — the new Arrow plugin
#   ACTS_BUILD_EXAMPLES_PARQUET=ON   — wires Arrow output converters + ParquetWriter
#   ACTS_ARROW_ISOLATED=OFF          — link Arrow as shared (matches apt packages)
# Everything else matches the spack-shipped ACTS so downstream code paths
# (DD4hep/EDM4hep input, Geant4 sim, Podio I/O, Python bindings) behave the
# same.
RUN git clone --depth 100 --branch ${ACTS_REF} ${ACTS_REPO} /opt/acts-arrow-src \
 && cd /opt/acts-arrow-src \
 && git checkout ${ACTS_SHA}

# ACTS ships an in-tree Python package ``codegen/`` whose sympy-based scripts
# run at compile time to generate C++ math headers. Install it into the spack
# python so the generators can import their helper module.
RUN PY=$(ls -d /spack/opt/spack/linux-x86_64/python-venv-*/bin/python3 | head -1) \
 && $PY -m pip install --no-cache-dir /opt/acts-arrow-src/codegen

# Build the spack-deps prefix list deterministically so CMake finds every
# dep from the existing image.
RUN bash -c 'set -euo pipefail; \
    PREFIX_LIST=$(find ${SPACK_VIEW} -maxdepth 1 -mindepth 1 -type d | tr "\n" ";"); \
    echo "$PREFIX_LIST" > /opt/acts-arrow-cmake-prefix.txt; \
    wc -l /opt/acts-arrow-cmake-prefix.txt; head -c 300 /opt/acts-arrow-cmake-prefix.txt; echo'

RUN mkdir -p /opt/acts-arrow-build && cd /opt/acts-arrow-build \
 && cmake /opt/acts-arrow-src \
    -DCMAKE_PREFIX_PATH="$(cat /opt/acts-arrow-cmake-prefix.txt)" \
    -DLZ4_LIB=/usr/lib/x86_64-linux-gnu/liblz4.so \
    -DZSTD_LIB=/usr/lib/x86_64-linux-gnu/libzstd.so \
    -DSnappy_LIB=/usr/lib/x86_64-linux-gnu/libsnappy.so \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_INSTALL_PREFIX=/opt/acts-arrow \
    -DACTS_USE_SYSTEM_LIBS=ON \
    -DACTS_BUILD_EXAMPLES=ON \
    -DACTS_BUILD_EXAMPLES_DD4HEP=ON \
    -DACTS_BUILD_PLUGIN_DD4HEP=ON \
    -DACTS_BUILD_EXAMPLES_EDM4HEP=ON \
    -DACTS_BUILD_PLUGIN_EDM4HEP=ON \
    -DACTS_BUILD_PLUGIN_PODIO=ON \
    -DACTS_BUILD_EXAMPLES_GEANT4=ON \
    -DACTS_BUILD_PLUGIN_GEANT4=ON \
    -DACTS_BUILD_EXAMPLES_HEPMC3=ON \
    -DACTS_BUILD_EXAMPLES_PYTHIA8=ON \
    -DACTS_BUILD_PLUGIN_JSON=ON \
    -DACTS_BUILD_PLUGIN_TGEO=ON \
    -DACTS_BUILD_FATRAS=ON \
    -DACTS_BUILD_PLUGIN_ARROW=ON \
    -DACTS_BUILD_EXAMPLES_PARQUET=ON \
    -DACTS_ARROW_ISOLATED=OFF \
    -DACTS_BUILD_PYTHON_BINDINGS=ON \
    -DACTS_BUILD_EXAMPLES_PYTHON_BINDINGS=ON \
    -DACTS_LOG_FAILURE_THRESHOLD=MAX \
    -DACTS_ENABLE_LOG_FAILURE_THRESHOLD=ON \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
    -DPYBIND11_LTO_CXX_FLAGS=""

# Avoid `| tail -200`: shell pipefail is off by default, so a pipe would
# mask gmake's exit code and silently produce a half-installed image.
#
# MAKEFLAGS: GNU make 4.4 switched the jobserver from anonymous pipes to
# named FIFOs, but the gcc 13 ``lto-wrapper`` (invoked when pybind11
# enables ``-flto`` on python-module links) can't parse the FIFO form
# (gcc bug #111527). Forcing the pipe variant restores compatibility
# without giving up LTO or parallel make.
RUN MAKEFLAGS="--jobserver-style=pipe" \
    cmake --build /opt/acts-arrow-build --target install --parallel ${JOBS}

# Drop the build tree (saves ~3 GB) but keep the source for traceability.
RUN rm -rf /opt/acts-arrow-build

# Activation script: when sourced, picks the rebuilt ACTS over the spack one.
RUN cat > /opt/acts-arrow/setup.sh <<'EOF'
# Source this to use the Arrow-enabled ACTS instead of the spack-shipped build.
ACTS_ARROW=/opt/acts-arrow
# The ACTS python bindings are .cpython-313-* — they only load under the
# spack-installed python 3.13. Prepend its bin to PATH so ``python3`` resolves
# to the right interpreter for anyone who sources this script.
SPACK_PY_BIN=$(ls -d /spack/opt/spack/linux-x86_64/python-venv-*/bin 2>/dev/null | head -1)
if [ -n "$SPACK_PY_BIN" ]; then
    export PATH="$SPACK_PY_BIN:$PATH"
fi
export PATH="$ACTS_ARROW/bin:$PATH"
export LD_LIBRARY_PATH="$ACTS_ARROW/lib:$ACTS_ARROW/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ACTS_ARROW/python:${PYTHONPATH:-}"
export Acts_DIR="$ACTS_ARROW/lib/cmake/Acts"
echo "INFO:    ACTS with Arrow plugin active at $ACTS_ARROW (python $($SPACK_PY_BIN/python3 --version 2>/dev/null))"
EOF

# Smoke test: ACTS import + arrow plugin import should both succeed.
RUN bash -c 'source /opt/acts-arrow/setup.sh \
 && python3 -c "import acts; import acts.examples; import acts.examples.arrow; \
                import acts.arrow; \
                print(\"ACTS\", acts.__version__); \
                print(\"simHitSchema\", acts.arrow.simHitSchema()); \
                print(\"particleSchema\", acts.arrow.particleSchema().names);"'

CMD ["/bin/bash"]
