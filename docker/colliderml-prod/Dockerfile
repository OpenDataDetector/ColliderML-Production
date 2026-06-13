# =============================================================================
# ColliderML production image — Phase 3: the Gaudi/Pandora layer
# =============================================================================
# FROM the Phase-2 base (bc/MG5/ODD/G4/python baked) and add the key4hep/Gaudi
# world the sw image lacks, so ONE image runs the whole pipeline incl. calo
# digitisation + Pandora particle flow:
#   - Gaudi + k4FWCore built from source against the image's deps (Approach B1;
#     ROOT/Boost/TBB/fmt/CLHEP/edm4hep/podio already in the image)
#   - a key4hep-style setup shim so our pinned build_pandora_stack.sh treats the
#     in-image stack as a key4hep env (KEY4HEP_SETUP)
#   - PandoraSDK + optimized LCContent + k4GaudiPandora + k4DetectorPerformance
#     (the org forks at the refs in k4ODD ci/pandora_stack.env)
#   - k4ODD itself (libk4ODDPlugins.so) + the calibrated ODDreconstruction.py
# =============================================================================
FROM localhost/colliderml/prod-base:20260613

ENV SPACK_BASE=/spack/opt/spack/linux-x86_64

# Gaudi header-only deps the image lacks.
RUN apt-get update -qq && apt-get install -y -qq \
      libmsgsl-dev librange-v3-dev nlohmann-json3-dev && rm -rf /var/lib/apt/lists/*

# --- Gaudi (from source, against the image's deps) --------------------------
ARG GAUDI_REF=v40r2
RUN set -e; source /opt/build-env.sh; \
    git clone --depth 1 --branch ${GAUDI_REF} https://gitlab.cern.ch/gaudi/Gaudi.git /opt/gaudi-src; \
    cmake -S /opt/gaudi-src -B /tmp/gaudi-build -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/gaudi-install \
      -DGAUDI_USE_AIDA=OFF -DGAUDI_USE_XERCESC=OFF -DGAUDI_USE_HEPPDT=OFF \
      -DGAUDI_USE_CPPUNIT=OFF -DGAUDI_USE_GPERFTOOLS=OFF -DGAUDI_USE_DOXYGEN=OFF \
      -DBUILD_TESTING=OFF; \
    cmake --build /tmp/gaudi-build -j6 --target install; \
    rm -rf /tmp/gaudi-build; echo "Gaudi ${GAUDI_REF} built"

# --- k4FWCore (needs Gaudi + edm4hep + podio) -------------------------------
RUN set -e; source /opt/build-env.sh; export CMAKE_PREFIX_PATH="/opt/gaudi-install:$CMAKE_PREFIX_PATH"; \
    git clone --depth 1 https://github.com/key4hep/k4FWCore.git /opt/k4fwcore-src; \
    cmake -S /opt/k4fwcore-src -B /tmp/k4fw-build -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/k4fwcore-install -DBUILD_TESTING=OFF; \
    cmake --build /tmp/k4fw-build -j6 --target install; \
    rm -rf /tmp/k4fw-build; echo "k4FWCore built"

# --- key4hep setup shim: makes the in-image stack look like a key4hep env so
# build_pandora_stack.sh (which sources $KEY4HEP_SETUP) finds Gaudi/k4FWCore/
# edm4hep/podio/dd4hep/root on CMAKE_PREFIX_PATH ----------------------------
RUN cat > /opt/key4hep-shim.sh <<'EOF'
source /opt/build-env.sh
export CMAKE_PREFIX_PATH="/opt/gaudi-install:/opt/k4fwcore-install:$CMAKE_PREFIX_PATH"
export LD_LIBRARY_PATH="/opt/gaudi-install/lib:/opt/k4fwcore-install/lib:${LD_LIBRARY_PATH:-}"
export PATH="/opt/gaudi-install/bin:/opt/k4fwcore-install/bin:$PATH"
_pyv=$(ls -d /spack/opt/spack/linux-x86_64/python-venv-*/bin 2>/dev/null | head -1)
[ -n "$_pyv" ] && export PATH="$_pyv:$PATH"
export PYTHONPATH="/opt/k4fwcore-install/python:${PYTHONPATH:-}"
EOF

# --- Pandora stack: clone k4ODD (carries ci/build_pandora_stack.sh + the
# pinned refs) and build the 4-package stack against the shim --------------
ARG K4ODD_REF=feat/pandora-calibrated-reco
RUN set -e; \
    git clone --depth 1 --branch ${K4ODD_REF} https://github.com/OpenDataDetector/k4ODD.git /opt/k4ODD; \
    cd /opt/k4ODD; \
    export KEY4HEP_SETUP=/opt/key4hep-shim.sh; \
    export PANDORA_STACK_DIR=/opt/pandora-stack PANDORA_STACK_BUILD=/tmp/pandora-build CMAKE_BUILD_PARALLEL_LEVEL=6; \
    bash ci/build_pandora_stack.sh; \
    rm -rf /tmp/pandora-build; \
    test -f /opt/pandora-stack/install/setup_stack.sh && echo "Pandora stack built"

# --- k4ODD plugins (libk4ODDPlugins.so) + calibrated options ----------------
RUN set -e; source /opt/key4hep-shim.sh; source /opt/pandora-stack/install/setup_stack.sh; \
    cmake -S /opt/k4ODD -B /tmp/k4odd-build -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/k4ODD/install; \
    cmake --build /tmp/k4odd-build -j6 --target install; \
    rm -rf /tmp/k4odd-build; echo "k4ODD built"

# k4ODD env for the calo/pandora stages: K4ODD_PATH=/opt/k4ODD, ODD already baked.
ENV K4ODD_PATH=/opt/k4ODD
ENV ODD_INSTALL=/opt/colliderml-cache/odd-v4-install

CMD ["/bin/bash"]
