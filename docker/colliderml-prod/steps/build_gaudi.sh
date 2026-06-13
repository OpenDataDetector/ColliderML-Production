#!/bin/bash
# Gaudi v40r2 from source, against the image's spack deps (ROOT/Boost/TBB/fmt/
# CLHEP/edm4hep/podio already present). Approach B1 (no spack concretization).
set -e
GAUDI_REF="${GAUDI_REF:-v40r2}"
source /opt/build-env.sh
# Gaudi's genconf/listcomponents dlopen the freshly-built plugins at build time, so
# the spack runtime libs (boost_context, RIO, python, tbb, fmt) must be on
# LD_LIBRARY_PATH (build-env.sh only sets CMAKE_PREFIX_PATH).
export LD_LIBRARY_PATH="$(ls -d /spack/opt/spack/linux-x86_64/*/lib /spack/opt/spack/linux-x86_64/*/lib64 /spack/opt/spack/linux-x86_64/*/lib/root 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"
git clone --depth 1 --branch "$GAUDI_REF" https://gitlab.cern.ch/gaudi/Gaudi.git /opt/gaudi-src
cmake -S /opt/gaudi-src -B /tmp/gaudi-build -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/gaudi-install \
  -DGAUDI_USE_AIDA=OFF -DGAUDI_USE_XERCESC=OFF -DGAUDI_USE_HEPPDT=OFF \
  -DGAUDI_USE_CPPUNIT=OFF -DGAUDI_USE_GPERFTOOLS=OFF -DGAUDI_USE_DOXYGEN=OFF \
  -DBUILD_TESTING=OFF
cmake --build /tmp/gaudi-build -j4 --target install
rm -rf /tmp/gaudi-build
echo "Gaudi ${GAUDI_REF} built"
