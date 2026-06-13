#!/bin/bash
# Gaudi v40r2 from source, against the image's spack deps (ROOT/Boost/TBB/fmt/
# CLHEP/edm4hep/podio already present). Approach B1 (no spack concretization).
set -e
GAUDI_REF="${GAUDI_REF:-v40r2}"
source /opt/build-env.sh
git clone --depth 1 --branch "$GAUDI_REF" https://gitlab.cern.ch/gaudi/Gaudi.git /opt/gaudi-src
cmake -S /opt/gaudi-src -B /tmp/gaudi-build -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/gaudi-install \
  -DGAUDI_USE_AIDA=OFF -DGAUDI_USE_XERCESC=OFF -DGAUDI_USE_HEPPDT=OFF \
  -DGAUDI_USE_CPPUNIT=OFF -DGAUDI_USE_GPERFTOOLS=OFF -DGAUDI_USE_DOXYGEN=OFF \
  -DBUILD_TESTING=OFF
cmake --build /tmp/gaudi-build -j4 --target install
rm -rf /tmp/gaudi-build
echo "Gaudi ${GAUDI_REF} built"
