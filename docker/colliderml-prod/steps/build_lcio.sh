#!/bin/bash
# LCIO (iLCSoft) — needed transitively: k4GaudiPandora's DDTrackCreatorBase uses
# k4Reco::GaudiTrkUtils, which links LCIO::lcio + KalTest + DDKalTest. The sw image
# ships none of these legacy tracking packages, so build them from source against the
# image's ROOT/CLHEP/DD4hep. (-> sw PR: add lcio/kaltest/ddkaltest/k4reco to spack.)
set -e
source /opt/build-env.sh
source /opt/steps/root_fix.sh
LCIO_REF="${LCIO_REF:-v02-23-02}"
git clone --depth 1 --branch "$LCIO_REF" https://github.com/iLCSoft/LCIO.git /opt/lcio-src
cmake -S /opt/lcio-src -B /tmp/lcio-build -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/lcio-install \
  -DBUILD_TESTING=OFF -DINSTALL_DOC=OFF -DBUILD_ROOTDICT=ON -DBUILD_LCIO_EXAMPLES=OFF
cmake --build /tmp/lcio-build -j4 --target install
rm -rf /tmp/lcio-build
test -f /opt/lcio-install/lib/cmake/LCIO/LCIOConfig.cmake || test -f /opt/lcio-install/lib64/cmake/LCIO/LCIOConfig.cmake
echo "LCIO ${LCIO_REF} built"
