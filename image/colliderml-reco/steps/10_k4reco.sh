#!/bin/bash
# k4Reco GaudiTrkUtils (the DDKalTest track refit that k4GaudiPandora's track creators call).
#
# The base image ships k4Reco 0.2.1, whose GaudiDDKalTestTrack interface takes
# edm4hep::TrackerHitPlane where this k4GaudiPandora needs edm4hep::TrackerHit, and which
# installs no CMake package config. The patch builds only the GaudiTrkUtils library of v0.3.0:
# the k4RecoPlugins module needs k4FWCore 1.4 (the base has 1.3) and Pandora does not use it.
source /opt/colliderml-reco/steps/common.sh
load_key4hep

fetch_commit "$K4RECO_REPO" "$K4RECO_COMMIT" /opt/k4reco-src
apply_patch /opt/k4reco-src k4reco_gauditrkutils_only.patch

cmake -S /opt/k4reco-src -B /tmp/k4reco-build -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/k4reco-install \
  -DPython_EXECUTABLE="$(command -v python3)" \
  -DBUILD_TESTING=OFF
cmake --build /tmp/k4reco-build -j"$JOBS" --target install
rm -rf /tmp/k4reco-build

test -f /opt/k4reco-install/lib/libGaudiTrkUtils.so
test -f /opt/k4reco-install/lib/cmake/k4Reco/k4RecoConfig.cmake
test -f /opt/k4reco-install/include/k4Reco/GaudiDDKalTest.h
