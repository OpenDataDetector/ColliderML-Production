#!/bin/bash
# KalTest (iLCSoft) — Kalman-filter track fitting core; needed by DDKalTest ->
# k4Reco::GaudiTrkUtils -> k4GaudiPandora. Deps: ROOT (+CLHEP), both in the image.
set -e
source /opt/build-env.sh
source /opt/steps/root_fix.sh
KALTEST_REF="${KALTEST_REF:-v02-05-02}"
git clone --depth 1 --branch "$KALTEST_REF" https://github.com/iLCSoft/KalTest.git /opt/kaltest-src
cmake -S /opt/kaltest-src -B /tmp/kaltest-build -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/kaltest-install -DBUILD_TESTING=OFF
cmake --build /tmp/kaltest-build -j4 --target install
rm -rf /tmp/kaltest-build
echo "KalTest ${KALTEST_REF} built"
