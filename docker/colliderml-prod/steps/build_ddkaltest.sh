#!/bin/bash
# DDKalTest (iLCSoft) — DD4hep adapter for KalTest. Deps: DD4hep + KalTest + LCIO + GSL.
set -e
source /opt/build-env.sh
source /opt/steps/root_fix.sh
export CMAKE_PREFIX_PATH="/opt/lcio-install:/opt/kaltest-install:$CMAKE_PREFIX_PATH"
DDKALTEST_REF="${DDKALTEST_REF:-v01-07-01}"
git clone --depth 1 --branch "$DDKALTEST_REF" https://github.com/iLCSoft/DDKalTest.git /opt/ddkaltest-src
cmake -S /opt/ddkaltest-src -B /tmp/ddkaltest-build -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/ddkaltest-install -DBUILD_TESTING=OFF
cmake --build /tmp/ddkaltest-build -j4 --target install
rm -rf /tmp/ddkaltest-build
echo "DDKalTest ${DDKALTEST_REF} built"
