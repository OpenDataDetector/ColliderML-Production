#!/bin/bash
# iLCUtil (ILCUTIL) — iLCSoft cmake macros + streamlog logging library, required by
# KalTest and DDKalTest (find_package(ILCUTIL)). Tiny, no exotic deps.
set -e
source /opt/build-env.sh
source /opt/steps/root_fix.sh
ILCUTIL_REF="${ILCUTIL_REF:-v01-09}"
git clone --depth 1 --branch "$ILCUTIL_REF" https://github.com/iLCSoft/iLCUtil.git /opt/ilcutil-src
cmake -S /opt/ilcutil-src -B /tmp/ilcutil-build -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/ilcutil-install -DBUILD_TESTING=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build /tmp/ilcutil-build -j4 --target install
rm -rf /tmp/ilcutil-build
echo "iLCUtil ${ILCUTIL_REF} built"
