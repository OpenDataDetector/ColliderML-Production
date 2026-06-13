#!/bin/bash
# k4ODD plugins (libk4ODDPlugins.so + libddODD) against Gaudi/k4FWCore/EDM4HEP/DD4hep
# (shim) + the pinned Pandora stack. Options live in the source clone at
# /opt/k4ODD/k4ODD/options (resolved at runtime via K4ODD_PATH=/opt/k4ODD).
set -e
source /opt/key4hep-shim.sh
source /opt/pandora-stack/install/setup_stack.sh
cmake -S /opt/k4ODD -B /tmp/k4odd-build -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/k4ODD/install
cmake --build /tmp/k4odd-build -j4 --target install
rm -rf /tmp/k4odd-build
echo "k4ODD built"
