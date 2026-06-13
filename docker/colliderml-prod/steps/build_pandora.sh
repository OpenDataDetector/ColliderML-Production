#!/bin/bash
# Clone k4ODD (carries ci/build_pandora_stack.sh + the pinned refs in
# ci/pandora_stack.env) and build the 4-package Pandora stack (PandoraSDK ->
# LCContent -> k4GaudiPandora -> k4DetectorPerformance) against the key4hep shim.
set -e
K4ODD_REF="${K4ODD_REF:-feat/pandora-calibrated-reco}"
git clone --depth 1 --branch "$K4ODD_REF" https://github.com/OpenDataDetector/k4ODD.git /opt/k4ODD
cd /opt/k4ODD
export KEY4HEP_SETUP=/opt/key4hep-shim.sh
export PANDORA_STACK_DIR=/opt/pandora-stack
export PANDORA_STACK_BUILD=/tmp/pandora-build
export CMAKE_BUILD_PARALLEL_LEVEL=4
bash ci/build_pandora_stack.sh
rm -rf /tmp/pandora-build
test -f /opt/pandora-stack/install/setup_stack.sh
echo "Pandora stack built"
