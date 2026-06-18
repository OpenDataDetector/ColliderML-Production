#!/bin/bash
# Clone k4ODD (carries ci/build_pandora_stack.sh + the pinned refs in
# ci/pandora_stack.env) and build the 4-package Pandora stack (PandoraSDK ->
# LCContent -> k4GaudiPandora -> k4DetectorPerformance) against the key4hep shim.
set -e
K4ODD_REF="${K4ODD_REF:-feat/pandora-calibrated-reco}"
git clone --depth 1 --branch "$K4ODD_REF" https://github.com/OpenDataDetector/k4ODD.git /opt/k4ODD
cd /opt/k4ODD

# k4GaudiPandora's test/ subdir requires k4geo (a key4hep geometry package we don't
# fold in). It is guarded by BUILD_TESTING, but the stack builder doesn't disable
# tests. Patch its cmake invocation to add -DBUILD_TESTING=OFF for all packages.
sed -i 's#-DCMAKE_BUILD_TYPE=RelWithDebInfo \\#-DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=OFF \\#' ci/build_pandora_stack.sh
grep -q "BUILD_TESTING=OFF" ci/build_pandora_stack.sh && echo "patched build_pandora_stack.sh: tests off"

export KEY4HEP_SETUP=/opt/key4hep-shim.sh
export PANDORA_STACK_DIR=/opt/pandora-stack
export PANDORA_STACK_BUILD=/tmp/pandora-build
export CMAKE_BUILD_PARALLEL_LEVEL=4

# The PandoraSDK org-fork hand-rolls a minimal PandoraSDKConfig.cmake but ships NO
# PandoraSDKConfigVersion.cmake (it relied on a stock cvmfs PandoraSDK to supply the
# version). Without cvmfs, LCContent's find_package(PandoraSDK 03.00.00 REQUIRED)
# sees version "unknown" and rejects it. Pre-seed a ConfigVersion in the prefix
# (version derived from the pinned PANDORASDK_REF). k4GaudiPandora/k4DetectorPerformance
# request no version, so this is the only version check in the chain.
PSDK_VER=$(grep 'PANDORASDK_REF' ci/pandora_stack.env | grep -oE 'v[0-9]+-[0-9]+-[0-9]+' | head -1 | tr -d v | tr - .)
[ -z "$PSDK_VER" ] && PSDK_VER="03.04.01"
# Pre-create install/lib64: build_pandora_stack.sh runs `set -eo pipefail` and ends with
# `find "$prefix/lib" "$prefix/lib64" ... 2>/dev/null` to list the install tree. On this
# Ubuntu/key4hep base CMake installs to lib/ (no lib64/), so find exits 1 on the missing
# path (2>/dev/null hides the message, not the exit code) and set -e kills the script
# right BEFORE it writes setup_stack.sh — leaving a fully-built stack with no setup file.
# An empty lib64/ lets the listing find succeed.
mkdir -p /opt/pandora-stack/install/lib64
cat > /opt/pandora-stack/install/PandoraSDKConfigVersion.cmake <<EOF
set(PACKAGE_VERSION "${PSDK_VER}")
if(PACKAGE_VERSION VERSION_LESS PACKAGE_FIND_VERSION)
  set(PACKAGE_VERSION_COMPATIBLE FALSE)
else()
  set(PACKAGE_VERSION_COMPATIBLE TRUE)
  if(PACKAGE_VERSION VERSION_EQUAL PACKAGE_FIND_VERSION)
    set(PACKAGE_VERSION_EXACT TRUE)
  endif()
endif()
EOF
echo "pre-seeded PandoraSDKConfigVersion.cmake version=${PSDK_VER}"

bash ci/build_pandora_stack.sh
rm -rf /tmp/pandora-build
test -f /opt/pandora-stack/install/setup_stack.sh
echo "Pandora stack built"
