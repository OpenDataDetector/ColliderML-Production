#!/bin/bash
# k4Reco (key4hep) — we only need k4Reco::GaudiTrkUtils (the DDKalTest-based track
# refit that k4GaudiPandora's DDTrackCreatorBase uses). Build against the image's
# Gaudi/k4FWCore/edm4hep/DD4hep + the just-built LCIO/KalTest/DDKalTest.
#
# Patch: drop the top-level `find_package(k4SimGeant4 REQUIRED)` — k4SimGeant4 is the
# Geant4 sim wrapper (needs edm4hep 1.0) and is NOT referenced anywhere in the k4Reco
# source (verified); only its find_package gates configuration. Removing it lets
# GaudiTrkUtils (+ the reco plugins) build against the image's edm4hep 0.99.2.
set -e
source /opt/key4hep-shim.sh
export CMAKE_PREFIX_PATH="/opt/lcio-install:/opt/kaltest-install:/opt/ddkaltest-install:$CMAKE_PREFIX_PATH"
PYBIN=$(ls -d /spack/opt/spack/linux-x86_64/python-3.13*/bin | head -1); export PATH="$PYBIN:$PATH"
PYEXE="$PYBIN/python3"

K4RECO_REF="${K4RECO_REF:-main}"
git clone --depth 1 --branch "$K4RECO_REF" https://github.com/key4hep/k4Reco.git /opt/k4reco-src

# Drop the unused k4SimGeant4 hard requirement (Geant4 sim wrapper; not used in src).
sed -i 's/^[[:space:]]*find_package(k4SimGeant4 REQUIRED)/# k4SimGeant4 dropped: not used by GaudiTrkUtils/' /opt/k4reco-src/CMakeLists.txt
grep -q "k4SimGeant4 dropped" /opt/k4reco-src/CMakeLists.txt && echo "patched: dropped k4SimGeant4 requirement"

cmake -S /opt/k4reco-src -B /tmp/k4reco-build -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/k4reco-install \
  -DPython_EXECUTABLE="$PYEXE" -DPython3_EXECUTABLE="$PYEXE" \
  -DBUILD_TESTING=OFF
cmake --build /tmp/k4reco-build -j4 --target install
rm -rf /tmp/k4reco-build
# confirm GaudiTrkUtils library + its k4Reco export target exist
find /opt/k4reco-install -name "libGaudiTrkUtils*" 2>/dev/null | head -1
test -f /opt/k4reco-install/lib/cmake/k4Reco/k4RecoConfig.cmake || test -f /opt/k4reco-install/lib64/cmake/k4Reco/k4RecoConfig.cmake
echo "k4Reco ${K4RECO_REF} (GaudiTrkUtils) built"
