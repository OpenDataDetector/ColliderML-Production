#!/bin/bash
# k4Reco v0.3.0 GaudiTrkUtils (DDKalTest-based track refit) FROM SOURCE.
#
# Why v0.3.0 and source-built (the reco base ships spack k4reco-0.2.1):
#   - API match: our calibrated k4GaudiPandora fork (colliderml-odd-edits.1) calls
#     GaudiDDKalTestTrack::addHit(edm4hep::TrackerHit*) and propagateToLayer(unsigned,
#     edm4hep::TrackState&, ...). That signature exists in k4Reco main/v0.3.0; v0.2.x
#     uses edm4hep::TrackerHitPlane* and does NOT match -> compile errors.
#   - CMake export: stock k4Reco comments out the include() that generates
#     k4RecoConfig.cmake, so find_package(k4Reco) + k4Reco::GaudiTrkUtils cannot resolve.
# So build v0.3.0 against the spack base, with these patches:
#   1. enable the commented-out CreateProjectConfig include -> generates k4RecoConfig.cmake
#   2. relax find_package(k4FWCore 1.4): v0.3.0 PINS k4FWCore 1.4 but the base ships 1.3.
#      Only the k4RecoPlugins *components* use the 1.4 Functional API; GaudiTrkUtils does not.
#   3. drop the k4RecoPlugins module entirely (it needs k4FWCore 1.4 + isn't used by Pandora)
#      and the python confdb subdir, leaving just the GaudiTrkUtils library + its export.
#   4. drop the unused find_package(k4SimGeant4) (Geant4 sim wrapper, wants edm4hep 1.0).
# v0.3.0 already installs headers flat (BASE_DIRS) and has no COMPONENT-dev export bug.
set -e
source /opt/setup_spack.sh
eval "$(spack load --sh key4hep-stack)"
PYEXE=$(command -v python3)

K4RECO_REF="${K4RECO_REF:-v0.3.0}"
git clone --depth 1 --branch "$K4RECO_REF" https://github.com/key4hep/k4Reco.git /opt/k4reco-src

sed -i 's/^[[:space:]]*find_package(k4SimGeant4 REQUIRED)/# k4SimGeant4 dropped: not used by GaudiTrkUtils/' \
  /opt/k4reco-src/CMakeLists.txt || true
sed -i 's|^# *include(cmake/CreateProjectConfig.cmake)|include(cmake/CreateProjectConfig.cmake)|' \
  /opt/k4reco-src/CMakeLists.txt
grep -q '^include(cmake/CreateProjectConfig.cmake)' /opt/k4reco-src/CMakeLists.txt \
  && echo "patched: enabled k4Reco config export"
sed -i -E 's/find_package\(k4FWCore [0-9.]+/find_package(k4FWCore/' /opt/k4reco-src/CMakeLists.txt
sed -i 's|^add_subdirectory(python)|# python confdb subdir dropped (no plugins module)|' \
  /opt/k4reco-src/CMakeLists.txt

# Strip the k4RecoPlugins module (k4FWCore-1.4 Functional API) + drop it from the export.
python3 - <<'PY'
import re
f = "/opt/k4reco-src/k4Reco/CMakeLists.txt"
s = open(f).read()
s = re.sub(r"gaudi_add_module\(k4RecoPlugins.*?(?=install\(TARGETS)", "", s, flags=re.DOTALL)
s = s.replace("install(TARGETS GaudiTrkUtils k4RecoPlugins", "install(TARGETS GaudiTrkUtils")
open(f, "w").write(s)
print("patched: k4RecoPlugins module removed (build GaudiTrkUtils only)")
PY

cmake -S /opt/k4reco-src -B /tmp/k4reco-build -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/k4reco-install \
  -DPython_EXECUTABLE="$PYEXE" -DBUILD_TESTING=OFF
cmake --build /tmp/k4reco-build -j4 --target install
rm -rf /tmp/k4reco-build

find /opt/k4reco-install -name "libGaudiTrkUtils*" | head -1
test -f /opt/k4reco-install/lib/cmake/k4Reco/k4RecoConfig.cmake || \
  test -f /opt/k4reco-install/lib64/cmake/k4Reco/k4RecoConfig.cmake
test -e /opt/k4reco-install/include/k4Reco/GaudiDDKalTest.h
echo "k4Reco ${K4RECO_REF} (GaudiTrkUtils) built -> /opt/k4reco-install"
