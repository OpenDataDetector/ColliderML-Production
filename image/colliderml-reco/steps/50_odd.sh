#!/bin/bash
# Open Data Detector geometry (3 T solenoid) and its DD4hep plugin library.
#
# The pinned commit is the layered-calorimeter geometry the calorimeter and Pandora
# calibration were derived with. It provides the dd4hep::rec::LayeredCalorimeterData
# extensions for ECAL, HCAL, muon system and coil that k4GaudiPandora requires.
# The patch corrects the per-layer radiation and interaction lengths in those extensions
# (the inner quantities were overwritten for every slice of a layer).
#
# Simulation input must come from the same geometry.
source /opt/colliderml-reco/steps/common.sh
load_key4hep

fetch_commit "$ODD_REPO" "$ODD_COMMIT" /opt/odd-src
apply_patch /opt/odd-src odd_calo_layerdata.patch

cmake -S /opt/odd-src -B /tmp/odd-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/odd-install \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build /tmp/odd-build -j"$JOBS" --target install
rm -rf /tmp/odd-build

test -f /opt/odd-install/lib/libOpenDataDetector.so
grep -q 'name="Field_nominal_value" value="3\*tesla"' \
  /opt/odd-install/share/OpenDataDetector/xml/OpenDataDetector.xml
