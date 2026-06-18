#!/bin/bash
# ODD v6.0.2 geometry + libOpenDataDetector.so factory lib, built against the reco
# image's key4hep DD4hep (1.32.1).
#
# Why v6 (not v4.0.4): v4's calorimeters use the stock DD4hep_PolyhedraBarrelCalorimeter2
# plugin, which does NOT attach DetType type_flags — so under DD4hep 1.32.1 every calo
# DetElement loads with typeFlag=0x0 and Pandora's getExtension(CALORIMETER|BARREL|EM)
# finds nothing -> no PFOs. v6 ships its OWN ODDPolyhedraBarrelCalorimeter plugin (the
# merged addLayeredCalo work) that sets the flags correctly (ECalBarrel -> 0x812 etc.),
# which is exactly what the calibrated Pandora reco needs.
#
# pandora_reco.py / calo_digitization.py resolve geometry via ODD_INSTALL_DIR ->
# $ODD_INSTALL_DIR/share/OpenDataDetector/xml/OpenDataDetector.xml, and this_odd.sh
# puts libOpenDataDetector.so (the type_flag-setting plugin factory) on LD_LIBRARY_PATH.
set -e
source /opt/key4hep-shim.sh
ODD_REF="${ODD_REF:-v6.0.2}"
git clone --depth 1 --branch "$ODD_REF" https://gitlab.cern.ch/acts/OpenDataDetector.git /opt/odd-src
cmake -S /opt/odd-src -B /tmp/odd-build \
  -DCMAKE_INSTALL_PREFIX=/opt/odd-install \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build /tmp/odd-build -j6 --target install
rm -rf /tmp/odd-build
test -f /opt/odd-install/lib/libOpenDataDetector.so
test -f /opt/odd-install/share/OpenDataDetector/xml/OpenDataDetector.xml
echo "ODD ${ODD_REF} built -> /opt/odd-install"
