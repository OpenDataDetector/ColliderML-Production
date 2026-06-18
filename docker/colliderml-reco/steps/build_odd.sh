#!/bin/bash
# ODD v4.0.4 geometry + libOpenDataDetector.so factory lib, built against the
# reco image's key4hep DD4hep (1.32.1) so the cellID readout/decoders match the
# stack k4run/ddsim load. pandora_reco.py / calo_digitization.py resolve geometry
# via ODD_INSTALL_DIR -> $ODD_INSTALL_DIR/share/OpenDataDetector/xml/OpenDataDetector.xml.
set -e
source /opt/key4hep-shim.sh
ODD_REF="${ODD_REF:-v4.0.4}"
git clone --depth 1 --branch "$ODD_REF" https://gitlab.cern.ch/acts/OpenDataDetector.git /opt/odd-src
cmake -S /opt/odd-src -B /tmp/odd-build \
  -DCMAKE_INSTALL_PREFIX=/opt/odd-v4-install \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build /tmp/odd-build -j6 --target install
rm -rf /tmp/odd-build
test -f /opt/odd-v4-install/lib/libOpenDataDetector.so
test -f /opt/odd-v4-install/share/OpenDataDetector/xml/OpenDataDetector.xml
echo "ODD ${ODD_REF} built -> /opt/odd-v4-install"
