#!/bin/bash
# ODD geometry + libOpenDataDetector.so factory lib, built against the reco image's
# key4hep DD4hep (1.32.1).
#
# Geometry = gitlab.cern.ch/azaborow/OpenDataDetector branch addLayeredCalo_MuonCoil:
# this is the EXACT geometry the calibrated Pandora reco was benchmarked/validated
# against (k4ODD's CI clone_ODD test pins it). It has the layered ECAL/HCAL AND a
# layered MUON calorimeter with the dd4hep::rec::LayeredCalorimeterData extension that
# k4GaudiPandora's DDGeometryCreator requires for MUON_BARREL/ENDCAP. Stock acts ODD
# (v4 / v6.0.2) differs: v4's calos don't set DetType flags at all, and v6.0.2's muon
# is a spectrometer (ODDMuonBarrel) lacking LayeredCalorimeterData -> Pandora FATALs.
#
# pandora_reco.py / calo_digitization.py resolve geometry via ODD_INSTALL_DIR ->
# $ODD_INSTALL_DIR/share/OpenDataDetector/xml/OpenDataDetector.xml; this_odd.sh puts
# libOpenDataDetector.so (the type_flag + extension-setting plugin factory) on
# LD_LIBRARY_PATH.
set -e
source /opt/key4hep-shim.sh
ODD_REPO="${ODD_REPO:-https://gitlab.cern.ch/azaborow/OpenDataDetector.git}"
ODD_REF="${ODD_REF:-addLayeredCalo_MuonCoil}"
git clone --depth 1 --single-branch --branch "$ODD_REF" "$ODD_REPO" /opt/odd-src
cmake -S /opt/odd-src -B /tmp/odd-build \
  -DCMAKE_INSTALL_PREFIX=/opt/odd-install \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build /tmp/odd-build -j6 --target install
rm -rf /tmp/odd-build
test -f /opt/odd-install/lib/libOpenDataDetector.so
test -f /opt/odd-install/share/OpenDataDetector/xml/OpenDataDetector.xml
echo "ODD ${ODD_REF} built -> /opt/odd-install"
