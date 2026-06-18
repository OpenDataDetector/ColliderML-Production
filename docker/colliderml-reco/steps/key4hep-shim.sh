# Native key4hep activation for the RECO image (no cvmfs, no from-source shim).
#
# The reco base (ghcr.io/key4hep/key4hep-sim-reco-ubuntu24) ships the whole
# key4hep stack as a GLOBAL spack instance (ROOT+eve, Gaudi, k4FWCore, podio,
# edm4hep, DD4hep, and the prebuilt LCIO/KalTest/DDKalTest/k4Reco our Pandora
# charged-PF chain links). It has NO merged view, so activation = source spack
# + `spack load` the key4hep-stack bundle (emits PATH / LD_LIBRARY_PATH /
# CMAKE_PREFIX_PATH / PYTHONPATH for the full transitive closure).
#
# Lives at /opt/key4hep-shim.sh: the same path the (sw-image) build steps used,
# so build_pandora.sh (KEY4HEP_SETUP) and build_k4odd.sh source it unchanged,
# and the calo_digitization / pandora_reco runtime env blocks reuse it verbatim.
source /opt/setup_spack.sh
eval "$(spack load --sh key4hep-stack)"
