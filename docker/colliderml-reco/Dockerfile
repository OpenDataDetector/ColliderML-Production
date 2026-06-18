# =============================================================================
# ColliderML RECO image — Container B of the two-container pipeline
# =============================================================================
# FROM the upstream key4hep no-cvmfs sim+reco image (self-contained spack stack:
# ROOT 6.36.04 +eve, Gaudi 40.0, k4FWCore 1.3, podio 1.4.1, edm4hep 0.99.2,
# DD4hep 1.32.1, and the prebuilt LCIO/KalTest/DDKalTest/k4Reco our charged-PF
# Pandora chain links). This image exists BECAUSE the sw image's ROOT 6.38 has
# no Eve component, which KalTest's TRKTrack (-> TEveTrackPropagator) requires.
#
# We bake on top, all against the image's native key4hep stack (no cvmfs):
#   - ODD v4.0.4 geometry + libOpenDataDetector.so (rebuilt vs DD4hep 1.32.1)
#   - the calibrated Pandora stack (PandoraSDK / optimized LCContent /
#     k4GaudiPandora / k4DetectorPerformance) at the pinned org-fork refs
#   - k4ODD plugins (libk4ODDPlugins.so) + the calibrated ODDreconstruction.py /
#     ODDdigitisation.py options
#
# Build steps live in steps/*.sh, run with explicit `bash` (podman-hpc's builder
# uses dash and ignores SHELL; bare `source` is a bashism). Each step is COPYed
# in its own layer right before its RUN so editing a later step does not
# invalidate the (long) earlier build layers.
#
# Build on the LOGIN node: the steps git-clone (k4ODD, ODD, Pandora forks) and
# apt-install, which need internet (compute nodes have none). The build is light
# (4 small Pandora pkgs + k4ODD + ODD; the heavy Gaudi/tracking stack is already
# in the base), so login-node memory is comfortable at -j4..6.
# =============================================================================
FROM ghcr.io/key4hep/key4hep-sim-reco-ubuntu24:reco-image-amd64

# Build tooling the from-source ODD/Pandora/k4ODD builds need (base is minimal).
RUN apt-get update -qq && apt-get install -y -qq \
      git ninja-build build-essential && rm -rf /var/lib/apt/lists/*

# Native key4hep activation, exposed at the path the reused build steps expect.
COPY steps/key4hep-shim.sh /opt/key4hep-shim.sh

# --- k4Reco v0.3.0 GaudiTrkUtils from source (base's spack k4reco ships no CMake export) ---
# Must precede the Pandora build: k4GaudiPandora find_package(k4Reco) needs the real
# k4RecoConfig.cmake + k4Reco::GaudiTrkUtils target this produces in /opt/k4reco-install.
COPY steps/build_k4reco.sh /opt/steps/build_k4reco.sh
RUN bash /opt/steps/build_k4reco.sh

# --- Calibrated Pandora stack (PandoraSDK/LCContent/k4GaudiPandora/k4DetPerf) -
COPY steps/build_pandora.sh /opt/steps/build_pandora.sh
RUN bash /opt/steps/build_pandora.sh

# --- k4ODD plugins (libk4ODDPlugins.so) + calibrated options -----------------
COPY steps/build_k4odd.sh /opt/steps/build_k4odd.sh
RUN bash /opt/steps/build_k4odd.sh

# --- base-image k4FWCore 1.3 <-> Gaudi 40 k4run fix (EventLoopMgr.Warnings) --
COPY steps/patch_k4fwcore.sh /opt/steps/patch_k4fwcore.sh
RUN bash /opt/steps/patch_k4fwcore.sh

# --- ODD v6.0.2 geometry + factory lib (sets calo DetType flags; needs no Pandora,
# so kept LATE to preserve the cached Pandora/k4Reco layers when the ODD ref changes) --
COPY steps/build_odd.sh /opt/steps/build_odd.sh
RUN bash /opt/steps/build_odd.sh

# --- reco-side postproc python libs (convert_pfos / convert_calo_digi) -------
# Best-effort into the key4hep python; the converters are uproot/pyarrow based
# and can also run in the sim image, but bake them so the reco image is
# self-sufficient. --user keeps the read-only spack prefix untouched.
RUN bash -lc 'source /opt/key4hep-shim.sh && \
      python3 -m pip install --no-cache-dir --user \
        pyarrow uproot pandas awkward polars tqdm || \
      echo "WARNING: reco python libs not baked (converters can run in sim image)"'

# Env for the calo_digitization / pandora_reco stages.
ENV K4ODD_PATH=/opt/k4ODD
ENV ODD_INSTALL_DIR=/opt/odd-install

CMD ["/bin/bash"]
