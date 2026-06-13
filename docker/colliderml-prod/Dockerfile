# =============================================================================
# ColliderML production image — Phase 3: the Gaudi/Pandora layer
# =============================================================================
# FROM the Phase-2 base (bc/MG5/ODD/G4/python baked) and add the key4hep/Gaudi
# world the sw image lacks, so ONE image runs the whole pipeline incl. calo
# digitisation + Pandora particle flow:
#   - the full Boost component set Gaudi needs, built into the image's boost prefix
#   - Gaudi + k4FWCore built from source against the image's deps (Approach B1;
#     ROOT/Boost/TBB/fmt/CLHEP/edm4hep/podio already in the image)
#   - a key4hep-style setup shim so our pinned build_pandora_stack.sh treats the
#     in-image stack as a key4hep env (KEY4HEP_SETUP)
#   - PandoraSDK + optimized LCContent + k4GaudiPandora + k4DetectorPerformance
#     (the org forks at the refs in k4ODD ci/pandora_stack.env)
#   - k4ODD itself (libk4ODDPlugins.so) + the calibrated ODDreconstruction.py
#
# Each from-source step lives in steps/*.sh and is run with an explicit `bash`:
# podman-hpc's builder does NOT honour the `SHELL` instruction, so the default
# RUN shell is dash and bare `source` (a bashism) fails with exit 127.
# =============================================================================
FROM localhost/colliderml/prod-base:20260613

ENV SPACK_BASE=/spack/opt/spack/linux-x86_64

# Gaudi header-only deps the image lacks + wget for the boost source fetch.
RUN apt-get update -qq && apt-get install -y -qq \
      libmsgsl-dev librange-v3-dev nlohmann-json3-dev wget && rm -rf /var/lib/apt/lists/*

COPY steps/ /opt/steps/

# --- Boost components Gaudi needs (-> sw PR: boost +fiber +python +log ...) --
RUN bash /opt/steps/build_boost.sh

# --- Gaudi (from source, against the image's deps) --------------------------
RUN bash /opt/steps/build_gaudi.sh

# --- k4FWCore (needs Gaudi + edm4hep + podio) -------------------------------
RUN bash /opt/steps/build_k4fwcore.sh

# --- key4hep setup shim (build-time + runtime) ------------------------------
RUN cp /opt/steps/key4hep-shim.sh /opt/key4hep-shim.sh

# --- Pandora stack (PandoraSDK/LCContent/k4GaudiPandora/k4DetectorPerformance)
RUN bash /opt/steps/build_pandora.sh

# --- k4ODD plugins (libk4ODDPlugins.so) + calibrated options ----------------
RUN bash /opt/steps/build_k4odd.sh

# k4ODD env for the calo/pandora stages: K4ODD_PATH=/opt/k4ODD, ODD already baked.
ENV K4ODD_PATH=/opt/k4ODD
ENV ODD_INSTALL=/opt/colliderml-cache/odd-v4-install

CMD ["/bin/bash"]
