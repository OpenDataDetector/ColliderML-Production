#!/bin/bash
# PandoraSDK, LCContent, k4GaudiPandora and k4DetectorPerformance, installed into one prefix.
#
# PANDORA_VARIANT (from the Containerfile) selects the patches and the compiler flags:
#   optimised  the *_OPT_COMMIT sources (tag colliderml-reco-20261127 of the OpenDataDetector
#              forks), built with PANDORA_CXX_FLAGS and PANDORA_LD_FLAGS.
#              The default flags target x86-64-v3 and keep IEEE-754 results (no FMA
#              contraction, no fast-math).
#   reference  the *_COMMIT sources with the *_reference patches, built as CMake RelWithDebInfo with no extra flags:
#              the sources and flags of the earlier image reco:20261106.
# patches/README.md describes the patches, validation/RESULTS.md how the outputs compare.
#
# CPATH (env/key4hep-shim.sh) also lists the include directories of the base image's own
# PandoraSDK and LCContent. The optimised k4GaudiPandora sources put this prefix ahead of them.
# The reference variant does not, as in reco:20261106: its libk4GaudiPandoraPlugins.so comes
# out bit-identical to the one in that image. k4DetectorPerformance does not use Pandora.
# setup_stack.sh, written at the end, puts this prefix first on CPATH for later builds.
source /opt/colliderml-reco/steps/common.sh
load_key4hep

STACK=/opt/pandora-stack
PREFIX="$STACK/install"
BUILD=/tmp/pandora-build
mkdir -p "$STACK/src" "$PREFIX/lib" "$PREFIX/lib64"

fetch_commit "$K4DETPERFORMANCE_REPO" "$K4DETPERFORMANCE_COMMIT" "$STACK/src/k4DetectorPerformance"

case "${PANDORA_VARIANT:?set by the Containerfile}" in
  optimised)
    : "${PANDORA_CXX_FLAGS:?set by the Containerfile}" "${PANDORA_LD_FLAGS?set by the Containerfile}"
    fetch_commit "$PANDORASDK_REPO" "$PANDORASDK_OPT_COMMIT" "$STACK/src/PandoraSDK"
    fetch_commit "$LCCONTENT_REPO" "$LCCONTENT_OPT_COMMIT" "$STACK/src/LCContent"
    fetch_commit "$K4GAUDIPANDORA_REPO" "$K4GAUDIPANDORA_OPT_COMMIT" "$STACK/src/k4GaudiPandora"
    build_type=Release
    cxx_flags="$PANDORA_CXX_FLAGS"
    ld_flags="$PANDORA_LD_FLAGS"
    ;;
  reference)
    fetch_commit "$PANDORASDK_REPO" "$PANDORASDK_COMMIT" "$STACK/src/PandoraSDK"
    fetch_commit "$LCCONTENT_REPO" "$LCCONTENT_COMMIT" "$STACK/src/LCContent"
    fetch_commit "$K4GAUDIPANDORA_REPO" "$K4GAUDIPANDORA_COMMIT" "$STACK/src/k4GaudiPandora"
    apply_patch "$STACK/src/LCContent" lccontent_reference.patch
    apply_patch "$STACK/src/k4GaudiPandora" k4gaudipandora_reference.patch
    build_type=RelWithDebInfo
    cxx_flags=""
    ld_flags=""
    ;;
  *)
    echo "unknown PANDORA_VARIANT: $PANDORA_VARIANT (optimised or reference)" >&2
    exit 1
    ;;
esac

# The PandoraSDK fork installs PandoraSDKConfig.cmake without a version file, and LCContent
# asks for find_package(PandoraSDK 03.00.00).
cat > "$PREFIX/PandoraSDKConfigVersion.cmake" <<CMAKE
set(PACKAGE_VERSION "${PANDORASDK_VERSION}")
if(PACKAGE_VERSION VERSION_LESS PACKAGE_FIND_VERSION)
  set(PACKAGE_VERSION_COMPATIBLE FALSE)
else()
  set(PACKAGE_VERSION_COMPATIBLE TRUE)
  if(PACKAGE_VERSION VERSION_EQUAL PACKAGE_FIND_VERSION)
    set(PACKAGE_VERSION_EXACT TRUE)
  endif()
endif()
CMAKE

# build_package <name> [extra cmake arguments]
# The install RPATH puts this prefix first: the base image has its own PandoraSDK and
# LCContent, and RPATH is searched before LD_LIBRARY_PATH.
build_package() {
  local name="$1"; shift
  cmake -S "$STACK/src/$name" -B "$BUILD/$name" -GNinja \
    -DCMAKE_BUILD_TYPE="$build_type" \
    -DBUILD_TESTING=OFF \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_PREFIX_PATH="$PREFIX;${CMAKE_PREFIX_PATH//:/;}" \
    -DCMAKE_INSTALL_RPATH="$PREFIX/lib;$PREFIX/lib64" \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
    -DCMAKE_CXX_FLAGS="$cxx_flags" \
    -DCMAKE_SHARED_LINKER_FLAGS="$ld_flags" \
    -DCMAKE_MODULE_LINKER_FLAGS="$ld_flags" \
    "$@"
  cmake --build "$BUILD/$name" -j"$JOBS" --target install
  rm -rf "$BUILD/$name"
}

build_package PandoraSDK
build_package LCContent -DPandoraSDK_DIR="$PREFIX"
build_package k4GaudiPandora
build_package k4DetectorPerformance
rmdir "$BUILD"

cat > "$PREFIX/setup_stack.sh" <<SETUP
# Source after the key4hep environment: puts this Pandora stack ahead of the base image's.
export PANDORA_STACK_PREFIX="$PREFIX"
export CMAKE_PREFIX_PATH="$PREFIX\${CMAKE_PREFIX_PATH:+:\$CMAKE_PREFIX_PATH}"
export CPATH="$PREFIX/include\${CPATH:+:\$CPATH}"
export LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export PYTHONPATH="$PREFIX/python\${PYTHONPATH:+:\$PYTHONPATH}"
SETUP

for lib in libPandoraSDK.so libLCContent.so libk4GaudiPandoraPlugins.so \
           libk4DetectorPerformancePFlowPlugins.so; do
  test -f "$PREFIX/lib/$lib"
done
