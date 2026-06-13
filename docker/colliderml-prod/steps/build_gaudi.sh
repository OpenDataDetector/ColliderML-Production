#!/bin/bash
# Gaudi v40r2 from source, against the image's spack deps (ROOT/Boost/TBB/fmt/
# CLHEP/edm4hep/podio already present). Approach B1 (no spack concretization).
set -e
GAUDI_REF="${GAUDI_REF:-v40r2}"
source /opt/build-env.sh
# Gaudi's genconf/listcomponents dlopen the freshly-built plugins at build time, so
# the spack runtime libs (boost_context, RIO, python, tbb, fmt) must be on
# LD_LIBRARY_PATH (build-env.sh only sets CMAKE_PREFIX_PATH).
export LD_LIBRARY_PATH="$(ls -d /spack/opt/spack/linux-x86_64/*/lib /spack/opt/spack/linux-x86_64/*/lib64 /spack/opt/spack/linux-x86_64/*/lib/root 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"
# ROOTSYS + module-path fix so Gaudi genconf's cling init finds libc.pcm.
source /opt/steps/root_fix.sh
git clone --depth 1 --branch "$GAUDI_REF" https://gitlab.cern.ch/gaudi/Gaudi.git /opt/gaudi-src

# .confdb2 merge backend: Gaudi's merge_confdb2_parts forces the reliable dbm.gnu
# ONLY for python 3.13.0-3.13.7 (assuming a cpython fix in 3.13.8+). Our image
# python is 3.13.11, where shelve falls through to the new dbm.sqlite3 default,
# which is fragile for confdb2 (sqlite locking on overlay/Lustre -> the merge can
# silently produce no Gaudi.confdb2 and the install then fails). Extend Gaudi's own
# gdbm workaround to all linux pythons.
_MERGE=/opt/gaudi-src/GaudiKernel/scripts/merge_confdb2_parts
sed -i 's/(3, 13, 0) <= sys.version_info <= (3, 13, 7) or sys.version_info\[:3\] == (3, 14, 0)/True/' "$_MERGE" || true
grep -q 'dbm._defaultmod = dbm.gnu' "$_MERGE" && echo "patched merge_confdb2_parts to force dbm.gnu"

cmake -S /opt/gaudi-src -B /tmp/gaudi-build -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/gaudi-install \
  -DGAUDI_USE_AIDA=OFF -DGAUDI_USE_XERCESC=OFF -DGAUDI_USE_HEPPDT=OFF \
  -DGAUDI_USE_CPPUNIT=OFF -DGAUDI_USE_GPERFTOOLS=OFF -DGAUDI_USE_DOXYGEN=OFF \
  -DBUILD_TESTING=OFF
# Build everything (compile + genconf + .confdb2 merge), but DON'T install yet:
# the spack/buildcache python's dbm can land the merged DB at a name other than
# the exact Gaudi.confdb2 the install step expects, so we normalize first.
cmake --build /tmp/gaudi-build -j4

echo "=== .confdb2 artifacts after build ==="
ls -la /tmp/gaudi-build/Gaudi.confdb2* 2>/dev/null || echo "  (no /tmp/gaudi-build/Gaudi.confdb2*)"
find /tmp/gaudi-build -maxdepth 1 -name '*.confdb2*' 2>/dev/null

# The install expects exactly /tmp/gaudi-build/Gaudi.confdb2. If the merge produced
# a suffixed/alternate name, normalize it; if it's missing entirely, re-run the
# merge target explicitly so any error is visible.
if [ ! -f /tmp/gaudi-build/Gaudi.confdb2 ]; then
  if [ -f /tmp/gaudi-build/Gaudi.confdb2.db ]; then
    cp -f /tmp/gaudi-build/Gaudi.confdb2.db /tmp/gaudi-build/Gaudi.confdb2
    echo "normalized Gaudi.confdb2.db -> Gaudi.confdb2"
  else
    echo "Gaudi.confdb2 missing - re-running merge target explicitly:"
    cmake --build /tmp/gaudi-build --target Gaudi_MergeConfDB2 || true
    ls -la /tmp/gaudi-build/Gaudi.confdb2* 2>/dev/null || echo "  still none"
    [ -f /tmp/gaudi-build/Gaudi.confdb2.db ] && cp -f /tmp/gaudi-build/Gaudi.confdb2.db /tmp/gaudi-build/Gaudi.confdb2
  fi
fi

cmake --install /tmp/gaudi-build
test -f /opt/gaudi-install/lib/libGaudiKernel.so
rm -rf /tmp/gaudi-build
echo "Gaudi ${GAUDI_REF} built"
