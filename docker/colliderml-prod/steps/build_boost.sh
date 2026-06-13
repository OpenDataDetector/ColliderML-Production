#!/bin/bash
# Build the full Boost component set Gaudi v40r2 needs INTO the image's existing
# spack Boost 1.88 prefix (one boost, same 1.88.0 ABI / gcc 13.3 -> ACTS unaffected).
# Gaudi requires: filesystem regex fiber thread python unit_test_framework
# program_options log log_setup graph (+ transitive context/atomic/chrono/...).
# The sw image ships boost built minimal (filesystem only); this fills the gap.
# (-> sw PR: boost +fiber+python+log+... ; this whole script then disappears.)
set -e
source /opt/build-env.sh
BPREFIX=$(ls -d /spack/opt/spack/linux-x86_64/boost-1.88.0-*/ | head -1); BPREFIX=${BPREFIX%/}
PYBIN=$(ls /spack/opt/spack/linux-x86_64/python-venv-*/bin/python3 2>/dev/null | head -1)
[ -z "$PYBIN" ] && PYBIN=$(ls /spack/opt/spack/linux-x86_64/python-*/bin/python3 2>/dev/null | head -1)
PYVER=$($PYBIN -c 'import sys;print("%d.%d"%sys.version_info[:2])')
PYINC=$($PYBIN -c 'import sysconfig;print(sysconfig.get_path("include"))')
PYLIBDIR=$($PYBIN -c 'import sysconfig;print(sysconfig.get_config_var("LIBDIR"))')
echo "boost=$BPREFIX python=$PYVER inc=$PYINC libdir=$PYLIBDIR"

cd /tmp
wget -q https://archives.boost.io/release/1.88.0/source/boost_1_88_0.tar.gz -O boost.tgz
tar xf boost.tgz
cd boost_1_88_0
printf 'using python : %s : %s : %s : %s ;\n' "$PYVER" "$PYBIN" "$PYINC" "$PYLIBDIR" > /tmp/user-config.jam

LIBS=filesystem,regex,fiber,context,thread,python,test,program_options,log,graph,atomic,chrono,date_time,system
./bootstrap.sh --with-libraries=$LIBS --prefix="$BPREFIX" >/tmp/boost-bootstrap.log 2>&1
./b2 -j6 --user-config=/tmp/user-config.jam --prefix="$BPREFIX" \
     cxxstd=20 variant=release link=shared,static threading=multi install >/tmp/boost-b2.log 2>&1

test -f "$BPREFIX/lib/cmake/boost_python-1.88.0/boost_python-config.cmake"
test -f "$BPREFIX/lib/cmake/boost_fiber-1.88.0/boost_fiber-config.cmake"
rm -rf /tmp/boost_1_88_0 /tmp/boost.tgz
echo "Boost components built into $BPREFIX"
