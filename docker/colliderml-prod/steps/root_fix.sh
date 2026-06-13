# Sourceable: fix ROOT (buildcache) C++ module lookup.
# ROOT 6.38 is installed from a spack buildcache; cling's modules-cache-path is the
# original build-stage dir (/tmp/root/spack-stage/.../spack-build-<short>/lib), which
# does not exist in the image. Gaudi genconf and k4run start a full ApplicationMgr
# whose `run`/`genconf` wrapper does NOT set ROOTSYS, so cling falls back to that
# stale path and dies ("module file 'libc.pcm' not found").
# Two-pronged fix: (1) source thisroot.sh so ROOTSYS + the real module path are set;
# (2) symlink the stale build-stage path to ROOT's real lib dir as a backstop.
_R=$(ls -d /spack/opt/spack/linux-x86_64/root-6.38.00-* 2>/dev/null | head -1)
if [ -n "$_R" ]; then
  source "$_R/bin/thisroot.sh" 2>/dev/null || true
  _H=$(basename "$_R"); _H=${_H##*-}; _S=${_H:0:7}
  _STALE="/tmp/root/spack-stage/spack-stage-root-6.38.00-${_H}/spack-build-${_S}"
  mkdir -p "$_STALE" 2>/dev/null && ln -sfn "$_R/lib/root" "$_STALE/lib" 2>/dev/null
  unset _H _S _STALE
fi
unset _R
