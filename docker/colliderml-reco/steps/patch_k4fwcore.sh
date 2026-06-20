#!/bin/bash
# Patch the base image's k4FWCore 1.3 ApplicationMgr.py for Gaudi 40 compatibility.
#
# k4FWCore 1.3 (this key4hep snapshot) pairs with Gaudi 40.0, but its
# ApplicationMgr.fix_properties() does `self._mgr.EventLoop = EventLoopMgr(Warnings=False)`
# in an except branch — purely to suppress two "no external input" warnings. In Gaudi 40
# the EventLoopMgr configurable no longer has a `Warnings` property, so this raises
# AttributeError and EVERY k4run (bare or ours) dies before running any algorithm. The
# block is non-functional (warning suppression only), so guard it with try/except.
set -e
source /opt/key4hep-shim.sh
AM=$(python3 -c "import k4FWCore, os; print(os.path.join(os.path.dirname(k4FWCore.__file__), 'ApplicationMgr.py'))")
python3 - "$AM" <<'PY'
import sys
f = sys.argv[1]
s = open(f).read()
old = "            self._mgr.EventLoop = EventLoopMgr(Warnings=False)"
new = ("            try:\n"
       "                self._mgr.EventLoop = EventLoopMgr(Warnings=False)\n"
       "            except Exception:\n"
       "                pass  # Gaudi 40: EventLoopMgr.Warnings removed (warning-suppression only)")
if new.strip() in s:
    print("ApplicationMgr.py already patched"); sys.exit(0)
assert old in s, f"anchor not found in {f}"
open(f, "w").write(s.replace(old, new))
print(f"patched {f} (Gaudi 40 EventLoopMgr.Warnings guard)")
PY
