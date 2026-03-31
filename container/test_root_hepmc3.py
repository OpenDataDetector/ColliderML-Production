#!/usr/bin/env python3
"""Test ROOT/HepMC3 dictionary integration inside the ODD container."""
import sys

try:
    import ROOT
    print(f"ROOT {ROOT.gROOT.GetVersion()} loaded")
except ImportError:
    print("[FAIL] Cannot import ROOT")
    sys.exit(1)

# Test 1: Load HepMC3 ROOT I/O library
res = ROOT.gSystem.Load("libHepMC3rootIO")
print(f"gSystem.Load('libHepMC3rootIO') returned {res}")
if res < 0:
    print("[FAIL] Could not load libHepMC3rootIO")
    sys.exit(1)
print("[PASS] HepMC3 ROOT I/O library loaded")

# Test 2: Resolve GenRunInfoData.h (this is what crashes without the fix)
ok = ROOT.gInterpreter.Declare('#include "HepMC3/Data/GenRunInfoData.h"')
if ok:
    print("[PASS] GenRunInfoData.h resolved by ROOT")
else:
    print("[FAIL] GenRunInfoData.h NOT resolved")
    sys.exit(1)

# Test 3: Full GenEvent header
ok2 = ROOT.gInterpreter.Declare('#include "HepMC3/GenEvent.h"')
if ok2:
    print("[PASS] GenEvent.h resolved by ROOT")
else:
    print("[FAIL] GenEvent.h NOT resolved")
    sys.exit(1)

print("=== ROOT/HepMC3 dictionary integration WORKS ===")
