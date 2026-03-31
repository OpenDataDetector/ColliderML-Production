#!/bin/bash
# =============================================================
# ColliderML Container End-to-End Pipeline Test
#
# Runs a minimal LO pp->Z pipeline through all stages to validate
# the container environment. Uses sm model (no heft needed).
#
# Usage on lxplus:
#   apptainer exec --writable-tmpfs \
#     --bind /eos:/eos --bind /afs:/afs --bind /tmp:/tmp \
#     --bind /run/user --env KRB5CCNAME=$KRB5CCNAME \
#     colliderml-production_latest.sif \
#     bash container/test_pipeline.sh
#
# Or after entering the container:
#   bash container/test_pipeline.sh
# =============================================================

set -euo pipefail

# Source container env if not already sourced
if [ -z "${ACTS_DIR:-}" ]; then
    source /opt/colliderml/setup_env.sh
fi

WORK_DIR=${1:-/tmp/colliderml_e2e_test_$$}
mkdir -p "$WORK_DIR"
echo "=== ColliderML E2E Test ==="
echo "Working directory: $WORK_DIR"
echo ""

PASS=0
FAIL=0
report() {
    local name=$1 status=$2
    if [ "$status" -eq 0 ]; then
        echo "[PASS] $name"
        PASS=$((PASS + 1))
    else
        echo "[FAIL] $name"
        FAIL=$((FAIL + 1))
    fi
}

# ----- Test 1: Python imports -----
echo "--- Test 1: Python imports ---"
python3 -c "
import acts
import acts.examples.pythia8
import acts.examples.hepmc3
from DDSim.DD4hepSimulation import DD4hepSimulation
import ROOT
import HepMC3
import uproot
import awkward
import pyarrow
import pandas
import pyedm4hep
import pyhepmc
print('All imports OK')
print(f'  ACTS {acts.__version__}')
print(f'  ROOT {ROOT.gROOT.GetVersion()}')
" 2>&1
report "Python imports" $?

# ----- Test 2: ROOT/HepMC3 dictionary (THE critical fix) -----
echo ""
echo "--- Test 2: ROOT/HepMC3 dictionary loading ---"
python3 -c "
import ROOT
# This is what crashes without ROOT_INCLUDE_PATH fix
ROOT.gSystem.Load('libHepMC3rootIO')
print('HepMC3 ROOT I/O dictionary loaded successfully')
# Try to instantiate a HepMC3 class through ROOT to verify full dict chain
ROOT.gInterpreter.Declare('#include \"HepMC3/GenEvent.h\"')
print('HepMC3 header resolution OK')
" 2>&1
report "ROOT/HepMC3 dictionary" $?

# ----- Test 3: MadGraph Init (LO pp->Z, sm model) -----
echo ""
echo "--- Test 3: MadGraph Init ---"
MG_TEST_DIR="$WORK_DIR/mg_test"
mkdir -p "$MG_TEST_DIR"
cat > "$MG_TEST_DIR/proc_card.dat" << 'EOF'
import model sm
generate p p > z
output mg_test_ppz
EOF
cd "$MG_TEST_DIR"
$MG5_DIR/bin/mg5_aMC proc_card.dat --no_gui 2>&1 | tail -5
report "MadGraph Init" $?

# ----- Test 4: MadGraph Generation -----
echo ""
echo "--- Test 4: MadGraph Generation (10 events) ---"
cd "$MG_TEST_DIR/mg_test_ppz"
cat > run.sh << 'RUNEOF'
launch
set nevents 10
set iseed 42
RUNEOF
echo "exit" >> run.sh
$MG5_DIR/bin/mg5_aMC run.sh --no_gui 2>&1 | tail -5
# Check that events were generated
HEPMC_FILE=$(find "$MG_TEST_DIR/mg_test_ppz" -name "*.hepmc*" -o -name "*.lhe*" | head -1)
if [ -n "$HEPMC_FILE" ]; then
    echo "  Output: $HEPMC_FILE"
    report "MadGraph Generation" 0
else
    echo "  No event file found"
    report "MadGraph Generation" 1
fi

# ----- Test 5: ACTS Pythia8 (PU generation) -----
echo ""
echo "--- Test 5: ACTS Pythia8 PU generation ---"
python3 -c "
import acts
import acts.examples
import acts.examples.pythia8
from pathlib import Path

# Minimal Pythia8 generation test
s = acts.examples.Sequencer(events=2, numThreads=1)
rnd = acts.examples.RandomNumbers(seed=42)

vtxGen = acts.examples.GaussianVertexGenerator(
    stddev=acts.Vector4(0, 0, 0, 0),
    mean=acts.Vector4(0, 0, 0, 0),
)

gen = acts.examples.pythia8.Pythia8Generator(
    level=acts.logging.WARNING,
    pdgBeam0=2212, pdgBeam1=2212,
    cmsEnergy=13000,
    settings=['SoftQCD:all = on'],
)

evGen = acts.examples.EventGenerator(
    level=acts.logging.WARNING,
    generators=[acts.examples.EventGenerator.Generator(
        multiplicity=acts.examples.FixedMultiplicityGenerator(n=1),
        vertex=vtxGen,
        particles=gen,
    )],
    outputParticles='particles',
    randomNumbers=rnd,
)
s.addReader(evGen)
s.run()
print(f'Generated {s.config.events} Pythia8 PU events OK')
" 2>&1
report "ACTS Pythia8 PU" $?

# ----- Test 6: DDSim smoke test -----
echo ""
echo "--- Test 6: DDSim import + ODD geometry ---"
python3 -c "
from DDSim.DD4hepSimulation import DD4hepSimulation
import os
odd_path = os.environ.get('ODD_PATH', '')
assert odd_path, 'ODD_PATH not set'
odd_xml = os.path.join(odd_path, 'xml', 'OpenDataDetector.xml')
assert os.path.exists(odd_xml), f'ODD XML not found: {odd_xml}'
print(f'DDSim OK, ODD geometry at {odd_xml}')
" 2>&1
report "DDSim + ODD geometry" $?

# ----- Summary -----
echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
if [ "$FAIL" -gt 0 ]; then
    echo "Container validation FAILED — fix issues above before running production."
    exit 1
else
    echo "Container validation PASSED — ready for end-to-end production runs."
    exit 0
fi
