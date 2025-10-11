#!/bin/bash
# Test script for multi-config SLURM job submission
# This tests the new multi-config feature

echo "=========================================="
echo "Testing Multi-Config Job Submission"
echo "=========================================="

# Test 1: Single config (backward compatibility)
echo ""
echo "Test 1: Single config - should work as before"
echo "python run_stage.py ../../configs_production/full_pileup_pilot/ttbar/pythia_config.yaml --dry-run --execution-mode multi_node_slurm"
echo "(Not running - would require git commit)"

# Test 2: Multiple configs - should enforce multi_node_slurm
echo ""
echo "Test 2: Multiple configs with wrong execution mode - should fail"
echo "python run_stage.py config1.yaml config2.yaml --execution-mode interactive"
echo "(Expected: error message about requiring multi_node_slurm)"

# Test 3: Multiple configs with correct mode - should work
echo ""
echo "Test 3: Multiple configs with multi_node_slurm - should work"
echo "python run_stage.py config1.yaml config2.yaml --execution-mode multi_node_slurm --dry-run"
echo "(Expected: combined job script generated)"

# Test 4: Check imports work
echo ""
echo "Test 4: Checking if modules import correctly..."
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software/colliderml_dev/scripts/cli
python3 -c "
import sys
try:
    import multi_config_job
    print('✓ multi_config_job imports successfully')
    from multi_config_job import MultiConfigJobSubmitter, validate_multi_config_compatibility
    print('✓ MultiConfigJobSubmitter and helpers import successfully')
    import cli_utils
    print('✓ cli_utils imports successfully')
    if hasattr(cli_utils, 'load_and_process_config'):
        print('✓ load_and_process_config function exists')
    else:
        print('✗ load_and_process_config function NOT found')
        sys.exit(1)
    print('\nAll imports successful!')
except Exception as e:
    print(f'✗ Import error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

echo ""
echo "=========================================="
echo "Test Complete"
echo "=========================================="

