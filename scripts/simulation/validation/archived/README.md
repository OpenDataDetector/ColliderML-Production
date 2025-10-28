# Archived Validation Scripts

**Date Archived:** 2025-10-13

**Reason:** Replaced by unified validation system in Sprint S0.2

## Scripts Archived

1. `validate_generation.py` - Basic file existence checks for generation stage
2. `validate_pythia_generation.py` - File size-based validation (successful pattern!)
3. `validate_simulation.py` - Basic validation for simulation outputs
4. `validate_digitization.py` - Check for tracking/digitization outputs
5. `validate_merge_smear.py` - Pileup merging validation
6. `validate_convert_all.py` - HDF5 conversion validation

## Why Replaced?

These scripts had several issues:
- Inconsistent interfaces (different command-line arguments)
- Varying levels of validation depth
- No error recovery logic
- No integration with pipeline orchestration
- Incomplete coverage (missing checks)

## Replacement System

See: **Sprint S0.2 - Validation + Error Guardian System**

Location: `/pscratch/sd/d/danieltm/ColliderML/sprints/S0.2_ValidationGuardian/`

The new system provides:
- Unified validation library (`validation_lib.py`)
- Consistent file size-based validation across all stages
- Error guardian with automatic failure recovery
- Integration with SLURM requeuing
- Configurable thresholds via YAML
- Structured JSON output for programmatic use

## Useful Reference

`validate_pythia_generation.py` was the most complete validation script and served as the model for the new validation library. It successfully implemented:
- File size collection across runs
- Median calculation
- Threshold-based outlier detection (80% of median)
- Structured reporting

This pattern has been generalized and extended to all pipeline stages.

## If You Need to Use These

These scripts are preserved for reference but are **deprecated**. They may still work for ad-hoc validation but are not maintained.

For current validation, use:
```bash
python /path/to/validation_lib.py --stage <stage_name> --runs-dir <runs_directory>
```

See Sprint S0.2 documentation for details.

