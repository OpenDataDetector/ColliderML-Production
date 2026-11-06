#!/bin/bash
# Release 1 Geant4 (ddsim_run.py) or tracking (digi_and_reco.py) for runs FIRST..FIRST+ntasks-1,
# one srun task per run, in the Release 1 environment, exactly as the Release 1 job scripts did
# (hard_scatter/jets/v1/logs/stage_{simulation,digitization}/job_multinode.sh) but with the code
# from the c8d5550 snapshot (release1_code/c8d5550) instead of the moving colliderml_dev checkout.
#
#   sbatch -A m5386 -C cpu -q regular -N <nodes> --ntasks <runs> --cpus-per-task <8|16> -t 03:00:00 \
#     --image registry.cern.ch/atlasadc/atlas-grid-almalinux9 --module=cvmfs \
#     r1_stage.sbatch <ddsim_run|digi_and_reco> <config.yaml> <runs_dir> <dataset> <first>
SCRIPT=$1; CFG=$2; RUNS=$3; DS=$4; FIRST=$5
CODE=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software/release1_code/c8d5550/scripts/simulation
SW=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software
export SLURM_CPU_BIND="cores"
echo "=== $(date '+%F %T') $SCRIPT runs $FIRST..$((FIRST + SLURM_NTASKS - 1)) on $SLURM_JOB_NUM_NODES nodes (job $SLURM_JOB_ID)"
set +e
srun --exact --kill-on-bad-exit=0 -u shifter bash -c "
cd /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase && \
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase && \
source /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh && \
cd $SW && \
source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_107 x86_64-el9-gcc13-opt && \
source $SW/OtherLibraries/dd4hep_versions/dd4hep-colliderml-new/bin/thisdd4hep.sh && \
source $SW/OtherLibraries/acts_versions/acts_v4/build/this_acts.sh && \
set -o pipefail; N=\$(($FIRST + SLURM_PROCID)); \
python $CODE/$SCRIPT.py --config $CFG --output $RUNS --output-subdir \$N --seed ${DS}_v1_run\$N \
  | awk 'BEGIN{p=1} /G4Exception-START/{p=0} /G4Exception-END/{p=1;next} p' | grep -v -E '^G4WT[0-9]+ >\s*$' | grep -v -E '^[[:space:]]*$'; \
echo \"=== run \$N rc=\${PIPESTATUS[0]}\""
echo "=== $(date '+%F %T') done rc=$?"
