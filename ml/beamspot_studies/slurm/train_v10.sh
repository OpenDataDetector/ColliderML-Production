#!/bin/bash
set -eo pipefail
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PS1="${PS1:-}"
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate collider-env

REPO=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software/colliderml_dev
SIM_BASE=/global/cfs/cdirs/m4958/data/ColliderML/simulation
OUT_BASE=/global/cfs/cdirs/m4958/data/ColliderML/ml/beamspot_studies

COMMON="--d-model 256 --n-heads 8 --n-layers 8 --d-ff 1024 --cls-input-dim 8 \
  --loss truncated_huber --batch-size 256 --lr 5e-4 --epochs 50 --patience 10 \
  --numeric-sort --max-files 50"

DATASET=$1
PARQUET=$2
OUTDIR=$3
WANDB_NAME=$4

echo "=== Training v10 on ${DATASET} ==="
echo "=== Started at $(date) ==="
python ${REPO}/ml/beamspot_studies/training/train.py \
  --parquet-base ${PARQUET} \
  --output-dir ${OUTDIR} \
  --wandb-name ${WANDB_NAME} \
  ${COMMON}
echo "=== Done at $(date) ==="
