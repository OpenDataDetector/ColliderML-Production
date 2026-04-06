#!/bin/bash
# Phase 6: Cross-track attention training launcher.
#
# Usage:
#   bash train_v11_cross.sh <dataset_label> <parquet_base> <output_dir> <wandb_name>
#
# Example:
#   bash train_v11_cross.sh randomized_xy \
#     /global/cfs/cdirs/m4958/data/ColliderML/simulation/beamspot_studies/ttbar_randomized_xy/v1/parquet \
#     /global/cfs/cdirs/m4958/data/ColliderML/ml/beamspot_studies/baseline_randomized_xy_v11_cross \
#     v11-randomized-xy-cross
#
# Warm-starts the per-track encoder from the v10-randomized checkpoint. The
# cross-track block is trained from scratch.
set -eo pipefail
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PS1="${PS1:-}"
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate collider-env

REPO=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software/colliderml_dev
OUT_BASE=/global/cfs/cdirs/m4958/data/ColliderML/ml/beamspot_studies

DATASET=$1
PARQUET=$2
OUTDIR=$3
WANDB_NAME=$4

WARM_START_CKPT="${OUT_BASE}/baseline_randomized_xy_v10/checkpoints/best-epoch=049-val/loss=0.0046.ckpt"

echo "=== Training v11 cross-track on ${DATASET} ==="
echo "=== Warm-start: ${WARM_START_CKPT} ==="
echo "=== Started at $(date) ==="

python ${REPO}/ml/beamspot_studies/training/train.py \
  --parquet-base "${PARQUET}" \
  --output-dir "${OUTDIR}" \
  --wandb-project colliderml-beamspot-crosstrack \
  --wandb-name "${WANDB_NAME}" \
  --d-model 256 --n-heads 8 --n-layers 8 --d-ff 1024 --cls-input-dim 8 \
  --loss truncated_huber --lr 3e-4 --epochs 30 --patience 10 \
  --numeric-sort --max-files 50 \
  --cross-track --batch-size-events 8 --max-tracks-per-event 128 \
  --n-cross-layers 2 \
  --init-from-checkpoint "${WARM_START_CKPT}"

echo "=== Done at $(date) ==="
