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

echo "=== Resuming v10 nominal at $(date) ==="
python ${REPO}/ml/beamspot_studies/training/train.py \
  --parquet-base ${SIM_BASE}/hard_scatter/ttbar/v1/parquet \
  --output-dir ${OUT_BASE}/baseline_nominal_v10 \
  --wandb-name v10-nominal \
  --d-model 256 --n-heads 8 --n-layers 8 --d-ff 1024 --cls-input-dim 8 \
  --loss truncated_huber --batch-size 256 --lr 5e-4 --epochs 50 --patience 10 \
  --numeric-sort --max-files 50 --resume
echo "=== Done at $(date) ==="
