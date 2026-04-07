#!/bin/bash
# Phase 6 ablation: 2x3 sweep of (loss x model) for the IQR money plot.
#
# Usage:
#   bash train_v12.sh <loss> <stage> [warm_start_ckpt]
#
# Where:
#   loss    = "th" (truncated_huber) | "h" (huber)
#   stage   = "scratch" | "cross" | "track"
#   warm_start_ckpt = required for stage in {cross, track}
#
# All runs train on ttbar_randomized_xy with --numeric-sort --max-files 50.
# Outputs go to baseline_randomized_xy_v12_<loss>_<stage>.
set -eo pipefail
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PS1="${PS1:-}"
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate collider-env

REPO=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software/colliderml_dev
SIM=/global/cfs/cdirs/m4958/data/ColliderML/simulation
OUT_BASE=/global/cfs/cdirs/m4958/data/ColliderML/ml/beamspot_studies
PARQUET=$SIM/beamspot_studies/ttbar_randomized_xy/v1/parquet

LOSS_KEY=$1
STAGE=$2
WARM_CKPT=${3:-}

case $LOSS_KEY in
  th) LOSS_FLAG="truncated_huber" ;;
  h)  LOSS_FLAG="huber" ;;
  *)  echo "Unknown loss: $LOSS_KEY (expected th or h)"; exit 1 ;;
esac

OUTDIR="$OUT_BASE/baseline_randomized_xy_v12_${LOSS_KEY}_${STAGE}"
WANDB_NAME="v12-${LOSS_KEY}-${STAGE}"

# Common architecture (matches v10/v11 — d_model=256, 8 layers)
ARCH_ARGS="--d-model 256 --n-heads 8 --n-layers 8 --d-ff 1024 --cls-input-dim 8"
COMMON_ARGS="--numeric-sort --max-files 50 --patience 10 --lr 5e-4"

case $STAGE in
  scratch)
    # 50 epochs from scratch, per-track architecture (= v10 baseline equivalent)
    EXTRA="--epochs 50 --batch-size 256"
    ;;
  cross)
    # 30 epochs warm-started from $WARM_CKPT, with cross-track attention
    if [ -z "$WARM_CKPT" ]; then echo "stage=cross requires warm_start_ckpt"; exit 1; fi
    EXTRA="--epochs 30 --batch-size-events 8 --max-tracks-per-event 128 \
      --cross-track --n-cross-layers 2 --init-from-checkpoint $WARM_CKPT"
    # Override LR for warm-start
    COMMON_ARGS="--numeric-sort --max-files 50 --patience 10 --lr 3e-4"
    ;;
  track)
    # 30 epochs warm-started from $WARM_CKPT, per-track only (ablation control).
    # batch_size=512 chosen so that gradient updates per epoch (~1080) match
    # the cross-track variant's (~1125 batches/epoch with batch_size_events=8).
    # This makes the ablation a clean apples-to-apples comparison.
    if [ -z "$WARM_CKPT" ]; then echo "stage=track requires warm_start_ckpt"; exit 1; fi
    EXTRA="--epochs 30 --batch-size 512 --init-from-checkpoint $WARM_CKPT"
    COMMON_ARGS="--numeric-sort --max-files 50 --patience 10 --lr 3e-4"
    ;;
  *)
    echo "Unknown stage: $STAGE"; exit 1 ;;
esac

echo "=== v12 ${LOSS_KEY} ${STAGE} ==="
echo "    OUTDIR: $OUTDIR"
echo "    WANDB:  $WANDB_NAME"
echo "    LOSS:   $LOSS_FLAG"
[ -n "$WARM_CKPT" ] && echo "    WARM:   $WARM_CKPT"
echo "=== Started at $(date) ==="

python ${REPO}/ml/beamspot_studies/training/train.py \
  --parquet-base "$PARQUET" \
  --output-dir "$OUTDIR" \
  --wandb-project colliderml-beamspot-crosstrack \
  --wandb-name "$WANDB_NAME" \
  $ARCH_ARGS \
  --loss $LOSS_FLAG \
  $COMMON_ARGS \
  $EXTRA

echo "=== Done at $(date) ==="
