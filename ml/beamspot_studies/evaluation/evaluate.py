"""
Evaluate a trained track regression model on a dataset.

Loads a checkpoint, runs inference, computes resolution metrics,
and generates standard HEP performance plots.

Usage:
    python evaluate.py --checkpoint /path/to/best.ckpt \
        --parquet-base /path/to/parquet --output-dir /path/to/plots
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.datasets import TrackHitDataset, model_to_raw_params, PARAM_NAMES_RAW, N_OUTPUT
from models.track_transformer import TrackTransformer
from training.train import TrackRegressionModule
from evaluation.plotting import make_all_residual_plots, plot_summary_table


def load_model(checkpoint_path, device="cpu"):
    """Load trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract hparams and output_scales
    hparams = ckpt.get("hyper_parameters", {})
    state = ckpt.get("state_dict", {})

    # Build module
    module = TrackRegressionModule(hparams_dict=hparams)
    module.load_state_dict(state, strict=False)
    module.eval()
    module.to(device)

    return module


@torch.no_grad()
def run_inference(module, dataset, batch_size=512, device="cpu"):
    """Run inference on a dataset, return denormalized predictions and truth."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    scales = module.output_scales.cpu().numpy()

    all_pred_raw, all_truth_raw, all_reco_raw = [], [], []

    for batch in loader:
        pred_norm = module(
            batch["hit_features"].to(device),
            batch["padding_mask"].to(device),
        ).cpu().numpy()

        truth_norm = batch["truth_params"].numpy()
        reco_model = batch["reco_params"].numpy()

        # Denormalize predictions and truth
        pred_denorm = pred_norm * scales
        truth_denorm = truth_norm * scales

        # Convert to raw 5-param format [d0, z0, phi, theta, qop]
        for i in range(len(pred_denorm)):
            all_pred_raw.append(model_to_raw_params(pred_denorm[i]))
            all_truth_raw.append(model_to_raw_params(truth_denorm[i]))
            all_reco_raw.append(model_to_raw_params(reco_model[i]))

    return np.stack(all_pred_raw), np.stack(all_truth_raw), np.stack(all_reco_raw)


def compute_metrics(pred_raw, truth_raw, reco_raw):
    """Compute per-parameter resolution and bias."""
    metrics = {}
    for i, name in enumerate(PARAM_NAMES_RAW):
        ml_res = pred_raw[:, i] - truth_raw[:, i]
        kf_res = reco_raw[:, i] - truth_raw[:, i]

        # Robust resolution (IQR-based)
        from scipy.stats import iqr
        ml_sigma = iqr(ml_res) / 1.349
        kf_sigma = iqr(kf_res) / 1.349

        metrics[name] = {
            "ml_resolution": float(np.std(ml_res)),
            "ml_resolution_robust": float(ml_sigma),
            "ml_bias": float(np.mean(ml_res)),
            "kf_resolution": float(np.std(kf_res)),
            "kf_resolution_robust": float(kf_sigma),
            "kf_bias": float(np.mean(kf_res)),
            "improvement": float(kf_sigma / ml_sigma) if ml_sigma > 0 else float("inf"),
        }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--parquet-base", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--dataset-name", type=str, default="eval",
                        help="Label for this evaluation (used in plot titles)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    module = load_model(args.checkpoint, device)

    # Load data
    print(f"Loading data from {args.parquet_base}")
    dataset = TrackHitDataset(args.parquet_base, max_files=args.max_files)
    print(f"Dataset: {len(dataset)} tracks")

    # Run inference
    print("Running inference...")
    pred_raw, truth_raw, reco_raw = run_inference(
        module, dataset, batch_size=args.batch_size, device=device
    )

    # Compute metrics
    metrics = compute_metrics(pred_raw, truth_raw, reco_raw)
    print("\n=== Resolution Summary ===")
    for name, m in metrics.items():
        print(f"  {name:8s}: ML={m['ml_resolution_robust']:.4g}  "
              f"CKF={m['kf_resolution_robust']:.4g}  "
              f"ratio={m['improvement']:.2f}x")

    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"dataset": args.dataset_name, "n_tracks": len(dataset),
                    "params": metrics}, f, indent=2)

    # Generate plots
    print("\nGenerating plots...")
    make_all_residual_plots(pred_raw, reco_raw, truth_raw, output_dir=str(output_dir))

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
