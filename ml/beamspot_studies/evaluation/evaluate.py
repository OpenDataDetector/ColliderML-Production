"""
Evaluate a trained track regression model on a dataset.

Loads a checkpoint, runs inference, computes resolution metrics,
generates standard HEP performance plots, and optionally logs to W&B.

Usage:
    python evaluate.py --checkpoint /path/to/best.ckpt \
        --parquet-base /path/to/parquet --output-dir /path/to/plots \
        --wandb-run-id <id> --wandb-project colliderml-beamspot
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.datasets import TrackHitDataset, model_to_raw_params, PARAM_NAMES_RAW, N_OUTPUT
from models.track_transformer import TrackTransformer
from training.train import TrackRegressionModule
from evaluation.plotting import make_all_residual_plots


def load_model(checkpoint_path, device="cpu"):
    """Load trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    state = ckpt.get("state_dict", {})
    module = TrackRegressionModule(hparams_dict=hparams)
    module.load_state_dict(state, strict=False)
    module.eval()
    module.to(device)
    return module


@torch.no_grad()
def run_inference(module, dataset, batch_size=512, device="cpu"):
    """Run inference, return denormalized (pred, truth, reco) in raw 5-param format."""
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

        pred_denorm = pred_norm * scales
        truth_denorm = truth_norm * scales

        # Vectorized sin/cos → phi conversion
        def batch_to_raw(m):
            phi = np.arctan2(m[:, 2], m[:, 3])
            return np.stack([m[:, 0], m[:, 1], phi, m[:, 4], m[:, 5]], axis=1)

        all_pred_raw.append(batch_to_raw(pred_denorm))
        all_truth_raw.append(batch_to_raw(truth_denorm))
        all_reco_raw.append(batch_to_raw(reco_model))

    return np.concatenate(all_pred_raw), np.concatenate(all_truth_raw), np.concatenate(all_reco_raw)


def compute_metrics(pred_raw, truth_raw, reco_raw):
    """Compute per-parameter resolution and bias."""
    from scipy.stats import iqr
    metrics = {}
    for i, name in enumerate(PARAM_NAMES_RAW):
        ml_res = pred_raw[:, i] - truth_raw[:, i]
        kf_res = reco_raw[:, i] - truth_raw[:, i]
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


def log_to_wandb(figs, metrics, args):
    """Resume an existing W&B run and log evaluation plots + metrics."""
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        id=args.wandb_run_id,
        resume="must",
    )

    # Log each figure as an image
    for name, fig in figs.items():
        wandb.log({f"eval/{name}": wandb.Image(fig)})

    # Log metrics as a summary table
    columns = ["param", "ml_res", "kf_res", "improvement", "ml_bias"]
    rows = []
    for pname, m in metrics.items():
        rows.append([
            pname,
            m["ml_resolution_robust"],
            m["kf_resolution_robust"],
            m["improvement"],
            m["ml_bias"],
        ])
    wandb.log({"eval/metrics_table": wandb.Table(columns=columns, data=rows)})

    # Also log scalar metrics for easy comparison
    for pname, m in metrics.items():
        wandb.log({
            f"eval/{pname}_resolution_ml": m["ml_resolution_robust"],
            f"eval/{pname}_resolution_kf": m["kf_resolution_robust"],
            f"eval/{pname}_improvement": m["improvement"],
        })

    wandb.finish()
    print(f"Logged evaluation to W&B run {args.wandb_run_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--parquet-base", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--dataset-name", type=str, default="eval")
    parser.add_argument("--wandb-run-id", type=str, default=None,
                        help="W&B run ID to resume and log plots to")
    parser.add_argument("--wandb-project", type=str, default="colliderml-beamspot")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    module = load_model(args.checkpoint, device)

    print(f"Loading data from {args.parquet_base}")
    dataset = TrackHitDataset(args.parquet_base, max_files=args.max_files)
    print(f"Dataset: {len(dataset)} tracks")

    print("Running inference...")
    pred_raw, truth_raw, reco_raw = run_inference(
        module, dataset, batch_size=args.batch_size, device=device,
    )

    metrics = compute_metrics(pred_raw, truth_raw, reco_raw)
    print("\n=== Resolution Summary ===")
    for name, m in metrics.items():
        print(f"  {name:8s}: ML={m['ml_resolution_robust']:.4g}  "
              f"CKF={m['kf_resolution_robust']:.4g}  "
              f"ratio={m['improvement']:.2f}x")

    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"dataset": args.dataset_name, "n_tracks": len(dataset),
                    "params": metrics}, f, indent=2)

    print("\nGenerating plots...")
    figs = make_all_residual_plots(pred_raw, reco_raw, truth_raw, output_dir=str(output_dir))
    plt.close("all")

    if args.wandb_run_id:
        print(f"\nLogging to W&B run {args.wandb_run_id}...")
        log_to_wandb(figs, metrics, args)

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
