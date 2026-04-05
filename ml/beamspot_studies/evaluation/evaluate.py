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
    """Run inference, return denormalized (pred, truth, reco) in raw 5-param format.

    Output scales are fixed constants (OUTPUT_SCALES in datasets.py),
    identical across all datasets.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    scales = module.output_scales.cpu().numpy()

    all_pred_raw, all_truth_raw, all_reco_raw = [], [], []

    for batch in loader:
        cls_feats = batch.get("cls_features", None)
        if cls_feats is not None:
            cls_feats = cls_feats.to(device)
        pred_norm = module(
            batch["hit_features"].to(device),
            batch["padding_mask"].to(device),
            cls_feats,
        ).cpu().numpy()

        truth_norm = batch["truth_params"].numpy()
        reco_model = batch["reco_params"].numpy()

        pred_denorm = pred_norm * scales
        truth_denorm = truth_norm * scales

        # Vectorized sin/cos → phi conversion
        def batch_to_raw(m):
            phi = np.arctan2(m[:, 2], m[:, 3])
            theta = np.arctan2(1.0, m[:, 4])  # theta = atan2(1, cot_theta)
            return np.stack([m[:, 0], m[:, 1], phi, theta, m[:, 5]], axis=1)

        all_pred_raw.append(batch_to_raw(pred_denorm))
        all_truth_raw.append(batch_to_raw(truth_denorm))
        all_reco_raw.append(batch_to_raw(reco_model))

    return np.concatenate(all_pred_raw), np.concatenate(all_truth_raw), np.concatenate(all_reco_raw)


def make_zero_baseline(truth_raw):
    """Create 'predict zero' baseline: d0=0, z0=0, phi/theta/qop = truth mean.

    This measures how well you'd do by just knowing the beam spot position
    but nothing about individual track parameters.
    """
    zero_pred = np.zeros_like(truth_raw)
    zero_pred[:, 0] = 0.0  # d0 = 0 (beam spot)
    zero_pred[:, 1] = 0.0  # z0 = 0 (beam spot)
    zero_pred[:, 2] = np.mean(truth_raw[:, 2])  # phi = mean (best constant)
    zero_pred[:, 3] = np.mean(truth_raw[:, 3])  # theta = mean
    zero_pred[:, 4] = 0.0  # qop = 0 (symmetric charge)
    return zero_pred


def compute_metrics(pred_raw, truth_raw, reco_raw, param_names=None,
                    pred_label="ml", ref_label="kf"):
    """Compute per-parameter metrics: std, IQR/1.349, MAE, bias.

    Works for any technique — just change pred_label.
    """
    from scipy.stats import iqr
    if param_names is None:
        param_names = PARAM_NAMES_RAW
    metrics = {}
    for i, name in enumerate(param_names):
        pred_res = pred_raw[:, i] - truth_raw[:, i]
        ref_res = reco_raw[:, i] - truth_raw[:, i]

        metrics[name] = {
            f"{pred_label}_std": float(np.std(pred_res)),
            f"{pred_label}_iqr": float(iqr(pred_res) / 1.349),
            f"{pred_label}_mae": float(np.mean(np.abs(pred_res))),
            f"{pred_label}_bias": float(np.mean(pred_res)),
            f"{ref_label}_std": float(np.std(ref_res)),
            f"{ref_label}_iqr": float(iqr(ref_res) / 1.349),
            f"{ref_label}_mae": float(np.mean(np.abs(ref_res))),
            f"{ref_label}_bias": float(np.mean(ref_res)),
        }
    return metrics


def print_metrics(metrics, labels=None):
    """Print a clean table of all metrics for multiple techniques.

    Args:
        metrics: dict of {param: {label_std, label_iqr, ...}} from compute_metrics
        labels: list of technique labels to show (default: ["ml", "kf"])
    """
    if labels is None:
        labels = ["ml", "kf"]
    params = list(metrics.keys())
    # Header
    header = f"{'Param':>8s}"
    subheader = f"{'':>8s}"
    for lbl in labels:
        header += f" {'STD':>10s} {'IQR':>10s} {'MAE':>10s}"
        subheader += f" {'--- '+lbl+' ---':>31s}"
    print(header)
    print(subheader)
    print("-" * (8 + 31 * len(labels)))
    for name in params:
        m = metrics[name]
        row = f"{name:>8s}"
        for lbl in labels:
            row += f" {m.get(f'{lbl}_std', float('nan')):10.4g} {m.get(f'{lbl}_iqr', float('nan')):10.4g} {m.get(f'{lbl}_mae', float('nan')):10.4g}"
        print(row)


def log_to_wandb(figs, metrics, args, pred_label="ml", ref_label="kf"):
    """Resume an existing W&B run and log evaluation plots + metrics."""
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        id=args.wandb_run_id,
        resume="must",
    )

    for name, fig in figs.items():
        wandb.log({f"eval/{name}": wandb.Image(fig)})

    columns = ["param", f"{pred_label}_std", f"{pred_label}_iqr", f"{pred_label}_mae",
               f"{pred_label}_bias", f"{ref_label}_std", f"{ref_label}_iqr", f"{ref_label}_mae"]
    rows = []
    for pname, m in metrics.items():
        rows.append([pname] + [m[c] for c in columns[1:]])
    wandb.log({"eval/metrics_table": wandb.Table(columns=columns, data=rows)})

    for pname, m in metrics.items():
        for metric in ["std", "iqr", "mae"]:
            wandb.log({f"eval/{pname}_{metric}_{pred_label}": m[f"{pred_label}_{metric}"]})
            wandb.log({f"eval/{pname}_{metric}_{ref_label}": m[f"{ref_label}_{metric}"]})

    wandb.finish()
    print(f"Logged evaluation to W&B run {args.wandb_run_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--parquet-base", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--skip-event-range", type=int, nargs=2, default=None,
                        metavar=("START", "END"),
                        help="Skip files overlapping this event range (avoid train/eval overlap)")
    parser.add_argument("--numeric-sort", action="store_true",
                        help="Sort files by numeric event index (not lexicographic)")
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
    skip_range = tuple(args.skip_event_range) if args.skip_event_range else None
    dataset = TrackHitDataset(args.parquet_base, max_files=args.max_files,
                              skip_event_range=skip_range,
                              numeric_sort=args.numeric_sort)
    print(f"Dataset: {len(dataset)} tracks")

    print("Running inference...")
    pred_raw, truth_raw, reco_raw = run_inference(
        module, dataset, batch_size=args.batch_size, device=device,
    )

    # Zero baseline: predict d0=0, z0=0 (beam spot only, no track fitting)
    zero_pred = make_zero_baseline(truth_raw)
    zero_metrics = compute_metrics(zero_pred, truth_raw, reco_raw,
                                   pred_label="zero", ref_label="kf")

    # Merge zero baseline metrics into main metrics
    metrics = compute_metrics(pred_raw, truth_raw, reco_raw)
    for name in metrics:
        for key in zero_metrics[name]:
            if key.startswith("zero_"):
                metrics[name][key] = zero_metrics[name][key]

    print("\n=== Resolution Summary ===")
    print_metrics(metrics, labels=["ml", "kf", "zero"])

    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"dataset": args.dataset_name, "n_tracks": len(dataset),
                    "params": metrics}, f, indent=2)

    print("\nGenerating plots...")
    figs = make_all_residual_plots(pred_raw, reco_raw, truth_raw,
                                   zero_pred=zero_pred,
                                   output_dir=str(output_dir))
    plt.close("all")

    if args.wandb_run_id:
        print(f"\nLogging to W&B run {args.wandb_run_id}...")
        log_to_wandb(figs, metrics, args)

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
