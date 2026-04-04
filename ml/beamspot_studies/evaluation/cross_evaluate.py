"""
Cross-evaluation: run each trained model on all datasets.

Produces the 3x3 matrix of performance metrics and comparison plots.

Usage:
    python cross_evaluate.py --output-dir /path/to/results
"""

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.evaluate import load_model, run_inference, compute_metrics
from evaluation.plotting import plot_summary_table, PARAM_LABELS
from data.datasets import TrackHitDataset, PARAM_NAMES_RAW

import matplotlib.pyplot as plt


SIM_BASE = "/global/cfs/cdirs/m4958/data/ColliderML/simulation"
ML_BASE = "/global/cfs/cdirs/m4958/data/ColliderML/ml/beamspot_studies"

MODELS = {
    "nominal": f"{ML_BASE}/baseline_nominal/checkpoints",
    "shifted_300um": f"{ML_BASE}/baseline_shifted_300um/checkpoints",
    "shifted_25um": f"{ML_BASE}/baseline_shifted_25um/checkpoints",
}

DATASETS = {
    "nominal": f"{SIM_BASE}/hard_scatter/ttbar/v1/parquet",
    "shifted_300um": f"{SIM_BASE}/beamspot_studies/ttbar_shifted_300um/v1/parquet",
    "shifted_25um": f"{SIM_BASE}/beamspot_studies/ttbar_shifted_25um/v1/parquet",
}


def find_best_checkpoint(checkpoint_dir):
    """Find the best checkpoint in a directory."""
    ckpt_dir = Path(checkpoint_dir)
    best_files = list(ckpt_dir.glob("best-*.ckpt"))
    if best_files:
        return str(best_files[0])
    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return str(last)
    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")


def plot_cross_matrix(matrix, param_name, model_names, dataset_names,
                      figsize=(8, 6)):
    """Plot a heatmap of the cross-evaluation matrix for one parameter."""
    label, unit = PARAM_LABELS[param_name]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto")

    ax.set_xticks(range(len(dataset_names)))
    ax.set_xticklabels([f"Eval: {n}" for n in dataset_names], rotation=30, ha="right")
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels([f"Train: {n}" for n in model_names])

    for i in range(len(model_names)):
        for j in range(len(dataset_names)):
            ax.text(j, i, f"{matrix[i, j]:.4g}", ha="center", va="center",
                    fontsize=10, fontweight="bold" if i == j else "normal")

    plt.colorbar(im, ax=ax, label=f"{label} resolution [{unit}]")
    ax.set_title(f"{label} Resolution: Cross-Evaluation Matrix")
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-files", type=int, default=5,
                        help="Max files per dataset for evaluation")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model_names = list(MODELS.keys())
    dataset_names = list(DATASETS.keys())

    # Load all datasets
    print("Loading datasets...")
    datasets = {}
    for name, path in DATASETS.items():
        print(f"  {name}...")
        datasets[name] = TrackHitDataset(path, max_files=args.max_files)
        print(f"    {len(datasets[name])} tracks")

    # Cross-evaluate
    all_results = {}
    for mi, model_name in enumerate(model_names):
        ckpt = find_best_checkpoint(MODELS[model_name])
        print(f"\nLoading model: {model_name} ({ckpt})")
        module = load_model(ckpt, device)

        for di, ds_name in enumerate(dataset_names):
            key = f"{model_name}_on_{ds_name}"
            print(f"  Evaluating on {ds_name}...")

            pred_raw, truth_raw, reco_raw = run_inference(
                module, datasets[ds_name], batch_size=args.batch_size, device=device
            )
            metrics = compute_metrics(pred_raw, truth_raw, reco_raw)
            all_results[key] = metrics

            for pname, m in metrics.items():
                print(f"    {pname}: ML={m['ml_resolution_robust']:.4g} "
                      f"CKF={m['kf_resolution_robust']:.4g}")

    # Save all results
    with open(output_dir / "cross_eval_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Plot cross-evaluation matrices
    for param in PARAM_NAMES_RAW:
        matrix = np.zeros((len(model_names), len(dataset_names)))
        for mi, mn in enumerate(model_names):
            for di, dn in enumerate(dataset_names):
                key = f"{mn}_on_{dn}"
                matrix[mi, di] = all_results[key][param]["ml_resolution_robust"]

        fig = plot_cross_matrix(matrix, param, model_names, dataset_names)
        fig.savefig(output_dir / f"{param}_cross_matrix.pdf", bbox_inches="tight")
        plt.close(fig)

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
