"""
XGBoost BDT baseline for track parameter regression.

Extracts a fixed-size feature vector per track from the first 3 hits,
last hit, hit count, deltas, and sagitta. Trains one XGBoost regressor
per output parameter.

Usage:
    python bdt_baseline.py --parquet-base /path/to/parquet --output-dir /path/to/output
    python bdt_baseline.py --parquet-base /path/to/parquet --output-dir /path/to/output --eval-only
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import xgboost as xgb

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.datasets import TrackHitDataset, model_to_raw_params, PARAM_NAMES_RAW

FEATURE_NAMES = [
    "r0", "phi0", "z0_hit",
    "r1", "phi1", "z1_hit",
    "r2", "phi2", "z2_hit",
    "r_last", "phi_last", "z_last",
    "n_hits",
    "delta_r", "delta_phi", "delta_z",
    "sagitta",
]


def extract_bdt_features(dataset):
    """Extract fixed-size feature vectors from a TrackHitDataset (vectorized).

    Returns (features, truth_raw, reco_raw) where:
        features: (N, 17) numpy array
        truth_raw: (N, 5) [d0, z0, phi, theta, qop]
        reco_raw: (N, 5)
    """
    N = len(dataset)
    input_std = dataset._input_std
    output_scales = dataset._output_scales

    # Denormalize all hit features at once: (N, max_hits, 10)
    raw_hits = dataset.hit_features.numpy() * input_std  # broadcast
    n_hits = dataset.n_hits.numpy()  # (N,)
    mask = dataset.padding_mask.numpy()  # (N, max_hits)

    # columns: [r, phi, z, vol, lay, det, dr, dphi, dr_dphi, dz_dr]
    r_all = raw_hits[:, :, 0]    # (N, max_hits)
    phi_all = raw_hits[:, :, 1]
    z_all = raw_hits[:, :, 2]

    features = np.zeros((N, len(FEATURE_NAMES)), dtype=np.float32)

    # First 3 hits: use index 0,1,2 but clamp to n_hits-1
    for j in range(3):
        idx = np.minimum(j, n_hits - 1).clip(0)
        features[:, j*3]     = r_all[np.arange(N), idx]
        features[:, j*3 + 1] = phi_all[np.arange(N), idx]
        features[:, j*3 + 2] = z_all[np.arange(N), idx]

    # Last hit
    last_idx = (n_hits - 1).clip(0)
    features[:, 9]  = r_all[np.arange(N), last_idx]
    features[:, 10] = phi_all[np.arange(N), last_idx]
    features[:, 11] = z_all[np.arange(N), last_idx]

    # Hit count
    features[:, 12] = n_hits

    # Deltas first-to-last
    features[:, 13] = features[:, 9] - features[:, 0]   # delta_r
    dphi = features[:, 10] - features[:, 1]
    features[:, 14] = (dphi + np.pi) % (2 * np.pi) - np.pi  # delta_phi
    features[:, 15] = features[:, 11] - features[:, 2]  # delta_z

    # Sagitta: middle hit deviation from linear interpolation in phi
    mid_idx = (n_hits // 2).clip(0)
    r_mid = r_all[np.arange(N), mid_idx]
    phi_mid = phi_all[np.arange(N), mid_idx]
    r_first = features[:, 0]
    r_last = features[:, 9]
    frac = (r_mid - r_first) / (r_last - r_first + 1e-10)
    phi_expected = features[:, 1] + frac * features[:, 14]
    sagitta = phi_mid - phi_expected
    features[:, 16] = (sagitta + np.pi) % (2 * np.pi) - np.pi
    features[n_hits < 3, 16] = 0  # no sagitta for < 3 hits

    # Truth: denormalize and convert to raw 5-param [d0, z0, phi, theta, qop]
    truth_m = dataset.truth_params.numpy() * output_scales  # (N, 6)
    truth_phi = np.arctan2(truth_m[:, 2], truth_m[:, 3])
    truth_raw = np.stack([truth_m[:, 0], truth_m[:, 1], truth_phi, truth_m[:, 4], truth_m[:, 5]], axis=1)

    # Reco: already raw model format
    reco_m = dataset.reco_params.numpy()  # (N, 6)
    reco_phi = np.arctan2(reco_m[:, 2], reco_m[:, 3])
    reco_raw = np.stack([reco_m[:, 0], reco_m[:, 1], reco_phi, reco_m[:, 4], reco_m[:, 5]], axis=1)

    return features, truth_raw, reco_raw


def train_bdt(features, truth_raw, val_split=0.1, seed=42):
    """Train one XGBoost regressor per parameter. Returns dict of models."""
    N = len(features)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(N)
    n_val = int(N * val_split)
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]

    X_train, X_val = features[train_idx], features[val_idx]

    models = {}
    for i, name in enumerate(PARAM_NAMES_RAW):
        y_train = truth_raw[train_idx, i]
        y_val = truth_raw[val_idx, i]

        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            early_stopping_rounds=10,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        models[name] = model
        print(f"  {name}: best_iteration={model.best_iteration}, "
              f"best_score={model.best_score:.6f}")

    return models, train_idx, val_idx


def predict_bdt(models, features):
    """Run prediction with all BDT models. Returns (N, 5) predictions."""
    pred = np.zeros((len(features), 5), dtype=np.float32)
    for i, name in enumerate(PARAM_NAMES_RAW):
        pred[:, i] = models[name].predict(features)
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-base", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-files", type=int, default=16)
    parser.add_argument("--eval-parquet-base", type=str, default=None,
                        help="Separate dataset for evaluation")
    parser.add_argument("--eval-max-files", type=int, default=5)
    parser.add_argument("--wandb-project", type=str, default="colliderml-beamspot")
    parser.add_argument("--wandb-name", type=str, default="bdt-baseline")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    print(f"Loading training data from {args.parquet_base}...")
    t0 = time.time()
    train_ds = TrackHitDataset(args.parquet_base, max_files=args.max_files)
    print(f"  {len(train_ds)} tracks in {time.time()-t0:.1f}s")

    print("Extracting BDT features...")
    t0 = time.time()
    features, truth_raw, reco_raw = extract_bdt_features(train_ds)
    print(f"  Features shape: {features.shape} in {time.time()-t0:.1f}s")

    # Train
    print("\nTraining BDT models...")
    models, train_idx, val_idx = train_bdt(features, truth_raw)

    # Save models
    for name, model in models.items():
        model.save_model(str(output_dir / f"bdt_{name}.json"))
    print(f"\nModels saved to {output_dir}")

    # Evaluate on validation split
    print("\n=== Validation Set Results ===")
    val_pred = predict_bdt(models, features[val_idx])
    val_truth = truth_raw[val_idx]
    val_reco = reco_raw[val_idx]

    from scipy.stats import iqr
    metrics = {}
    for i, name in enumerate(PARAM_NAMES_RAW):
        bdt_res = val_pred[:, i] - val_truth[:, i]
        kf_res = val_reco[:, i] - val_truth[:, i]
        bdt_sigma = iqr(bdt_res) / 1.349
        kf_sigma = iqr(kf_res) / 1.349
        ratio = kf_sigma / bdt_sigma if bdt_sigma > 0 else float("inf")
        print(f"  {name:8s}: BDT={bdt_sigma:.4g}  CKF={kf_sigma:.4g}  ratio={ratio:.2f}x")
        metrics[name] = {
            "bdt_resolution_robust": float(bdt_sigma),
            "kf_resolution_robust": float(kf_sigma),
            "improvement": float(ratio),
        }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Generate plots
    print("\nGenerating plots...")
    from evaluation.plotting import make_all_residual_plots
    import matplotlib
    matplotlib.use("Agg")
    figs = make_all_residual_plots(val_pred, val_reco, val_truth, output_dir=str(output_dir))

    # Log to W&B
    import wandb
    run = wandb.init(project=args.wandb_project, name=args.wandb_name)
    for name, fig in figs.items():
        wandb.log({f"bdt/{name}": wandb.Image(fig)})
    columns = ["param", "bdt_res", "kf_res", "improvement"]
    rows = [[n, m["bdt_resolution_robust"], m["kf_resolution_robust"], m["improvement"]]
            for n, m in metrics.items()]
    wandb.log({"bdt/metrics_table": wandb.Table(columns=columns, data=rows)})
    wandb.finish()

    print(f"\nDone. Results in {output_dir}")


if __name__ == "__main__":
    main()
