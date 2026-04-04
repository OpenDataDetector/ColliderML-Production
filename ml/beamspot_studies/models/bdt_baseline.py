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

# BDT uses a different parameterization than the transformer
BDT_PARAM_NAMES = ["d0", "z0", "phi", "cot_theta", "qop"]

def extract_bdt_features(dataset):
    """Extract rich feature vectors from a TrackHitDataset (vectorized).

    Features per track:
      - First 3 hits (r, phi, z): 9
      - Last hit (r, phi, z): 3
      - Hit count: 1
      - First-to-last deltas (dr, dphi, dz): 3
      - Sagitta (phi deviation at 3 points): 3
      - Aggregate stats (mean/std of r, phi, z): 6
      - Conformal coords of first 2 hits: 4
      - Circle fit curvature from first 3 hits: 1
      - dz/dr slope (linear fit proxy): 1
      - Hits per detector type (pixel vs strip): 2
      Total: ~33 features

    Returns (features, feature_names, truth_raw, reco_raw)
    """
    N = len(dataset)
    input_std = dataset._input_std
    output_scales = dataset._output_scales

    # Denormalize all hit features at once: (N, max_hits, 10)
    raw_hits = dataset.hit_features.numpy() * input_std
    n_hits = dataset.n_hits.numpy()
    mask = dataset.padding_mask.numpy()

    # columns: [r, phi, z, vol, lay, det, dr, dphi, dr_dphi, dz_dr]
    r_all = raw_hits[:, :, 0]
    phi_all = raw_hits[:, :, 1]
    z_all = raw_hits[:, :, 2]
    vol_all = raw_hits[:, :, 3]
    det_all = raw_hits[:, :, 5]

    arange_N = np.arange(N)
    last_idx = (n_hits - 1).clip(0)
    mid_idx = (n_hits // 2).clip(0)
    q1_idx = (n_hits // 4).clip(0)
    q3_idx = (3 * n_hits // 4).clip(0, last_idx)

    feats = {}

    # First 3 hits
    for j in range(3):
        idx = np.minimum(j, n_hits - 1).clip(0)
        feats[f"r{j}"] = r_all[arange_N, idx]
        feats[f"phi{j}"] = phi_all[arange_N, idx]
        feats[f"z{j}"] = z_all[arange_N, idx]

    # Last hit
    feats["r_last"] = r_all[arange_N, last_idx]
    feats["phi_last"] = phi_all[arange_N, last_idx]
    feats["z_last"] = z_all[arange_N, last_idx]

    # Hit count
    feats["n_hits"] = n_hits.astype(np.float32)

    # First-to-last deltas
    feats["delta_r"] = feats["r_last"] - feats["r0"]
    dphi_total = feats["phi_last"] - feats["phi0"]
    feats["delta_phi"] = (dphi_total + np.pi) % (2 * np.pi) - np.pi
    feats["delta_z"] = feats["z_last"] - feats["z0"]

    # Sagitta at 3 points (quarter, mid, three-quarter)
    for name, sidx in [("sagitta_mid", mid_idx), ("sagitta_q1", q1_idx), ("sagitta_q3", q3_idx)]:
        r_s = r_all[arange_N, sidx]
        phi_s = phi_all[arange_N, sidx]
        frac = (r_s - feats["r0"]) / (feats["delta_r"] + 1e-10)
        phi_exp = feats["phi0"] + frac * feats["delta_phi"]
        sag = phi_s - phi_exp
        feats[name] = (sag + np.pi) % (2 * np.pi) - np.pi
        feats[name] = np.where(n_hits >= 3, feats[name], 0.0)

    # Aggregate stats (mean/std of r, phi, z over real hits)
    # Use masked arrays
    r_masked = np.where(mask, r_all, np.nan)
    phi_masked = np.where(mask, phi_all, np.nan)
    z_masked = np.where(mask, z_all, np.nan)
    feats["r_mean"] = np.nanmean(r_masked, axis=1)
    feats["r_std"] = np.nanstd(r_masked, axis=1)
    feats["z_mean"] = np.nanmean(z_masked, axis=1)
    feats["z_std"] = np.nanstd(z_masked, axis=1)
    feats["phi_mean"] = np.nanmean(phi_masked, axis=1)
    feats["phi_std"] = np.nanstd(phi_masked, axis=1)

    # Conformal coordinates of first 2 hits: u = x/(x²+y²), v = y/(x²+y²)
    for j in range(2):
        r_j = feats[f"r{j}"]
        phi_j = feats[f"phi{j}"]
        x_j = r_j * np.cos(phi_j)
        y_j = r_j * np.sin(phi_j)
        r2 = x_j**2 + y_j**2 + 1e-10
        feats[f"u{j}"] = x_j / r2
        feats[f"v{j}"] = y_j / r2

    # Circle fit curvature from first 3 hits (Menger curvature)
    # K = 4*area(triangle) / (|p0-p1| * |p1-p2| * |p2-p0|)
    x0 = feats["r0"] * np.cos(feats["phi0"])
    y0 = feats["r0"] * np.sin(feats["phi0"])
    x1 = feats["r1"] * np.cos(feats["phi1"])
    y1 = feats["r1"] * np.sin(feats["phi1"])
    x2 = feats["r2"] * np.cos(feats["phi2"])
    y2 = feats["r2"] * np.sin(feats["phi2"])
    area2 = np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
    d01 = np.sqrt((x1-x0)**2 + (y1-y0)**2) + 1e-10
    d12 = np.sqrt((x2-x1)**2 + (y2-y1)**2) + 1e-10
    d02 = np.sqrt((x2-x0)**2 + (y2-y0)**2) + 1e-10
    feats["curvature"] = np.clip(4 * area2 / (d01 * d12 * d02 + 1e-10), -1, 1)
    feats["curvature"] = np.where(n_hits >= 3, feats["curvature"], 0.0)

    # dz/dr slope (first to last)
    feats["dz_dr_total"] = feats["delta_z"] / (feats["delta_r"] + 1e-10)

    # Hits per detector type: pixel (det < 3) vs strip (det >= 3)
    pixel_mask = mask & (det_all < 3)
    strip_mask = mask & (det_all >= 3)
    feats["n_pixel_hits"] = pixel_mask.sum(axis=1).astype(np.float32)
    feats["n_strip_hits"] = strip_mask.sum(axis=1).astype(np.float32)

    # z-extrapolation features (critical for z0 and theta)
    # Linear extrapolation of z(r) to r=0: z_intercept ≈ z0
    # slope = (z_last - z_first) / (r_last - r_first) ≈ cot(theta)
    dz_dr_slope = feats["delta_z"] / (feats["delta_r"] + 1e-10)
    feats["dz_dr_slope"] = dz_dr_slope
    feats["z_intercept"] = feats["z0"] - dz_dr_slope * feats["r0"]  # extrapolate to r=0

    # z-sagitta: deviation of middle z from linear z(r) interpolation
    z_mid = z_all[arange_N, mid_idx]
    z_mid_expected = feats["z0"] + dz_dr_slope * (r_all[arange_N, mid_idx] - feats["r0"])
    feats["z_sagitta_mid"] = np.where(n_hits >= 3, z_mid - z_mid_expected, 0.0)

    # Better slope from innermost 2 hits (less affected by material)
    dr_01 = feats["r1"] - feats["r0"]
    feats["dz_dr_inner"] = (feats["z1"] - feats["z0"]) / (dr_01 + 1e-10)
    feats["z_intercept_inner"] = feats["z0"] - feats["dz_dr_inner"] * feats["r0"]

    # cot(theta) proxy from outer hits (less affected by d0)
    dr_12 = feats["r2"] - feats["r1"]
    feats["dz_dr_outer"] = np.where(n_hits >= 3, (feats["z2"] - feats["z1"]) / (dr_12 + 1e-10), 0.0)

    # Stack into array
    feature_names = list(feats.keys())
    features = np.stack([feats[k] for k in feature_names], axis=1).astype(np.float32)

    # Replace any NaN/inf with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Truth and reco — use BDT-friendly parameterization:
    # [d0, z0, phi, cot_theta, qop]
    # cot(theta) = cos(theta)/sin(theta) — the natural z/r slope
    truth_m = dataset.truth_params.numpy() * output_scales
    truth_phi = np.arctan2(truth_m[:, 2], truth_m[:, 3])
    truth_cot_theta = np.cos(truth_m[:, 4]) / (np.sin(truth_m[:, 4]) + 1e-10)
    truth_raw = np.stack([truth_m[:, 0], truth_m[:, 1], truth_phi, truth_cot_theta, truth_m[:, 5]], axis=1)

    reco_m = dataset.reco_params.numpy()
    reco_phi = np.arctan2(reco_m[:, 2], reco_m[:, 3])
    reco_cot_theta = np.cos(reco_m[:, 4]) / (np.sin(reco_m[:, 4]) + 1e-10)
    reco_raw = np.stack([reco_m[:, 0], reco_m[:, 1], reco_phi, reco_cot_theta, reco_m[:, 5]], axis=1)

    return features, feature_names, truth_raw, reco_raw


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
    for i, name in enumerate(BDT_PARAM_NAMES):
        y_train = truth_raw[train_idx, i]
        y_val = truth_raw[val_idx, i]

        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            early_stopping_rounds=20,
            random_state=seed,
            n_jobs=4,
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
    for i, name in enumerate(BDT_PARAM_NAMES):
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
    features, feature_names, truth_raw, reco_raw = extract_bdt_features(train_ds)
    print(f"  Features shape: {features.shape} ({len(feature_names)} features) in {time.time()-t0:.1f}s")
    print(f"  Features: {feature_names}")

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

    def cot_to_theta(arr):
        """Convert [d0,z0,phi,cot_theta,qop] -> [d0,z0,phi,theta,qop]."""
        out = arr.copy()
        out[:, 3] = np.arctan2(1.0, arr[:, 3])
        return out

    val_pred_std = cot_to_theta(val_pred)
    val_truth_std = cot_to_theta(val_truth)
    val_reco_std = cot_to_theta(val_reco)
    STD_PARAM_NAMES = ["d0", "z0", "phi", "theta", "qop"]

    metrics = {}
    print("  (standard parameterization [d0, z0, phi, theta, qop]):")
    for i, name in enumerate(STD_PARAM_NAMES):
        bdt_res = val_pred_std[:, i] - val_truth_std[:, i]
        kf_res = val_reco_std[:, i] - val_truth_std[:, i]
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

    # Generate plots — convert from BDT parameterization [d0,z0,phi,cot_theta,qop]
    # back to standard [d0,z0,phi,theta,qop] for plotting
    print("\nGenerating plots...")
    from evaluation.plotting import make_all_residual_plots
    import matplotlib
    matplotlib.use("Agg")

    figs = make_all_residual_plots(
        cot_to_theta(val_pred), cot_to_theta(val_reco), cot_to_theta(val_truth),
        output_dir=str(output_dir),
    )

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
