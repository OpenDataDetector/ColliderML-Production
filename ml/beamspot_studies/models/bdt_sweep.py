"""
BDT hyperparameter sweep: XGBoost vs LightGBM configurations.

Trains each config on the same data, reports STD/IQR/MAE for all params.

Usage:
    python -u bdt_sweep.py --max-files 2
"""

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import argparse
import json
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
import lightgbm as lgb

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.datasets import TrackHitDataset
from models.bdt_baseline import extract_bdt_features, BDT_PARAM_NAMES
from evaluation.evaluate import compute_metrics, print_metrics

STD_PARAM_NAMES = ["d0", "z0", "phi", "theta", "qop"]


def cot_to_theta(arr):
    out = arr.copy()
    out[:, 3] = np.arctan2(1.0, arr[:, 3])
    return out


CONFIGS = [
    {
        "name": "XGB-base",
        "lib": "xgb",
        "params": dict(n_estimators=500, max_depth=8, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                        early_stopping_rounds=20, n_jobs=4),
    },
    {
        "name": "XGB-big",
        "lib": "xgb",
        "params": dict(n_estimators=1000, max_depth=12, learning_rate=0.03,
                        subsample=0.8, colsample_bytree=0.7, tree_method="hist",
                        early_stopping_rounds=30, n_jobs=4, min_child_weight=5),
    },
    {
        "name": "XGB-huber",
        "lib": "xgb",
        "params": dict(n_estimators=500, max_depth=8, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                        early_stopping_rounds=20, n_jobs=4,
                        objective="reg:pseudohubererror"),
    },
    {
        "name": "LGB-default",
        "lib": "lgb",
        "params": dict(n_estimators=500, max_depth=-1, num_leaves=31,
                        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                        n_jobs=4, verbose=-1, max_bin=512),
    },
    {
        "name": "LGB-wide",
        "lib": "lgb",
        "params": dict(n_estimators=1000, max_depth=-1, num_leaves=256,
                        learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
                        n_jobs=4, verbose=-1, max_bin=512, min_child_samples=20),
    },
    {
        "name": "LGB-deep",
        "lib": "lgb",
        "params": dict(n_estimators=1000, max_depth=12, num_leaves=4096,
                        learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
                        n_jobs=4, verbose=-1, max_bin=1024, min_child_samples=10),
    },
    {
        "name": "LGB-huber",
        "lib": "lgb",
        "params": dict(n_estimators=1000, max_depth=-1, num_leaves=256,
                        learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
                        n_jobs=4, verbose=-1, max_bin=512, objective="huber"),
    },
]


def train_and_eval(config, X_train, y_train, X_val, y_val):
    """Train one BDT config on all params, return predictions."""
    pred = np.zeros((len(X_val), 5))
    iters = []

    for i, name in enumerate(BDT_PARAM_NAMES):
        if config["lib"] == "xgb":
            m = xgb.XGBRegressor(**config["params"], random_state=42)
            m.fit(X_train, y_train[:, i],
                  eval_set=[(X_val, y_val[:, i])], verbose=False)
            best_iter = m.best_iteration
        else:
            m = lgb.LGBMRegressor(**config["params"], random_state=42)
            es = config["params"].get("early_stopping_rounds", 20)
            m.fit(X_train, y_train[:, i],
                  eval_set=[(X_val, y_val[:, i])],
                  callbacks=[lgb.early_stopping(es, verbose=False),
                             lgb.log_evaluation(0)])
            best_iter = m.best_iteration_

        pred[:, i] = m.predict(X_val)
        iters.append(best_iter)

    return pred, iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-base", type=str,
                        default="/global/cfs/cdirs/m4958/data/ColliderML/simulation/hard_scatter/ttbar/v1/parquet")
    parser.add_argument("--max-files", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="/tmp/bdt_sweep")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    ds = TrackHitDataset(args.parquet_base, max_files=args.max_files)
    features, fnames, truth, reco = extract_bdt_features(ds)
    print(f"{len(features)} tracks, {len(fnames)} features\n")

    # Split
    N = len(features)
    rng = np.random.RandomState(42)
    idx = rng.permutation(N)
    n_val = int(N * 0.1)
    tr, va = idx[n_val:], idx[:n_val]

    # Run sweep
    all_results = {}
    for config in CONFIGS:
        t0 = time.time()
        print(f"--- {config['name']} ---")

        pred_cot, iters = train_and_eval(
            config, features[tr], truth[tr], features[va], truth[va])

        elapsed = time.time() - t0
        iters_str = ", ".join(f"{BDT_PARAM_NAMES[i]}={it}" for i, it in enumerate(iters))
        print(f"  Time: {elapsed:.0f}s | Iters: {iters_str}")

        # Convert to standard theta and compute metrics
        pred_std = cot_to_theta(pred_cot)
        truth_std = cot_to_theta(truth[va])
        reco_std = cot_to_theta(reco[va])

        metrics = compute_metrics(pred_std, truth_std, reco_std,
                                  param_names=STD_PARAM_NAMES,
                                  pred_label="bdt", ref_label="kf")

        # Print compact summary
        for name in STD_PARAM_NAMES:
            m = metrics[name]
            print(f"  {name:8s}: STD={m['bdt_std']:.4g}  IQR={m['bdt_iqr']:.4g}  "
                  f"MAE={m['bdt_mae']:.4g}  | KF: STD={m['kf_std']:.4g}  IQR={m['kf_iqr']:.4g}")

        all_results[config["name"]] = {"metrics": metrics, "time": elapsed, "iters": iters}
        print()

    # Summary table
    print("\n" + "=" * 100)
    print(f"{'Config':<15s}", end="")
    for name in STD_PARAM_NAMES:
        print(f" {name+' IQR':>10s}", end="")
    print(f" {'Time':>6s}")
    print("-" * 100)
    for cname, res in all_results.items():
        print(f"{cname:<15s}", end="")
        for name in STD_PARAM_NAMES:
            print(f" {res['metrics'][name]['bdt_iqr']:10.4g}", end="")
        print(f" {res['time']:5.0f}s")
    # CKF reference
    print(f"{'CKF':<15s}", end="")
    kf = list(all_results.values())[0]["metrics"]
    for name in STD_PARAM_NAMES:
        print(f" {kf[name]['kf_iqr']:10.4g}", end="")
    print()

    with open(Path(args.output_dir) / "sweep_results.json", "w") as f:
        json.dump({k: {"time": v["time"], "iters": v["iters"],
                        "metrics": {p: {mk: mv for mk, mv in m.items()}
                                    for p, m in v["metrics"].items()}}
                   for k, v in all_results.items()}, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/sweep_results.json")


if __name__ == "__main__":
    main()
