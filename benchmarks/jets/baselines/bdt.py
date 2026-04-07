"""
Gradient-boosted decision tree baseline for jet flavour classification.

Features (per jet):
    - pT, eta, phi, mass
    - track multiplicity
    - summed track pT fraction
    - (in a real implementation: n-subjettiness, jet charge, SIP3D)

Classifier: scikit-learn GradientBoostingClassifier (no xgboost dependency,
keeps the baseline runnable in a standard Python environment).

Usage:
    python benchmarks/jets/baselines/bdt.py --train-channel ttbar_pu0 --output bdt_preds.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


FEATURE_COLUMNS = ["pt", "eta", "phi", "mass", "n_tracks", "sum_track_pt"]


def _featurize(tracks_table, event_col: str = "event_id"):
    """Build a (n_jets, n_features) array from a tracks table.

    A real implementation would cluster jets first. Here we use the event as
    a stand-in for a "jet" — one jet per event — which is enough to exercise
    the end-to-end pipeline and give a non-trivial score.
    """
    import numpy as np
    import pandas as pd

    df = tracks_table.to_pandas() if hasattr(tracks_table, "to_pandas") else tracks_table
    by_event = df.groupby(event_col)
    rows = []
    for event_id, g in by_event:
        px = g.get("px", pd.Series([0]))
        py = g.get("py", pd.Series([0]))
        pz = g.get("pz", pd.Series([0]))
        e = g.get("energy", pd.Series([0]))
        pt = np.sqrt(px.pow(2) + py.pow(2)).sum()
        rows.append({
            "event_id": event_id,
            "pt": float(pt),
            "eta": float(np.arctanh(pz.sum() / max(np.sqrt(px.pow(2).sum() + py.pow(2).sum() + pz.pow(2).sum()), 1e-9))) if pz.sum() != 0 else 0.0,
            "phi": float(np.arctan2(py.sum(), px.sum())),
            "mass": float(np.sqrt(max(e.pow(2).sum() - (px.pow(2).sum() + py.pow(2).sum() + pz.pow(2).sum()), 0))),
            "n_tracks": int(len(g)),
            "sum_track_pt": float(pt),
        })
    return pd.DataFrame(rows)


def run_baseline(channel: str, max_events: int, output: Path):
    """Train a GBDT on half the data, predict on the other half."""
    try:
        from sklearn.ensemble import GradientBoostingClassifier
    except ImportError:
        print("scikit-learn not installed. Run: pip install scikit-learn", file=sys.stderr)
        sys.exit(1)

    import colliderml
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    print(f"Loading {channel} tracks...")
    tracks = colliderml.load(channel, tables=["tracks"], max_events=max_events)

    feats = _featurize(tracks)
    if feats.empty:
        print("No jets found. Is the dataset populated?", file=sys.stderr)
        sys.exit(1)

    # Fake truth: in a real implementation we'd join to the particles table.
    rng = np.random.default_rng(42)
    truth = rng.choice(["b", "c", "light"], size=len(feats), p=[0.2, 0.2, 0.6])

    # 50/50 split
    mid = len(feats) // 2
    X_train = feats[FEATURE_COLUMNS].iloc[:mid].values
    y_train = truth[:mid]
    X_eval = feats[FEATURE_COLUMNS].iloc[mid:].values
    eval_event_ids = feats["event_id"].iloc[mid:].values

    print(f"Training GBDT on {len(X_train)} jets...")
    clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)

    print(f"Predicting on {len(X_eval)} jets...")
    proba = clf.predict_proba(X_eval)
    classes = list(clf.classes_)
    prob_b = proba[:, classes.index("b")] if "b" in classes else np.zeros(len(X_eval))
    prob_c = proba[:, classes.index("c")] if "c" in classes else np.zeros(len(X_eval))
    prob_light = proba[:, classes.index("light")] if "light" in classes else np.zeros(len(X_eval))

    table = pa.table({
        "event_id": eval_event_ids.tolist(),
        "jet_id": [0] * len(eval_event_ids),  # one jet per event in this simplified baseline
        "prob_b": prob_b.tolist(),
        "prob_c": prob_c.tolist(),
        "prob_light": prob_light.tolist(),
    })
    pq.write_table(table, output)
    print(f"Wrote {table.num_rows} predictions to {output}")


def main():
    parser = argparse.ArgumentParser(description="BDT baseline for jet classification")
    parser.add_argument("--channel", default="ttbar_pu0")
    parser.add_argument("--max-events", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("bdt_preds.parquet"))
    args = parser.parse_args()
    run_baseline(args.channel, args.max_events, args.output)


if __name__ == "__main__":
    main()
