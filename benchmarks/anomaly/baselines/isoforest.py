"""
Isolation Forest baseline for anomaly detection.

Features are per-event global kinematic observables:
    - total track pT
    - number of tracks
    - missing ET (approximate from track momentum imbalance)
    - leading track pT

Trained on SM events (ttbar, zmumu, zee), scored on a mixed held-out set
containing the four BSM channels.

Usage:
    python benchmarks/anomaly/baselines/isoforest.py --output iso_preds.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SM_CHANNELS = ["ttbar_pu0", "zmumu_pu0", "zee_pu0"]
BSM_CHANNELS = ["higgs_portal_pu0", "susy_gmsb_pu0", "hidden_valley_pu0", "zprime_pu0"]


def _event_features(tracks_table) -> "pd.DataFrame":
    """Summarise each event into a 4-dim feature vector."""
    import numpy as np
    import pandas as pd

    df = tracks_table.to_pandas() if hasattr(tracks_table, "to_pandas") else tracks_table
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    px = df.get("px", pd.Series(0.0, index=df.index))
    py = df.get("py", pd.Series(0.0, index=df.index))
    df["pt"] = np.sqrt(px.pow(2) + py.pow(2))
    g = df.groupby("event_id")
    out = g.agg(
        total_pt=("pt", "sum"),
        n_tracks=("pt", "count"),
        met_x=("px" if "px" in df.columns else "pt", "sum"),
        met_y=("py" if "py" in df.columns else "pt", "sum"),
        leading_pt=("pt", "max"),
    ).reset_index()
    out["met"] = np.sqrt(out["met_x"].pow(2) + out["met_y"].pow(2))
    return out[["event_id", "total_pt", "n_tracks", "met", "leading_pt"]]


def run_baseline(output: Path, max_events: int = 500):
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        print("scikit-learn not installed. Run: pip install scikit-learn", file=sys.stderr)
        sys.exit(1)

    import colliderml
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Load SM training features
    print("Loading SM channels for training...")
    sm_feats = []
    for ch in SM_CHANNELS:
        try:
            tracks = colliderml.load(ch, tables=["tracks"], max_events=max_events)
            sm_feats.append(_event_features(tracks))
        except Exception as e:
            print(f"  skipping {ch}: {e}")
    if not sm_feats:
        print("No SM training data available.", file=sys.stderr)
        sys.exit(1)
    X_train = pd.concat(sm_feats, ignore_index=True)
    feature_cols = ["total_pt", "n_tracks", "met", "leading_pt"]

    print(f"Training IsolationForest on {len(X_train)} SM events...")
    iso = IsolationForest(contamination="auto", random_state=42)
    iso.fit(X_train[feature_cols].values)

    # Score both SM (as normal) and BSM (as anomaly) on a held-out set
    print("Scoring mixed held-out set...")
    rows = []
    for ch in SM_CHANNELS + BSM_CHANNELS:
        try:
            tracks = colliderml.load(ch, tables=["tracks"], max_events=max_events // 2)
        except Exception:
            continue
        feats = _event_features(tracks)
        if feats.empty:
            continue
        # Higher score = more anomalous (flip sklearn's convention)
        scores = -iso.score_samples(feats[feature_cols].values)
        for evt, s in zip(feats["event_id"], scores):
            rows.append({"event_id": int(evt), "channel": ch, "anomaly_score": float(s)})

    if not rows:
        print("No held-out events scored.", file=sys.stderr)
        sys.exit(1)

    table = pa.table({
        "event_id": [r["event_id"] for r in rows],
        "channel": [r["channel"] for r in rows],
        "anomaly_score": [r["anomaly_score"] for r in rows],
    })
    pq.write_table(table, output)
    print(f"Wrote {table.num_rows} predictions to {output}")


def main():
    parser = argparse.ArgumentParser(description="IsolationForest anomaly baseline")
    parser.add_argument("--output", type=Path, default=Path("iso_preds.parquet"))
    parser.add_argument("--max-events", type=int, default=500)
    args = parser.parse_args()
    run_baseline(args.output, args.max_events)


if __name__ == "__main__":
    main()
