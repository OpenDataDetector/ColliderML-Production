"""Diagnostic: does the [EVT] token linearly predict the true primary vertex?

Loads a v12 cross-track model, runs it on the held-out randomized dataset,
extracts the (B_ev, d_model) [EVT] embedding for each event, and matches
each event to its true primary vertex (vx, vy, vz) from the source particles
parquet. Fits a separate linear probe (sklearn LinearRegression) for each
of vx, vy, vz, and reports the R^2 of the probe plus a scatter plot.

If the model has discovered the beam spot from the ensemble of tracks, the
[EVT] token should linearly encode the true vertex with high R^2 — at least
for vx and vy where the randomization spans 300 um.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

REPO = Path("/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software/colliderml_dev")
sys.path.insert(0, str(REPO / "ml" / "beamspot_studies"))

from data.datasets import TrackHitDataset  # noqa: E402
from data.event_collate import make_event_dataloader  # noqa: E402
from evaluation.evaluate import load_model  # noqa: E402

ML_BASE = Path("/global/cfs/cdirs/m4958/data/ColliderML/ml/beamspot_studies")
SIM_BASE = Path("/global/cfs/cdirs/m4958/data/ColliderML/simulation")

CKPT = next(
    (ML_BASE / "baseline_randomized_xy_v12_th_cross" / "checkpoints").rglob("loss=*.ckpt")
)
PARQUET = SIM_BASE / "beamspot_studies" / "ttbar_randomized_xy" / "v1" / "parquet"

EVAL_MAX_FILES = 5
EVAL_SKIP = (0, 7999)


def build_event_id_to_vertex(parquet_base: Path, max_files: int, skip_range, numeric_sort: bool) -> dict[int, tuple[float, float, float]]:
    """Replicate the dataset's file selection logic and walk particle tables to
    extract the primary vertex for each global event id.

    The dataset assigns event ids by a global counter that increments by
    `len(tracks_tbl)` per file (see datasets.py:_process_event loop). We
    reproduce that here to keep the event_id mapping consistent.
    """
    import glob
    files = sorted(glob.glob(str(parquet_base / "reco/tracks/*.parquet")))
    if numeric_sort:
        # Match TrackHitDataset._sort_numeric: sort by numeric event start index
        def _start(p):
            m = re.search(r"events(\d+)-", Path(p).name)
            return int(m.group(1)) if m else 0
        files = sorted(files, key=_start)

    # Apply skip-event-range first (matches dataset code)
    if skip_range is not None:
        skip_start, skip_end = skip_range
        keep = []
        for f in files:
            m = re.search(r"events(\d+)-(\d+)", Path(f).name)
            if not m:
                keep.append(f)
                continue
            es, ee = int(m.group(1)), int(m.group(2))
            if ee < skip_start or es > skip_end:
                keep.append(f)
        files = keep

    if max_files is not None:
        files = files[:max_files]

    eid_to_vertex: dict[int, tuple[float, float, float]] = {}
    global_event_counter = 0
    for tf in files:
        # Determine the corresponding particles file path
        particles_file = str(tf).replace("/reco/tracks/", "/truth/particles/").replace(
            ".reco.tracks.", ".truth.particles."
        )
        ptbl = pq.read_table(particles_file, columns=["vx", "vy", "vz"])
        n_events_in_file = len(ptbl)
        for ei in range(n_events_in_file):
            vx = ptbl["vx"][ei].as_py()[0]  # primary vertex (first particle)
            vy = ptbl["vy"][ei].as_py()[0]
            vz = ptbl["vz"][ei].as_py()[0]
            eid_to_vertex[global_event_counter + ei] = (float(vx), float(vy), float(vz))
        global_event_counter += n_events_in_file

    return eid_to_vertex


@torch.no_grad()
def collect_evt_embeddings(module, dataset, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Run the cross-track model in eval mode, return (event_ids, evt_embeddings)."""
    hparams = module.hparams
    loader = make_event_dataloader(
        dataset,
        event_ids=dataset.event_ids,  # full dataset, no Subset
        batch_size_events=int(hparams.get("batch_size_events", 8)),
        max_tracks_per_event=int(hparams.get("max_tracks_per_event", 128)),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    all_eids = []
    all_evts = []
    for batch in loader:
        track_mask = batch["track_mask"].to(device)
        evt = module.model.extract_evt_embedding(
            batch["hit_features"].to(device),
            batch["padding_mask"].to(device),
            batch.get("cls_features").to(device) if batch.get("cls_features") is not None else None,
            track_mask,
        )  # (B_ev, d_model)
        all_eids.append(batch["event_ids"].numpy())
        all_evts.append(evt.cpu().numpy())

    return np.concatenate(all_eids), np.concatenate(all_evts, axis=0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Checkpoint: {CKPT}")

    print("\nLoading model...")
    module = load_model(str(CKPT), device=device)

    print(f"\nLoading randomized eval dataset (max_files={EVAL_MAX_FILES}, skip {EVAL_SKIP})...")
    dataset = TrackHitDataset(
        str(PARQUET),
        max_files=EVAL_MAX_FILES,
        skip_event_range=EVAL_SKIP,
        numeric_sort=True,
    )
    print(f"  {len(dataset)} eval tracks across {len(set(int(e) for e in dataset.event_ids.tolist()))} events")

    print("\nBuilding event_id -> truth vertex lookup from particles parquet...")
    eid_to_vertex = build_event_id_to_vertex(
        PARQUET, max_files=EVAL_MAX_FILES, skip_range=EVAL_SKIP, numeric_sort=True,
    )
    print(f"  {len(eid_to_vertex)} events in lookup")

    print("\nExtracting [EVT] token embeddings...")
    eids, evts = collect_evt_embeddings(module, dataset, device)
    print(f"  Got {len(eids)} event embeddings (d_model={evts.shape[1]})")

    # Match each event to its true vertex
    truth_v = np.array([eid_to_vertex[int(e)] for e in eids])  # (N_ev, 3)
    print(f"  Truth vertex stats:")
    for i, name in enumerate(["vx", "vy", "vz"]):
        v = truth_v[:, i]
        unit = "mm" if name == "vz" else "um"
        scale = 1.0 if name == "vz" else 1000.0
        print(f"    {name}: mean={v.mean()*scale:+.2f} {unit}, std={v.std()*scale:.2f} {unit}")

    # Linear probe per axis
    print("\n=== Linear probe (W @ evt_emb + b -> truth vertex coordinate) ===")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    results = {}
    for i, (name, unit, scale) in enumerate([("v_x", "$\\mu$m", 1000), ("v_y", "$\\mu$m", 1000), ("v_z", "mm", 1.0)]):
        y = truth_v[:, i] * scale  # convert to display units
        # 80/20 train/test split for honest R^2
        n = len(y)
        rng = np.random.default_rng(42)
        idx = rng.permutation(n)
        n_train = int(0.8 * n)
        train_idx, test_idx = idx[:n_train], idx[n_train:]

        probe = LinearRegression()
        probe.fit(evts[train_idx], y[train_idx])
        y_pred_test = probe.predict(evts[test_idx])
        r2 = r2_score(y[test_idx], y_pred_test)
        residual_std = float(np.std(y[test_idx] - y_pred_test))
        truth_std = float(np.std(y[test_idx]))
        plain_unit = unit.replace("$\\mu$", "u")
        print(f"  {name}: R^2={r2:.4f}  truth std={truth_std:.2f} {plain_unit}  residual std={residual_std:.2f} {plain_unit}")
        results[name] = {"r2": r2, "truth_std": truth_std, "residual_std": residual_std}

        ax = axes[i]
        # Scatter (test set)
        lim_min = min(y[test_idx].min(), y_pred_test.min())
        lim_max = max(y[test_idx].max(), y_pred_test.max())
        ax.scatter(y[test_idx], y_pred_test, s=8, alpha=0.5, color="#1565C0")
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1, alpha=0.5)
        ax.set_xlabel(f"True ${name}$ [{unit}]")
        ax.set_ylabel(f"Linear probe of [EVT] [{unit}]")
        ax.set_title(f"${name}$ probe ($R^2$={r2:.3f})")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    fig.suptitle(
        r"Linear probe of cross-track [EVT] token: does it know the primary vertex?",
        fontsize=13,
    )
    plt.tight_layout()
    out = Path(__file__).resolve().parent / "figures" / "evt_token_vertex_probe.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # Save numbers
    import json
    out_json = Path(__file__).resolve().parent / "evt_token_probe_results.json"
    with open(out_json, "w") as fp:
        json.dump({
            "checkpoint": str(CKPT),
            "n_events": int(len(eids)),
            "d_model": int(evts.shape[1]),
            "results": results,
        }, fp, indent=2)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
