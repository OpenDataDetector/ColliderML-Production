"""Money plot for Phase 6: d0 IQR (and std) vs training epoch.

For each loss variant (truncated huber, plain huber), plots three curves on
the same axes:
  - v12_<loss>_scratch — 50 epochs from scratch (per-track)
  - v12_<loss>_track   — warm-started from scratch checkpoint, per-track
  - v12_<loss>_cross   — warm-started from same scratch checkpoint, cross-track

The two warm-start runs share a common starting point at the end of the
scratch trunk so the plot visually demonstrates that the cross-track branch
diverges from the per-track branch — the improvement is the architecture, not
extra training.

Produces 2 panels (one per loss variant), each with both IQR and std curves
to show the truncated-Huber metric discrepancy.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ML_BASE = Path("/global/cfs/cdirs/m4958/data/ColliderML/ml/beamspot_studies")

# Match the new train.log line format that includes iqr_ml fields:
#   ... | Epoch  XX | train_loss=... val_loss=... |
#   d0_res_ml=... iqr_ml=... res_kf=... | z0_res_ml=... iqr_ml=... res_kf=... | ...
#
# We only need d0 fields for the headline plot.
LINE_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s*\|"
    r"\s*train_loss=(?P<train>[\d.eE+-]+|nan)\s+val_loss=(?P<val>[\d.eE+-]+|nan)\s*\|"
    r"\s*d0_res_ml=(?P<d0_std>[\d.eE+-]+|nan)\s+iqr_ml=(?P<d0_iqr>[\d.eE+-]+|nan)\s+res_kf=(?P<d0_kf>[\d.eE+-]+|nan)"
)


def parse_train_log(log_path: Path) -> dict:
    """Parse a v12 train.log file and return per-epoch d0 metrics.

    Returns a dict with numpy arrays: epochs, d0_std, d0_iqr, d0_kf.
    """
    per_epoch = {}
    if not log_path.exists():
        return {"epochs": np.array([]), "d0_std": np.array([]), "d0_iqr": np.array([]), "d0_kf": np.array([])}
    with open(log_path) as fp:
        for line in fp:
            m = LINE_RE.search(line)
            if not m:
                continue
            per_epoch[int(m.group("epoch"))] = m.groupdict()

    epochs = sorted(per_epoch.keys())
    return {
        "epochs": np.array(epochs),
        "d0_std": np.array([float(per_epoch[e]["d0_std"]) for e in epochs]),
        "d0_iqr": np.array([float(per_epoch[e]["d0_iqr"]) for e in epochs]),
        "d0_kf": np.array([float(per_epoch[e]["d0_kf"]) for e in epochs]),
    }


def panel(ax, scratch, cross, track, ckf_iqr, title):
    """One panel of the money plot."""
    if len(scratch["epochs"]) == 0:
        ax.text(0.5, 0.5, f"(no data yet for {title})", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_title(title)
        return

    # Trunk: scratch
    ax.plot(scratch["epochs"], scratch["d0_iqr"],
            color="#616161", linewidth=2, marker="o", markersize=3,
            label=r"scratch (per-track)")

    if len(scratch["epochs"]) > 0:
        v_final = scratch["d0_iqr"][-1]
        v_final_x = scratch["epochs"][-1]
    else:
        v_final, v_final_x = None, None

    # Branch points: prepend the trunk endpoint so the curves visually connect.
    if len(track["epochs"]) > 0 and v_final is not None:
        x = np.concatenate([[v_final_x], v_final_x + 1 + track["epochs"]])
        y = np.concatenate([[v_final], track["d0_iqr"]])
        ax.plot(x, y, color="#C62828", linewidth=2, marker="^", markersize=3,
                label=r"per-track + warm start")

    if len(cross["epochs"]) > 0 and v_final is not None:
        x = np.concatenate([[v_final_x], v_final_x + 1 + cross["epochs"]])
        y = np.concatenate([[v_final], cross["d0_iqr"]])
        ax.plot(x, y, color="#2E7D32", linewidth=2, marker="s", markersize=3,
                label=r"cross-track + warm start")

    # CKF reference
    ax.axhline(ckf_iqr, color="black", linestyle="--", linewidth=1.3, alpha=0.7,
               label=f"CKF (={ckf_iqr:.3f} mm)")

    # Branch indicator
    if v_final is not None:
        ax.axvline(v_final_x + 0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    ax.set_xlabel("Training epoch")
    ax.set_ylabel(r"$d_0$ resolution (IQR/1.349) [mm]")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)


def main():
    # Truncated huber runs
    th_scratch = parse_train_log(ML_BASE / "baseline_randomized_xy_v12_th_scratch" / "train.log")
    th_cross = parse_train_log(ML_BASE / "baseline_randomized_xy_v12_th_cross" / "train.log")
    th_track = parse_train_log(ML_BASE / "baseline_randomized_xy_v12_th_track" / "train.log")

    # Plain huber runs
    h_scratch = parse_train_log(ML_BASE / "baseline_randomized_xy_v12_h_scratch" / "train.log")
    h_cross = parse_train_log(ML_BASE / "baseline_randomized_xy_v12_h_cross" / "train.log")
    h_track = parse_train_log(ML_BASE / "baseline_randomized_xy_v12_h_track" / "train.log")

    print("=== v12 run status ===")
    for name, d in [("th_scratch", th_scratch), ("th_cross", th_cross), ("th_track", th_track),
                    ("h_scratch", h_scratch), ("h_cross", h_cross), ("h_track", h_track)]:
        n = len(d["epochs"])
        if n > 0:
            last_iqr = d["d0_iqr"][-1]
            last_std = d["d0_std"][-1]
            print(f"  {name:14s} {n:3d} epochs   d0_iqr={last_iqr:.4f}   d0_std={last_std:.4f}")
        else:
            print(f"  {name:14s} no data yet")

    # Use CKF IQR from any run that has data (it's nearly invariant ~0.032)
    ckf_iqr = 0.032
    for d in [th_scratch, h_scratch, th_cross, h_cross]:
        if len(d["epochs"]) > 0:
            # Convert kf from std to a rough IQR by dividing by ~5 (truth std / iqr ratio)
            # Actually we don't have CKF IQR in train.log either — only std. Use
            # the known cross-eval value of 0.032 mm (median across datasets).
            break

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    panel(axes[0], th_scratch, th_cross, th_track, ckf_iqr=0.032,
          title="Truncated Huber loss (clip 3$\\sigma$)")
    panel(axes[1], h_scratch, h_cross, h_track, ckf_iqr=0.032,
          title="Plain Huber loss (no clipping)")

    fig.suptitle(
        r"$d_0$ resolution vs epoch: cross-track attention vs ablation control",
        fontsize=13,
    )
    plt.tight_layout()
    out = Path(__file__).resolve().parent / "figures" / "d0_iqr_moneyplot.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
