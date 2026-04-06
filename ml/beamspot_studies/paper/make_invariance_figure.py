"""Generate the key 'beam spot invariance' figure for the paper.

Shows d0 resolution vs eta for v10-randomized evaluated on three different
beam spot datasets (nominal, 25um, 300um), overlaid with CKF as the
beam-spot-invariant baseline. The three randomized curves should be
essentially indistinguishable, demonstrating beam-spot invariance.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.datasets import TrackHitDataset
from evaluation.evaluate import load_model, run_inference
from evaluation.plotting import _bin_resolution

SIM_BASE = "/global/cfs/cdirs/m4958/data/ColliderML/simulation"
ML_BASE = "/global/cfs/cdirs/m4958/data/ColliderML/ml/beamspot_studies"

RAND_CKPT = (
    f"{ML_BASE}/baseline_randomized_xy_v10/checkpoints/best-epoch=049-val/loss=0.0046.ckpt"
)
NOM_CKPT = (
    f"{ML_BASE}/baseline_nominal_v10/checkpoints/best-epoch=047-val/loss=0.0025.ckpt"
)

EVAL_DATASETS = {
    "Nominal": f"{SIM_BASE}/hard_scatter/ttbar/v1/parquet",
    r"Shifted $25\,\mu$m": f"{SIM_BASE}/beamspot_studies/ttbar_shifted_25um/v1/parquet",
    r"Shifted $300\,\mu$m": f"{SIM_BASE}/beamspot_studies/ttbar_shifted_300um/v1/parquet",
}

SKIP_RANGES = {
    "Nominal": (0, 9999),
    r"Shifted $25\,\mu$m": (0, 1999),
    r"Shifted $300\,\mu$m": (0, 1999),
}


def gather_residuals(ckpt_path, dataset_path, skip_range, max_files=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = load_model(ckpt_path, device)
    dataset = TrackHitDataset(
        dataset_path, max_files=max_files,
        skip_event_range=skip_range, numeric_sort=True,
    )
    pred, truth, reco = run_inference(module, dataset, batch_size=512, device=device)
    # d0 is index 0
    ml_res = pred[:, 0] - truth[:, 0]
    kf_res = reco[:, 0] - truth[:, 0]
    zero_res = 0.0 - truth[:, 0]  # zero baseline
    eta = -np.log(np.tan(truth[:, 3] / 2 + 1e-10))
    return eta, ml_res, kf_res, zero_res


def bin_iqr_vs_eta(eta, residuals, eta_edges):
    """Compute IQR-based resolution (IQR/1.349) in eta bins."""
    from scipy.stats import iqr as iqr_stat

    centers = 0.5 * (eta_edges[:-1] + eta_edges[1:])
    half_widths = 0.5 * np.diff(eta_edges)
    resolutions = np.full(len(centers), np.nan)
    errors = np.full(len(centers), np.nan)
    for i in range(len(centers)):
        mask = (eta >= eta_edges[i]) & (eta < eta_edges[i + 1])
        n = mask.sum()
        if n > 20:
            sigma = float(iqr_stat(residuals[mask]) / 1.349)
            resolutions[i] = sigma
            errors[i] = sigma / np.sqrt(2 * (n - 1))
    return centers, half_widths, resolutions, errors


def main():
    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    eta_edges = np.linspace(-3.2, 3.2, 21)

    print("=" * 60)
    print("Computing d0 residuals for v10-randomized on each dataset")
    print("=" * 60)

    rand_data = {}
    for name, path in EVAL_DATASETS.items():
        print(f"\n-> v10-randomized on {name}")
        skip = SKIP_RANGES[name]
        eta, ml, kf, zero = gather_residuals(RAND_CKPT, path, skip)
        rand_data[name] = (eta, ml, kf, zero)

    print("\n-> v10-nominal on Nominal (reference)")
    nom_eta, nom_ml, nom_kf, nom_zero = gather_residuals(
        NOM_CKPT, EVAL_DATASETS["Nominal"], SKIP_RANGES["Nominal"],
    )
    print("\n-> v10-nominal on Shifted 300um (off-diagonal failure)")
    nom_300_eta, nom_300_ml, nom_300_kf, nom_300_zero = gather_residuals(
        NOM_CKPT, EVAL_DATASETS[r"Shifted $300\,\mu$m"], SKIP_RANGES[r"Shifted $300\,\mu$m"],
    )

    # =====================================================
    # Figure: d0 resolution vs eta — beam spot invariance
    # =====================================================
    fig, (ax_main, ax_ratio) = plt.subplots(
        2, 1, figsize=(7, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    # Colors and markers for randomized on each dataset
    rand_colors = {
        "Nominal": "#1B5E20",           # dark green
        r"Shifted $25\,\mu$m": "#2E7D32",   # mid green
        r"Shifted $300\,\mu$m": "#66BB6A",  # light green
    }
    markers = {
        "Nominal": "o",
        r"Shifted $25\,\mu$m": "s",
        r"Shifted $300\,\mu$m": "^",
    }

    # Plot randomized-trained (3 curves, should overlap)
    rand_refs = []
    for name in EVAL_DATASETS:
        eta, ml, _, _ = rand_data[name]
        c, hw, res, err = bin_iqr_vs_eta(eta, ml, eta_edges)
        valid = ~np.isnan(res)
        ax_main.errorbar(
            c[valid], res[valid], xerr=hw[valid], yerr=err[valid],
            fmt=markers[name], color=rand_colors[name], markersize=5,
            linewidth=1.2, capsize=2,
            label=f"Randomized-trained, eval on {name}",
        )
        rand_refs.append((c, res))

    # Plot nominal-trained on nominal (best case fixed-BS)
    c, hw, res, err = bin_iqr_vs_eta(nom_eta, nom_ml, eta_edges)
    valid = ~np.isnan(res)
    ax_main.errorbar(
        c[valid], res[valid], xerr=hw[valid], yerr=err[valid],
        fmt="*", color="#1565C0", markersize=7, linewidth=1.2, capsize=2,
        label="Nominal-trained, eval on Nominal (matched)",
    )
    nom_matched = (c, res)

    # Plot nominal-trained on 300um (worst-case off-diagonal)
    c, hw, res, err = bin_iqr_vs_eta(nom_300_eta, nom_300_ml, eta_edges)
    valid = ~np.isnan(res)
    ax_main.errorbar(
        c[valid], res[valid], xerr=hw[valid], yerr=err[valid],
        fmt="X", color="#B71C1C", markersize=6, linewidth=1.2, capsize=2,
        label=r"Nominal-trained, eval on $300\,\mu$m (mismatched)",
    )

    # Plot CKF (use nominal as reference — it's invariant)
    c, hw, res, err = bin_iqr_vs_eta(nom_eta, nom_kf, eta_edges)
    valid = ~np.isnan(res)
    ax_main.errorbar(
        c[valid], res[valid], xerr=hw[valid], yerr=err[valid],
        fmt="o", color="#424242", markersize=4, linewidth=1, capsize=2,
        markerfacecolor="none", label="CKF (any dataset)",
    )

    ax_main.set_ylabel(r"$d_0$ resolution [mm]")
    ax_main.set_title(r"$d_0$ resolution vs $\eta$: beam-spot invariance")
    ax_main.set_yscale("log")
    ax_main.legend(fontsize=8, loc="upper center", ncol=2)
    ax_main.grid(True, alpha=0.3, which="both")

    # Ratio panel: randomized curves / CKF (should be ~1.1, invariant)
    kf_c, _, kf_res, _ = bin_iqr_vs_eta(nom_eta, nom_kf, eta_edges)
    for name in EVAL_DATASETS:
        eta, ml, _, _ = rand_data[name]
        c, hw, res, _ = bin_iqr_vs_eta(eta, ml, eta_edges)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(kf_res > 0, res / kf_res, np.nan)
        valid = ~np.isnan(ratio)
        ax_ratio.plot(
            c[valid], ratio[valid], markers[name], color=rand_colors[name],
            markersize=5, label=name,
        )
    ax_ratio.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax_ratio.set_ylabel("Rand/CKF")
    ax_ratio.set_xlabel(r"$\eta$")
    ax_ratio.set_ylim(0.5, 2.5)
    ax_ratio.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "d0_invariance.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close(fig)

    # =====================================================
    # Figure 2: simple bar chart of d0 IQR across models
    # =====================================================
    fig, ax = plt.subplots(figsize=(7, 4))

    models = [
        "Nominal-trained",
        r"$25\,\mu$m-trained",
        r"$300\,\mu$m-trained",
        "Randomized-trained",
    ]
    eval_datasets = ["Nominal", r"$25\,\mu$m", r"$300\,\mu$m"]

    # IQR values from cross-eval (mm)
    data = np.array([
        [0.0147, 0.0230, 0.0559],  # nominal-trained
        [0.0233, 0.0149, 0.0572],  # 25um-trained
        [0.0526, 0.0537, 0.0151],  # 300um-trained
        [0.0355, 0.0352, 0.0355],  # randomized-trained
    ])
    ckf = np.array([0.0318, 0.0317, 0.0321])

    x = np.arange(len(eval_datasets))
    width = 0.18
    colors = ["#1565C0", "#6A1B9A", "#C62828", "#2E7D32"]

    for i, (model, color) in enumerate(zip(models, colors)):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, data[i], width, label=model, color=color, alpha=0.85)
        # Bold the on-diagonal ones
        for j, bar in enumerate(bars):
            if i < 3 and i == j:  # on-diagonal
                bar.set_edgecolor("black")
                bar.set_linewidth(2)

    # CKF reference line
    for j, val in enumerate(ckf):
        ax.hlines(val, x[j] - 0.45, x[j] + 0.45,
                  colors="black", linestyles="--", linewidth=1.5,
                  label="CKF" if j == 0 else None)

    ax.set_ylabel(r"$d_0$ resolution (IQR/1.349) [mm]")
    ax.set_xlabel("Evaluation dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(eval_datasets)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.set_ylim(0, 0.075)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title(r"$d_0$ resolution across training/evaluation combinations")

    plt.tight_layout()
    out_path = output_dir / "d0_bar_chart.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
