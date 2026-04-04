"""
HEP-style track reconstruction performance plots.

Standard format:
  - Residual histograms with Gaussian core fit, ML vs KF overlaid
  - Resolution vs eta/pT profiles
  - All with ratio panels underneath
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit
from scipy.stats import iqr


PARAM_LABELS = {
    "d0": (r"$d_0$", "mm"),
    "z0": (r"$z_0$", "mm"),
    "phi": (r"$\phi$", "rad"),
    "theta": (r"$\theta$", "rad"),
    "qop": (r"$q/p$", "1/GeV"),
}

COLORS = {
    "ml": "#2196F3",
    "kf": "#F44336",
}


def _gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _fit_gaussian_core(residuals, n_sigma=2.0):
    """Fit a Gaussian to the core of a residual distribution."""
    med = np.median(residuals)
    width = iqr(residuals) / 1.349  # robust sigma estimate
    mask = np.abs(residuals - med) < n_sigma * width
    core = residuals[mask]
    if len(core) < 10:
        return med, width
    return np.mean(core), np.std(core)


def _auto_ratio_ylim(ax, ratio_values, pad=0.2):
    """Set ratio panel y-limits to capture most points with some padding."""
    valid = ratio_values[np.isfinite(ratio_values)]
    if len(valid) == 0:
        ax.set_ylim(0, 3)
        return
    lo = max(0, np.percentile(valid, 2) * (1 - pad))
    hi = np.percentile(valid, 98) * (1 + pad)
    hi = max(hi, 1.5)  # always show at least up to 1.5
    ax.set_ylim(lo, hi)


def _bin_resolution(values, residuals, bin_edges):
    """Compute resolution (std of residual) in bins, with uncertainties.

    Error on std estimate: sigma / sqrt(2*(N-1)) for Gaussian data.
    """
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    half_widths = 0.5 * np.diff(bin_edges)
    resolutions = np.full(len(centers), np.nan)
    res_errors = np.full(len(centers), np.nan)
    counts = np.zeros(len(centers), dtype=int)
    for i in range(len(centers)):
        mask = (values >= bin_edges[i]) & (values < bin_edges[i + 1])
        n = mask.sum()
        counts[i] = n
        if n > 20:
            _, sigma = _fit_gaussian_core(residuals[mask])
            resolutions[i] = sigma
            res_errors[i] = sigma / np.sqrt(2 * (n - 1))
    return centers, half_widths, resolutions, res_errors, counts


def plot_residual_histogram(ml_residuals, kf_residuals, param_name,
                            nbins=100, range_sigma=5.0, figsize=(7, 8)):
    """Residual histogram with Gaussian core fit and ratio panel.

    Args:
        ml_residuals: (N,) array of (pred - truth) for ML
        kf_residuals: (N,) array of (reco - truth) for KF
        param_name: one of d0, z0, phi, theta, qop
    """
    label, unit = PARAM_LABELS[param_name]

    # Determine range from KF (usually tighter)
    _, kf_sigma = _fit_gaussian_core(kf_residuals)
    _, ml_sigma = _fit_gaussian_core(ml_residuals)
    plot_sigma = max(kf_sigma, ml_sigma)
    xrange = (-range_sigma * plot_sigma, range_sigma * plot_sigma)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_main = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)

    bin_edges = np.linspace(xrange[0], xrange[1], nbins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Histograms
    kf_counts, _ = np.histogram(kf_residuals, bins=bin_edges)
    ml_counts, _ = np.histogram(ml_residuals, bins=bin_edges)

    # Normalize to unit area
    kf_norm = kf_counts / (kf_counts.sum() * np.diff(bin_edges)[0])
    ml_norm = ml_counts / (ml_counts.sum() * np.diff(bin_edges)[0])

    ax_main.step(bin_centers, kf_norm, where="mid", color=COLORS["kf"], linewidth=1.5,
                 label=f"CKF ($\\sigma$={kf_sigma:.4g} {unit})")
    ax_main.step(bin_centers, ml_norm, where="mid", color=COLORS["ml"], linewidth=1.5,
                 label=f"ML ($\\sigma$={ml_sigma:.4g} {unit})")

    # Gaussian fits
    try:
        x_fine = np.linspace(xrange[0], xrange[1], 300)
        kf_mu, kf_sig = _fit_gaussian_core(kf_residuals)
        ml_mu, ml_sig = _fit_gaussian_core(ml_residuals)
        kf_gauss = _gaussian(x_fine, 1 / (kf_sig * np.sqrt(2 * np.pi)), kf_mu, kf_sig)
        ml_gauss = _gaussian(x_fine, 1 / (ml_sig * np.sqrt(2 * np.pi)), ml_mu, ml_sig)
        ax_main.plot(x_fine, kf_gauss, "--", color=COLORS["kf"], alpha=0.6, linewidth=1)
        ax_main.plot(x_fine, ml_gauss, "--", color=COLORS["ml"], alpha=0.6, linewidth=1)
    except Exception:
        pass

    ax_main.set_ylabel("Normalized entries")
    ax_main.set_title(f"{label} residual (reco/pred $-$ truth)")
    ax_main.legend(fontsize=10)
    ax_main.set_xlim(xrange)
    ax_main.tick_params(labelbottom=False)

    # Ratio panel: ML / KF
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(kf_norm > 0, ml_norm / kf_norm, np.nan)
    ax_ratio.step(bin_centers, ratio, where="mid", color="black", linewidth=1)
    ax_ratio.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax_ratio.set_ylabel("ML / CKF")
    ax_ratio.set_xlabel(f"{label} residual [{unit}]")
    ax_ratio.set_ylim(0, 3)

    plt.tight_layout()
    return fig


def plot_resolution_vs_eta(ml_residuals, kf_residuals, truth_theta,
                           param_name, n_bins=25, eta_range=(-4, 4),
                           figsize=(8, 7)):
    """Resolution vs pseudorapidity with error bars and ratio panel."""
    label, unit = PARAM_LABELS[param_name]

    eta = -np.log(np.tan(truth_theta / 2 + 1e-10))
    eta_edges = np.linspace(eta_range[0], eta_range[1], n_bins + 1)

    kf_c, kf_hw, kf_res, kf_err, kf_counts = _bin_resolution(eta, kf_residuals, eta_edges)
    ml_c, ml_hw, ml_res, ml_err, _ = _bin_resolution(eta, ml_residuals, eta_edges)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_main = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)

    valid = ~np.isnan(kf_res) & ~np.isnan(ml_res)
    ax_main.errorbar(kf_c[valid], kf_res[valid], xerr=kf_hw[valid], yerr=kf_err[valid],
                     fmt="o", color=COLORS["kf"], markersize=4, linewidth=1, capsize=2, label="CKF")
    ax_main.errorbar(ml_c[valid], ml_res[valid], xerr=ml_hw[valid], yerr=ml_err[valid],
                     fmt="s", color=COLORS["ml"], markersize=4, linewidth=1, capsize=2, label="ML")

    ax_main.set_ylabel(f"{label} resolution ($\\sigma$) [{unit}]")
    ax_main.set_title(f"{label} resolution vs $\\eta$")
    ax_main.set_yscale("log")
    ax_main.legend()
    ax_main.grid(True, alpha=0.3, which="both")
    ax_main.tick_params(labelbottom=False)

    # Ratio with error propagation: ratio = ml/kf, err = ratio * sqrt((ml_err/ml)^2 + (kf_err/kf)^2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(kf_res > 0, ml_res / kf_res, np.nan)
        ratio_err = np.where(
            (kf_res > 0) & (ml_res > 0),
            ratio * np.sqrt((ml_err / ml_res)**2 + (kf_err / kf_res)**2),
            np.nan,
        )
    ax_ratio.errorbar(kf_c[valid], ratio[valid], xerr=kf_hw[valid], yerr=ratio_err[valid],
                      fmt="ko", markersize=4, linewidth=1, capsize=2)
    ax_ratio.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax_ratio.set_ylabel("ML / CKF")
    ax_ratio.set_xlabel(r"$\eta$")
    _auto_ratio_ylim(ax_ratio, ratio[valid])
    ax_ratio.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_resolution_vs_pt(ml_residuals, kf_residuals, truth_qop, truth_theta,
                          param_name, n_bins=20, pt_range=(0.5, 200),
                          figsize=(8, 7)):
    """Resolution vs pT with error bars and ratio panel."""
    label, unit = PARAM_LABELS[param_name]

    p = np.abs(1.0 / (truth_qop + 1e-10))
    pt = p * np.sin(truth_theta)
    pt_edges = np.logspace(np.log10(pt_range[0]), np.log10(pt_range[1]), n_bins + 1)

    kf_c, kf_hw, kf_res, kf_err, _ = _bin_resolution(pt, kf_residuals, pt_edges)
    ml_c, ml_hw, ml_res, ml_err, _ = _bin_resolution(pt, ml_residuals, pt_edges)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_main = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)

    valid = ~np.isnan(kf_res) & ~np.isnan(ml_res)
    ax_main.errorbar(kf_c[valid], kf_res[valid], xerr=kf_hw[valid], yerr=kf_err[valid],
                     fmt="o", color=COLORS["kf"], markersize=4, linewidth=1, capsize=2, label="CKF")
    ax_main.errorbar(ml_c[valid], ml_res[valid], xerr=ml_hw[valid], yerr=ml_err[valid],
                     fmt="s", color=COLORS["ml"], markersize=4, linewidth=1, capsize=2, label="ML")

    ax_main.set_ylabel(f"{label} resolution ($\\sigma$) [{unit}]")
    ax_main.set_title(f"{label} resolution vs $p_T$")
    ax_main.set_xscale("log")
    ax_main.set_yscale("log")
    ax_main.legend()
    ax_main.grid(True, alpha=0.3, which="both")
    ax_main.tick_params(labelbottom=False)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(kf_res > 0, ml_res / kf_res, np.nan)
        ratio_err = np.where(
            (kf_res > 0) & (ml_res > 0),
            ratio * np.sqrt((ml_err / ml_res)**2 + (kf_err / kf_res)**2),
            np.nan,
        )
    ax_ratio.errorbar(kf_c[valid], ratio[valid], xerr=kf_hw[valid], yerr=ratio_err[valid],
                      fmt="ko", markersize=4, linewidth=1, capsize=2)
    ax_ratio.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax_ratio.set_ylabel("ML / CKF")
    ax_ratio.set_xlabel(r"$p_T$ [GeV]")
    _auto_ratio_ylim(ax_ratio, ratio[valid])
    ax_ratio.set_xscale("log")
    ax_ratio.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_pull_distribution(residuals, errors, param_name, nbins=80, figsize=(6, 5)):
    """Pull distribution (residual / error). Should be unit Gaussian if errors are correct."""
    label, unit = PARAM_LABELS[param_name]
    pulls = residuals / (errors + 1e-10)

    # Clip extreme pulls for plotting
    pulls = pulls[np.abs(pulls) < 10]

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(pulls, bins=nbins, range=(-5, 5), density=True, alpha=0.7, color=COLORS["ml"],
            label=f"$\\mu$={np.mean(pulls):.3f}, $\\sigma$={np.std(pulls):.3f}")

    x = np.linspace(-5, 5, 200)
    ax.plot(x, _gaussian(x, 1 / np.sqrt(2 * np.pi), 0, 1), "k--", linewidth=1, label="Unit Gaussian")

    ax.set_xlabel(f"{label} pull")
    ax.set_ylabel("Normalized entries")
    ax.set_title(f"{label} pull distribution")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_summary_table(results_dict, param_names=None, figsize=(10, 4)):
    """Summary table of resolutions across datasets/models.

    Args:
        results_dict: {dataset_name: {"ml_res": {param: float}, "kf_res": {param: float}}}
    """
    if param_names is None:
        param_names = list(PARAM_LABELS.keys())

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    headers = ["Dataset"] + [f"{PARAM_LABELS[p][0]} ML" for p in param_names] + \
              [f"{PARAM_LABELS[p][0]} CKF" for p in param_names]

    rows = []
    for ds_name, res in results_dict.items():
        row = [ds_name]
        for p in param_names:
            row.append(f"{res['ml_res'].get(p, float('nan')):.4g}")
        for p in param_names:
            row.append(f"{res['kf_res'].get(p, float('nan')):.4g}")
        rows.append(row)

    table = ax.table(cellText=rows, colLabels=headers, loc="center",
                     cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    plt.title("Track Parameter Resolution Summary", fontsize=12, pad=20)
    plt.tight_layout()
    return fig


def make_all_residual_plots(ml_pred_raw, kf_reco_raw, truth_raw, output_dir=None):
    """Generate all standard residual histogram plots.

    Args:
        ml_pred_raw: (N, 5) [d0, z0, phi, theta, qop] — ML predictions in physical units
        kf_reco_raw: (N, 5) — CKF reconstructed in physical units
        truth_raw: (N, 5) — truth in physical units
        output_dir: if set, saves figures as PDF

    Returns:
        dict of {param_name: fig}
    """
    figs = {}
    param_names = ["d0", "z0", "phi", "theta", "qop"]

    for i, name in enumerate(param_names):
        ml_res = ml_pred_raw[:, i] - truth_raw[:, i]
        kf_res = kf_reco_raw[:, i] - truth_raw[:, i]

        # Residual histogram
        fig = plot_residual_histogram(ml_res, kf_res, name)
        figs[f"{name}_residual"] = fig
        if output_dir:
            fig.savefig(f"{output_dir}/{name}_residual.pdf", bbox_inches="tight")

        # Resolution vs eta
        fig = plot_resolution_vs_eta(ml_res, kf_res, truth_raw[:, 3], name)
        figs[f"{name}_vs_eta"] = fig
        if output_dir:
            fig.savefig(f"{output_dir}/{name}_resolution_vs_eta.pdf", bbox_inches="tight")

        # Resolution vs pT
        fig = plot_resolution_vs_pt(ml_res, kf_res, truth_raw[:, 4], truth_raw[:, 3], name)
        figs[f"{name}_vs_pt"] = fig
        if output_dir:
            fig.savefig(f"{output_dir}/{name}_resolution_vs_pt.pdf", bbox_inches="tight")

    return figs
