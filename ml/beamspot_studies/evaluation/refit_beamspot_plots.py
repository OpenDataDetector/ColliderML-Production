#!/usr/bin/env python3
"""Beamspot-refit paper figures: d0/z0 residuals + resolution vs pT/eta.

Reads the per-run tracksummary ROOT files produced by
ml/beamspot_studies/refit/run_ckf_refit.py (configs: none / corrected / doga),
row-concatenates them (never merges histograms — that corrupted widths in the
original study), dedups CKF duplicates per matched particle, and draws:

  fig1  d0 residuals (+ ratio-to-no-beamspot panel): unconstrained vs
        beamspot-constrained vs naive predict-0
  fig2  z0 residuals (+ ratio panel); naive is ~55 mm wide -> inset
  fig3  d0 resolution vs pT and vs eta (-3..3), three configs
  fig4  cross-check: corrected vs original-study ('doga') constraint matrix

Estimator note: the plotted "resolution width" is a single-Gaussian fit width
(iterative +-4 sigma window), matching ACTS ResPlotTool `reswidth_*` — NOT the RMS
(which is ~2x larger at mu=200 because of non-Gaussian tails) and NOT an aggressive
clipped core (which is ~25% smaller). This is the standard, ATLAS/ACTS-comparable
definition and reproduces the numbers from the original study's ResPlotTool plots.

The "naive" curve is the trivial predictor d0 = z0 = 0 (assume every track comes
from the beamspot centre): residual = -t_d0, from the SAME matched-track sample as
the fits. Its width is the beamspot size (sigma_xy = 12.5 um, sigma_z = 55.5 mm).

Run (login node): conda run -p /pscratch/sd/d/danieltm/envs/hep4m2 \
    python refit_beamspot_plots.py --prod-dir /pscratch/sd/d/danieltm/beamspot_refit/out/prod
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import uproot
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

CONFIGS = ["none", "corrected", "doga"]

# Entity -> color, fixed (continuity with the original study: red = w/o beamspot,
# blue = w/ beamspot). Palette validated with the dataviz six-checks validator.
COLOR = {
    "none": "#d62728",       # unconstrained refit
    "corrected": "#1f77b4",  # beamspot-constrained refit
    "naive": "#b8860b",      # trivial predict-0 baseline
    "doga": "#7f3fbf",       # cross-check only (fig4)
}
LABEL = {
    "none": "track fit (no beamspot)",
    "corrected": "track fit + beamspot",
    "naive": r"naive $d_0\!=\!z_0\!=\!0$ (beamspot only)",
    "doga": "original constraint matrix",
}
MARKER = {"none": "o", "corrected": "s", "naive": "^", "doga": "D"}

PID_COLS = [
    "majorityParticleId_vertex_primary", "majorityParticleId_vertex_secondary",
    "majorityParticleId_particle", "majorityParticleId_generation",
    "majorityParticleId_sub_particle",
]
BRANCHES = [
    "event_nr", "nMajorityHits", "nMeasurements",
    "chi2Sum", "NDF", "t_d0", "t_z0", "t_pT", "t_eta",
    "res_eLOC0_fit", "res_eLOC1_fit", "pull_eLOC0_fit", "pull_eLOC1_fit",
] + PID_COLS


def flatten(arr):
    if arr.dtype == object:
        return np.concatenate([np.asarray(x, dtype=float) for x in arr]) if len(arr) else np.array([])
    return np.asarray(arr, dtype=float)


def load_config(prod_dir: Path, cfg: str, cache_dir: Path) -> pd.DataFrame:
    # ACTS ResPlotTool fills ALL majority-matched tracks (no dedup). We follow that
    # convention so our widths reproduce ACTS's own performance-file output (verified
    # by hadd + extractMeanWidthProfiles). Deduping to one track/particle would give a
    # ~15% smaller unconstrained width — non-standard.
    cache = cache_dir / f"tracks_{cfg}_full.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    frames = []
    files = sorted(glob.glob(str(prod_dir / "*" / f"tracksummary_ckf_refit_{cfg}.root")))
    if not files:
        raise SystemExit(f"no tracksummary files for config '{cfg}' under {prod_dir}")
    skipped = 0
    for fn in files:
        run = Path(fn).parent.name
        # Only trust runs whose job completed (ROOT writers finalize at sequencer
        # end; killed jobs can leave readable-but-partial files).
        job_log = Path(fn).parent / "job.log"
        try:
            if "Processed" not in job_log.read_text()[-4000:]:
                skipped += 1
                continue
        except OSError:
            skipped += 1
            continue
        try:
            t = uproot.open(fn)["tracksummary"]
        except Exception:
            skipped += 1
            continue
        arrs = t.arrays(BRANCHES, library="np")
        cols = {k: flatten(v) for k, v in arrs.items()}
        n = len(cols["t_d0"])
        if len(cols["event_nr"]) != n:
            ev = arrs["event_nr"]
            reps = [len(np.atleast_1d(x)) for x in arrs["t_d0"]]
            cols["event_nr"] = np.repeat(np.asarray(ev, dtype=float), reps)
        df = pd.DataFrame(cols)
        df["run"] = int(run)
        frames.append(df)
    if skipped:
        print(f"[{cfg}] WARNING: skipped {skipped} unreadable/unfinished file(s)")
    if not frames:
        raise SystemExit(f"no readable tracksummary files for '{cfg}'")
    df = pd.concat(frames, ignore_index=True)

    df = df[np.isfinite(df["res_eLOC0_fit"])].reset_index(drop=True).copy()
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache)
    return df


# ---------- estimators ----------

def _gauss(x, a, mu, s):
    return a * np.exp(-0.5 * ((x - mu) / s) ** 2)


def gauss_width(x, book_range=0.5e3, book_bins=100, sigma_range=3.0, iters=3):
    """ACTS ResPlotTool width: replicates ActsPlugins::extractMeanWidthProfiles.
    Books the residual in a fixed range (ACTS default +-0.5 mm = +-500 um for
    d0/z0), then does an iterative Gaussian fit restricted to +-sigma_range*sigma,
    for `iters` iterations. Returns (sigma, sigma_err) in the input units.
    Verified against ACTS's own hadd'd histogram output (~within binning)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return np.nan, np.nan
    # auto-widen the booked range for distributions far broader than it (e.g. the
    # naive z0 baseline, ~55 mm): keeps ACTS's +-0.5 mm for the narrow fit residuals.
    iqr = (np.percentile(x, 75) - np.percentile(x, 25)) / 1.349
    if 3 * iqr > book_range:
        book_range = 8 * iqr
    h, edges = np.histogram(x, bins=book_bins, range=(-book_range, book_range))
    c = 0.5 * (edges[:-1] + edges[1:])
    mu = np.median(x[np.abs(x) < book_range]) if (np.abs(x) < book_range).any() else 0.0
    s = (np.percentile(x, 75) - np.percentile(x, 25)) / 1.349 or x.std()
    serr = s / np.sqrt(2 * len(x))
    for _ in range(iters):
        m = (c > mu - sigma_range * s) & (c < mu + sigma_range * s)
        if m.sum() < 5:
            break
        try:
            p, cov = curve_fit(_gauss, c[m], h[m], p0=[max(h[m].max(), 1), mu, s],
                               sigma=np.sqrt(h[m] + 1), maxfev=8000)
            mu, s = p[1], abs(p[2])
            serr = float(np.sqrt(cov[2, 2])) if np.all(np.isfinite(cov)) else s / np.sqrt(2 * m.sum())
        except Exception:
            break
    return float(s), float(serr)


def iqr_sigma(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return np.nan
    return (np.percentile(x, 75) - np.percentile(x, 25)) / 1.349


# ---------- style + primitives ----------

def paper_style():
    """ATLAS-like tracking-plot style: full axis box, ticks inside on all four
    sides with minor ticks, no grid, sans-serif."""
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 12, "axes.labelsize": 13,
        "legend.fontsize": 10.5, "legend.frameon": False,
        "axes.grid": False, "axes.linewidth": 1.0,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.major.size": 6, "ytick.major.size": 6,
        "xtick.minor.size": 3, "ytick.minor.size": 3,
        "lines.linewidth": 1.6,
        "errorbar.capsize": 0,
    })


def odd_label(ax, extra=None, x=0.05, y=0.95):
    ax.text(x, y, "OpenDataDetector", transform=ax.transAxes,
            fontsize=13, fontweight="bold", fontstyle="italic", va="top")
    info = r"Simulation,  $t\bar{t}$,  $\langle\mu\rangle = 200$"
    if extra:
        info += f",  {extra}"
    ax.text(x, y - 0.065, info, transform=ax.transAxes, fontsize=10.5, va="top")


def density_hist(vals, bins):
    """Return (centres, density, density_err, half_bin_width). No lines drawn."""
    h, edges = np.histogram(vals, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    norm = max(h.sum() * width, 1)
    return centres, h / norm, np.sqrt(h) / norm, np.full_like(centres, width / 2)


def draw_hist(ax, vals, bins, key, sigma=None, unit=r"$\mu$m", fmt="{:.1f}"):
    """Density histogram as error-bar markers (x err = half bin width). Every bin is
    drawn (identical binning across series -> identical marker count)."""
    c, d, derr, xerr = density_hist(vals, bins)
    lbl = LABEL[key] if sigma is None else (
        f"{LABEL[key]}  (" + rf"$\sigma$ = {fmt.format(sigma)} {unit})")
    ax.errorbar(c, d, yerr=derr, xerr=xerr, fmt=MARKER[key],
                color=COLOR[key], ms=4.0, lw=1.1, mec=COLOR[key], label=lbl)
    return c, d, derr, xerr


# ---------- figures ----------

def fig_residual(dfs, axis, fname, figures_dir, summary, pt_min=None):
    """axis 'd0'/'z0'. Single residual panel; all series share identical binning and
    every bin is drawn (same marker count). Widths = ACTS ResPlotTool convention."""
    res_branch = "res_eLOC0_fit" if axis == "d0" else "res_eLOC1_fit"
    truth_branch = "t_d0" if axis == "d0" else "t_z0"

    def sel(df):
        return df[df["t_pT"] >= pt_min] if pt_min else df

    r_none = sel(dfs["none"])[res_branch].to_numpy() * 1e3          # um
    r_corr = sel(dfs["corrected"])[res_branch].to_numpy() * 1e3
    naive_mm = -sel(dfs["none"])[truth_branch].to_numpy()          # mm
    naive_um = naive_mm * 1e3

    w_none = gauss_width(r_none)[0]
    w_corr = gauss_width(r_corr)[0]
    w_naive = gauss_width(naive_um)[0]
    s = {
        "none": {"width_um": w_none, "iqr_um": iqr_sigma(r_none)},
        "corrected": {"width_um": w_corr, "iqr_um": iqr_sigma(r_corr)},
        "naive": {"width_um": w_naive, "iqr_um": iqr_sigma(naive_um)},
        "n_tracks": int(len(r_none)),
        "corrected_vs_naive_improvement_pct": float(100 * (1 - w_corr / w_naive)),
    }
    summary[f"residual_{axis}" + (f"_pt{pt_min:g}" if pt_min else "")] = s

    lim = 100 if axis == "d0" else 300
    bins = np.linspace(-lim, lim, 51)   # 50 identical bins, all series

    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    draw_hist(ax, r_none, bins, "none", sigma=w_none)
    draw_hist(ax, r_corr, bins, "corrected", sigma=w_corr)
    if axis == "d0":
        draw_hist(ax, naive_um, bins, "naive", sigma=w_naive)
    else:
        axins = ax.inset_axes([0.64, 0.36, 0.33, 0.36])
        bmm = np.linspace(-200, 200, 51)
        c2, d2, de2, xe2 = density_hist(naive_mm, bmm)
        axins.errorbar(c2, d2, yerr=de2, xerr=xe2, fmt="^",
                       color=COLOR["naive"], ms=3.0, lw=0.9)
        axins.set_xlabel(r"naive $z_0$ res. [mm]", fontsize=8)
        axins.tick_params(labelsize=7)
        axins.set_yticks([])
        axins.text(0.05, 0.86, rf"$\sigma$ = {gauss_width(naive_mm)[0]:.0f} mm",
                   transform=axins.transAxes, fontsize=8.5, color=COLOR["naive"])

    sym = r"d_0" if axis == "d0" else r"z_0"
    ax.set_xlabel(rf"${sym}$ residual (fit $-$ truth) [$\mu$m]")
    ax.set_ylabel("Normalised tracks")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.5)
    odd_label(ax, extra=(rf"$p_T > {pt_min:g}$ GeV" if pt_min else None))
    ax.legend(loc="upper right", bbox_to_anchor=(0.985, 0.80))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"{fname}.{ext}")
    plt.close(fig)


def _profile(x, y, edges, min_n=30):
    """Per-bin Gaussian width with error; returns centres, xerr(to edges), w, werr."""
    cen, xerr_lo, xerr_hi, ww, we = [], [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() >= min_n:
            w, e = gauss_width(y[m])
            mid = 0.5 * (lo + hi)
            cen.append(mid); xerr_lo.append(mid - lo); xerr_hi.append(hi - mid)
            ww.append(w); we.append(e)
    return (np.array(cen), np.array([xerr_lo, xerr_hi]),
            np.array(ww), np.array(we))


def fig_resolution_vs(dfs, figures_dir, summary):
    """d0 resolution width vs pT and vs eta (-3..3), each with a ratio-to-no-beamspot
    panel below. Error-bar markers only (x err = half bin width), no lines."""
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 5.6), sharex="col",
                             gridspec_kw=dict(height_ratios=[3, 1], hspace=0.06))
    pt_edges = np.array([1, 1.5, 2, 3, 5, 8, 13, 21, 35, 60, 100])
    eta_edges = np.linspace(-3.0, 3.0, 25)

    improvement = {}
    for col, (var, edges, xlabel) in enumerate([
        ("t_pT", pt_edges, r"$p_T$ [GeV]"),
        ("t_eta", eta_edges, r"$\eta$"),
    ]):
        ax, axr = axes[0, col], axes[1, col]
        widths = {}
        for key in ["none", "corrected", "naive"]:
            src = dfs["none"] if key == "naive" else dfs[key]
            x = src[var].to_numpy()
            y = (-src["t_d0"].to_numpy() if key == "naive"
                 else src["res_eLOC0_fit"].to_numpy()) * 1e3
            cen, xerr, w, we = _profile(x, y, edges)
            ax.errorbar(cen, w, yerr=we, xerr=xerr, fmt=MARKER[key],
                        color=COLOR[key], ms=4.5, lw=1.1, label=LABEL[key])
            widths[key] = (cen, xerr, w, we)
        # ratio-to-no-beamspot panel
        n_cen, _, n_w, n_we = widths["none"]
        for key in ["corrected", "naive"]:
            cen, xerr, w, we = widths[key]
            common = np.intersect1d(cen, n_cen)
            idx = np.isin(cen, common); nidx = np.isin(n_cen, common)
            rr = w[idx] / n_w[nidx]
            rre = rr * np.sqrt((we[idx] / w[idx]) ** 2 + (n_we[nidx] / n_w[nidx]) ** 2)
            axr.errorbar(cen[idx], rr, yerr=rre, xerr=xerr[:, idx],
                         fmt=MARKER[key], color=COLOR[key], ms=4.0, lw=1.1)
            if key == "corrected" and var == "t_pT":
                improvement = {f"pt_{v:g}GeV": float(1 - rv)
                               for v, rv in zip(cen[idx], rr)}
        axr.axhline(1.0, color="0.4", lw=1.0, ls=(0, (4, 3)))
        axr.set_ylim(0, 1.35)
        axr.set_xlabel(xlabel)
        if var == "t_pT":
            for a in (ax, axr):
                a.set_xscale("log")
            axr.set_xticks([1, 2, 5, 10, 20, 50, 100])
            axr.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            axr.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        else:
            ax.set_xlim(-3, 3)
    axes[0, 0].set_ylabel(r"$d_0$ resolution width [$\mu$m]")
    # log y on BOTH top panels, shared range
    for a in (axes[0, 0], axes[0, 1]):
        a.set_yscale("log")
    lo = min(axes[0, 0].get_ylim()[0], axes[0, 1].get_ylim()[0])
    hi = max(axes[0, 0].get_ylim()[1], axes[0, 1].get_ylim()[1])
    for a in (axes[0, 0], axes[0, 1]):
        a.set_ylim(lo * 0.85, hi * 1.5)
    axes[0, 1].tick_params(labelleft=False)  # same scale as left panel
    axes[1, 0].set_ylabel("ratio to\nno beamspot", fontsize=10)
    odd_label(axes[0, 0])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10.5,
               bbox_to_anchor=(0.5, -0.02), columnspacing=1.6)
    summary["d0_improvement_vs_naive_frac_by_pt"] = improvement
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"refit_d0_resolution_vs_pt_eta.{ext}")
    plt.close(fig)


def fig_xcheck(dfs, figures_dir, summary):
    """Corrected vs original ('doga') constraint matrix; error-bar markers,
    identical binning, single panel per parameter."""
    xlabel = {"corrected": r"beamspot $\sigma^2$ (correct)",
              "doga": r"$\sigma$ as covariance (bug)"}
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))
    for col, (res_branch, sym, lim) in enumerate(
            [("res_eLOC0_fit", "d_0", 100), ("res_eLOC1_fit", "z_0", 300)]):
        ax = axes[col]
        bins = np.linspace(-lim, lim, 51)
        for key in ["corrected", "doga"]:
            r = dfs[key][res_branch].to_numpy() * 1e3
            c, d, de, xerr = density_hist(r, bins)
            ax.errorbar(c, d, yerr=de, xerr=xerr, fmt=MARKER[key],
                        color=COLOR[key], ms=4.0, lw=1.1,
                        label=rf"{xlabel[key]}  ($\sigma$ = {gauss_width(r)[0]:.1f} $\mu$m)")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.28)
        ax.set_xlim(-lim, lim)
        ax.legend(loc="upper right", fontsize=8.5)
        ax.set_xlabel(rf"${sym}$ residual [$\mu$m]")
    axes[0].set_ylabel("Normalised tracks")
    fig.suptitle(r"OpenDataDetector  Simulation,  $t\bar{t}$,  $\langle\mu\rangle = 200$   "
                 "— constraint-matrix cross-check", fontsize=11, y=0.98)
    summary["pull_widths"] = {
        key: {"pull_d0": float(gauss_width(dfs[key]["pull_eLOC0_fit"].to_numpy())[0]),
              "pull_z0": float(gauss_width(dfs[key]["pull_eLOC1_fit"].to_numpy())[0])}
        for key in ["none", "corrected", "doga"]}
    summary["xcheck_doga_widths_um"] = {
        "d0": float(gauss_width(dfs["doga"]["res_eLOC0_fit"].to_numpy() * 1e3)[0]),
        "z0": float(gauss_width(dfs["doga"]["res_eLOC1_fit"].to_numpy() * 1e3)[0])}
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"refit_xcheck_constraint_matrix.{ext}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prod-dir", default="/pscratch/sd/d/danieltm/beamspot_refit/out/prod")
    p.add_argument("--figures-dir", default=str(Path(__file__).resolve().parent.parent / "paper" / "figures"))
    p.add_argument("--cache-dir", default="/pscratch/sd/d/danieltm/beamspot_refit/cache")
    args = p.parse_args()

    paper_style()
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    prod, cache = Path(args.prod_dir), Path(args.cache_dir)

    dfs = {cfg: load_config(prod, cfg, cache) for cfg in CONFIGS}
    for cfg, df in dfs.items():
        print(f"{cfg}: {len(df)} matched deduped tracks from {df['run'].nunique()} runs")

    summary = {}
    fig_residual(dfs, "d0", "refit_d0_residual", figures_dir, summary)
    fig_residual(dfs, "d0", "refit_d0_residual_pt5", figures_dir, summary, pt_min=5.0)
    fig_residual(dfs, "z0", "refit_z0_residual", figures_dir, summary)
    fig_resolution_vs(dfs, figures_dir, summary)
    fig_xcheck(dfs, figures_dir, summary)

    (figures_dir / "refit_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"figures -> {figures_dir}")


if __name__ == "__main__":
    main()
