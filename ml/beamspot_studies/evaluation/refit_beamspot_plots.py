#!/usr/bin/env python3
"""Beamspot-refit paper figures: d0/z0 residuals + resolution vs pT/eta.

Reads the per-run tracksummary ROOT files produced by
ml/beamspot_studies/refit/run_ckf_refit.py (configs: none / corrected / doga),
row-concatenates them (never merges histograms — that corrupted widths in the
original study), keeps all majority-matched tracks (ACTS ResPlotTool convention),
and draws:

  fig1  d0 residuals: unconstrained vs beamspot-constrained vs naive predict-0
  fig2  z0 residuals; naive is ~55 mm wide -> inset
  fig3  d0 resolution vs pT and vs eta (-3..3), three configs (two figures)
  fig4  cross-check: corrected vs original-study ('doga') constraint matrix

The shared look (ODD label, error-bar-line style, colours, Gaussian width) lives
in odd_plot_style.py — reuse that module for other ODD studies. Series are told
apart by COLOUR only and drawn as error-bar lines (no markers).

Estimator note: the "resolution width" is a single-Gaussian fit width (iterative
+-3 sigma window), matching ACTS ResPlotTool `reswidth_*` — NOT the RMS and NOT an
aggressive clipped core. See odd_plot_style.gauss_width.

The "naive" curve is the trivial predictor d0 = z0 = 0 (assume every track comes
from the beamspot centre): residual = -t_d0, from the SAME matched-track sample as
the fits. Its width is the beamspot size (sigma_xy = 12.5 um, sigma_z = 55.5 mm).

Run (login node): conda run -p /pscratch/sd/d/danieltm/envs/hep4m2 \
    python refit_beamspot_plots.py --prod-dir /pscratch/sd/d/danieltm/beamspot_refit/out/prod
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import uproot
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from odd_plot_style import (density_hist, draw_hist, draw_series, gauss_width,  # noqa: E402
                            iqr_sigma, legend_lines, odd_label, paper_style)

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


# ---------- figures ----------

def fig_residual(dfs, axis, fname, figures_dir, summary, pt_min=None):
    """axis 'd0'/'z0'. Single residual panel; all series share identical binning.
    Widths = ACTS ResPlotTool convention (see odd_plot_style.gauss_width)."""
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
    draw_hist(ax, r_none, bins, COLOR["none"])
    draw_hist(ax, r_corr, bins, COLOR["corrected"])
    keys = ["none", "corrected"]
    labels = [rf"{LABEL['none']}  ($\sigma$ = {w_none:.1f} $\mu$m)",
              rf"{LABEL['corrected']}  ($\sigma$ = {w_corr:.1f} $\mu$m)"]
    if axis == "d0":
        draw_hist(ax, naive_um, bins, COLOR["naive"])
        keys.append("naive")
        labels.append(rf"{LABEL['naive']}  ($\sigma$ = {w_naive:.1f} $\mu$m)")
    else:
        axins = ax.inset_axes([0.65, 0.15, 0.32, 0.30])
        bmm = np.linspace(-200, 200, 51)
        c2, d2, de2, xe2 = density_hist(naive_mm, bmm)
        draw_series(axins, c2, d2, xe2, COLOR["naive"], yerr=de2, elinewidth=1.2)
        axins.set_ylim(top=axins.get_ylim()[1] * 1.45)  # headroom for the sigma text
        axins.set_xlabel(r"naive $z_0$ res. [mm]", fontsize=8)
        axins.tick_params(labelsize=7)
        axins.set_yticks([])
        axins.text(0.05, 0.87, rf"$\sigma$ = {gauss_width(naive_mm)[0]:.0f} mm",
                   transform=axins.transAxes, fontsize=8.5, color=COLOR["naive"])

    sym = r"d_0" if axis == "d0" else r"z_0"
    ax.set_xlabel(rf"${sym}$ residual (fit $-$ truth) [$\mu$m]")
    ax.set_ylabel("Normalised tracks")
    ax.set_xlim(-lim, lim)
    # extra top headroom so the label + legend stack sits above the peak
    ax.set_ylim(0, ax.get_ylim()[1] * 2.0)
    odd_label(ax, extra=(rf"$p_T > {pt_min:g}$ GeV" if pt_min else None))
    # legend (colour lines) stacked just below the ODD label, clear of all points
    legend_lines(ax, [COLOR[k] for k in keys], labels,
                 loc="upper left", bbox_to_anchor=(0.05, 0.70))
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
    """d0 resolution width vs pT and vs eta, drawn as TWO separate figures, each a
    width panel + ratio-to-no-beamspot panel with its own legend and ODD label."""
    specs = [
        ("t_pT", np.array([1, 1.5, 2, 3, 5, 8, 13, 21, 35, 60, 100]),
         r"$p_T$ [GeV]", "pt", True),
        ("t_eta", np.linspace(-3.0, 3.0, 25), r"$\eta$", "eta", False),
    ]
    improvement = {}
    for var, edges, xlabel, tag, logx in specs:
        fig, (ax, axr) = plt.subplots(
            2, 1, figsize=(6.4, 5.4), sharex=True,
            gridspec_kw=dict(height_ratios=[3, 1], hspace=0.06))
        widths = {}
        for key in ["none", "corrected", "naive"]:
            src = dfs["none"] if key == "naive" else dfs[key]
            x = src[var].to_numpy()
            y = (-src["t_d0"].to_numpy() if key == "naive"
                 else src["res_eLOC0_fit"].to_numpy()) * 1e3
            cen, xerr, w, we = _profile(x, y, edges)
            draw_series(ax, cen, w, xerr, COLOR[key], yerr=we)
            widths[key] = (cen, xerr, w, we)
        # ratio-to-no-beamspot panel
        n_cen, _, n_w, n_we = widths["none"]
        for key in ["corrected", "naive"]:
            cen, xerr, w, we = widths[key]
            common = np.intersect1d(cen, n_cen)
            idx = np.isin(cen, common); nidx = np.isin(n_cen, common)
            rr = w[idx] / n_w[nidx]
            rre = rr * np.sqrt((we[idx] / w[idx]) ** 2 + (n_we[nidx] / n_w[nidx]) ** 2)
            draw_series(axr, cen[idx], rr, xerr[:, idx], COLOR[key], yerr=rre)
            if key == "corrected" and var == "t_pT":
                improvement = {f"pt_{v:g}GeV": float(1 - rv)
                               for v, rv in zip(cen[idx], rr)}
        axr.axhline(1.0, color="0.4", lw=1.0, ls=(0, (4, 3)))
        axr.set_ylim(0, 1.35)
        axr.set_xlabel(xlabel)
        ax.set_ylabel(r"$d_0$ resolution width [$\mu$m]")
        axr.set_ylabel("ratio to\nno beamspot", fontsize=10)
        ax.set_yscale("log")
        ax.set_ylim(top=ax.get_ylim()[1] * 1.9)  # headroom for the legend
        if logx:
            for a in (ax, axr):
                a.set_xscale("log")
            axr.set_xticks([1, 2, 5, 10, 20, 50, 100])
            axr.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            axr.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        else:
            ax.set_xlim(-3, 3)
        odd_label(ax)
        legend_lines(ax, [COLOR[k] for k in ("none", "corrected", "naive")],
                     [LABEL["none"], LABEL["corrected"], LABEL["naive"]],
                     loc="upper right", fontsize=10)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(figures_dir / f"refit_d0_resolution_vs_{tag}.{ext}")
        plt.close(fig)
    summary["d0_improvement_vs_naive_frac_by_pt"] = improvement


def fig_xcheck(dfs, figures_dir, summary):
    """Corrected vs original ('doga') constraint matrix; error-bar lines,
    identical binning, single panel per parameter."""
    xlabel = {"corrected": r"beamspot $\sigma^2$ (correct)",
              "doga": r"$\sigma$ as covariance (bug)"}
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))
    for col, (res_branch, sym, lim) in enumerate(
            [("res_eLOC0_fit", "d_0", 100), ("res_eLOC1_fit", "z_0", 300)]):
        ax = axes[col]
        bins = np.linspace(-lim, lim, 51)
        keys, labels = ["corrected", "doga"], []
        for key in keys:
            r = dfs[key][res_branch].to_numpy() * 1e3
            c, d, de, xerr = density_hist(r, bins)
            draw_series(ax, c, d, xerr, COLOR[key], yerr=de)
            labels.append(rf"{xlabel[key]}  ($\sigma$ = {gauss_width(r)[0]:.1f} $\mu$m)")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.28)
        ax.set_xlim(-lim, lim)
        legend_lines(ax, [COLOR[k] for k in keys], labels, loc="upper right",
                     bbox_to_anchor=(0.97, 0.99), fontsize=8.5)
        ax.set_xlabel(rf"${sym}$ residual [$\mu$m]")
    axes[0].set_ylabel("Normalised tracks")
    fig.suptitle(r"ODD Simulation,  $t\bar{t}$,  $\sqrt{s}$ = 14 TeV,  $\langle\mu\rangle$ = 200   "
                 "— constraint-matrix cross-check", fontsize=11, y=0.98)
    # Pulls are unitless (sigma ~ 1): the default um-scale book_range (+-500) would
    # dump everything into two central bins and the fit would fall back to the IQR
    # seed. Book +-10 pull units so the iterative Gaussian fit actually runs.
    summary["pull_widths"] = {
        key: {"pull_d0": float(gauss_width(dfs[key]["pull_eLOC0_fit"].to_numpy(),
                                           book_range=10.0)[0]),
              "pull_z0": float(gauss_width(dfs[key]["pull_eLOC1_fit"].to_numpy(),
                                           book_range=10.0)[0])}
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
        print(f"{cfg}: {len(df)} matched tracks from {df['run'].nunique()} runs")

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
