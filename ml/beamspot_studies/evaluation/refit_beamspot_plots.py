#!/usr/bin/env python3
"""Beamspot-refit paper figures: d0/z0 residuals + resolution vs pT/|eta|.

Reads the per-run tracksummary ROOT files produced by
ml/beamspot_studies/refit/run_ckf_refit.py (configs: none / corrected / doga),
row-concatenates them (never merges histograms — that corrupted widths in the
original study), dedups CKF duplicates per matched particle, and draws:

  fig1  d0 residuals: unconstrained vs beamspot-constrained vs naive predict-0
  fig2  z0 residuals: same three curves (naive is ~55 mm wide -> inset)
  fig3  d0 resolution vs pT and vs |eta| (3 curves)
  fig4  cross-check (not for paper): corrected vs original-study ('doga')
        constraint matrix, incl. pull widths — quantifies the sigma-vs-sigma^2 bug

The "naive" curve is the trivial predictor d0 = z0 = 0 (i.e. assume every track
comes from the beamspot centre): residual = 0 - t_d0 = -t_d0, histogrammed from
the SAME matched-track sample as the fits. Its width is the beamspot size
(sigma_xy = 12.5 um, sigma_z = 55.5 mm) — the bar any fit must clear.

Run (login node): conda run -p /pscratch/sd/d/danieltm/envs/hep4m2 \
    python refit_beamspot_plots.py --prod-dir /pscratch/sd/d/danieltm/beamspot_refit/out/prod
"""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import uproot
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFIGS = ["none", "corrected", "doga"]

# Entity -> color, fixed (continuity with the original study: red = w/o beamspot,
# blue = w/ beamspot). Palette validated (lightness band, chroma floor, CVD
# separation, contrast) with the dataviz six-checks validator.
COLOR = {
    "none": "#d62728",       # unconstrained refit
    "corrected": "#1f77b4",  # beamspot-constrained refit
    "naive": "#b8860b",      # trivial predict-0 baseline (dashed)
    "doga": "#7f3fbf",       # cross-check only (fig4)
}
LABEL = {
    "none": "track fit (no beamspot)",
    "corrected": "track fit + beamspot",
    "naive": r"naive $d_0\!=\!z_0\!=\!0$ (beamspot only)",
    "doga": "original constraint matrix",
}

# majority particle id is stored as decomposed barcode fields in this writer
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
    cache = cache_dir / f"tracks_{cfg}.pkl"
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
            skipped += 1  # unfinished/unreadable (job may still be writing)
            continue
        arrs = t.arrays(BRANCHES, library="np")
        cols = {k: flatten(v) for k, v in arrs.items()}
        n = len(cols["t_d0"])
        # event_nr is per-event scalar in some layouts; broadcast via repeat if needed
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

    # matched tracks only (writer fills residuals for majority-matched tracks)
    df = df[np.isfinite(df["res_eLOC0_fit"])].copy()
    # dedup CKF duplicates: one track per (run, event, matched particle) — keep the
    # candidate with most measurements, then best chi2/ndf
    df["chi2ndf"] = df["chi2Sum"] / df["NDF"].replace(0, np.nan)
    df = (
        df.sort_values(["nMeasurements", "chi2ndf"], ascending=[False, True])
        .drop_duplicates(subset=["run", "event_nr"] + PID_COLS, keep="first")
        .reset_index(drop=True)
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache)
    return df


def core_sigma(x, clip=2.5, iters=5):
    """Iteratively clipped core Gaussian sigma (numpy only).

    Seeded from the IQR sigma (not the raw std) so heavy non-Gaussian tails —
    e.g. the z0 residuals at mu=200 — cannot drag the clip window outward.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return np.nan
    mu, sig = np.median(x), (np.percentile(x, 75) - np.percentile(x, 25)) / 1.349
    if not np.isfinite(sig) or sig <= 0:
        sig = x.std()
    for _ in range(iters):
        w = x[(x > mu - clip * sig) & (x < mu + clip * sig)]
        if len(w) < 10:
            break
        mu, sig = w.mean(), w.std()
    return sig


def iqr_sigma(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return np.nan
    return (np.percentile(x, 75) - np.percentile(x, 25)) / 1.349


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
        "lines.linewidth": 1.8,
        "errorbar.capsize": 0,
    })


def odd_label(ax, extra=None):
    """Experiment-style label block (top-left inside the axes)."""
    ax.text(0.05, 0.95, "OpenDataDetector", transform=ax.transAxes,
            fontsize=13, fontweight="bold", fontstyle="italic", va="top")
    info = r"Simulation,  $t\bar{t}$,  $\langle\mu\rangle = 200$"
    if extra:
        info += f",  {extra}"
    ax.text(0.05, 0.885, info, transform=ax.transAxes, fontsize=10.5, va="top")


def hist_points(ax, vals, bins, color, label, marker="o", sigma=None,
                unit=r"$\mu$m", fmt="{:.1f}"):
    """ATLAS-style binned points with Poisson error bars (density-normalised)."""
    h, edges = np.histogram(vals, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    norm = max(h.sum() * width, 1)
    m = h > 0
    lbl = label if sigma is None else (
        f"{label}  (" + rf"$\sigma$ = {fmt.format(sigma)} {unit})")
    ax.errorbar(centers[m], h[m] / norm, yerr=np.sqrt(h[m]) / norm,
                fmt=marker, color=color, ms=4.0, lw=1.2, label=lbl)


def hist_line(ax, vals, bins, color, label, sigma=None, unit=r"$\mu$m",
              fmt="{:.1f}"):
    """Dashed step outline — used for the analytic-reference naive curve."""
    h, edges = np.histogram(vals, bins=bins)
    width = edges[1] - edges[0]
    norm = max(h.sum() * width, 1)
    lbl = label if sigma is None else (
        f"{label}  (" + rf"$\sigma$ = {fmt.format(sigma)} {unit})")
    ax.stairs(h / norm, edges, color=color, lw=1.8, ls="--", label=lbl)


def fig_residual(dfs, axis, fname, figures_dir, summary, pt_min=None):
    """axis: 'd0' (res_eLOC0_fit, um) or 'z0' (res_eLOC1_fit, um + naive inset in mm).
    pt_min: optional track-pT selection in GeV (the b-tagging-like regime where the
    tracker genuinely adds information beyond the beamspot prior)."""
    res_branch = "res_eLOC0_fit" if axis == "d0" else "res_eLOC1_fit"
    truth_branch = "t_d0" if axis == "d0" else "t_z0"
    to_um = 1e3  # mm -> um

    def sel(df):
        return df[df["t_pT"] >= pt_min] if pt_min else df

    r_none = sel(dfs["none"])[res_branch].to_numpy() * to_um
    r_corr = sel(dfs["corrected"])[res_branch].to_numpy() * to_um
    naive_mm = -sel(dfs["none"])[truth_branch].to_numpy()
    naive_um = naive_mm * to_um

    s = {
        "none": {"core_um": core_sigma(r_none), "iqr_um": iqr_sigma(r_none)},
        "corrected": {"core_um": core_sigma(r_corr), "iqr_um": iqr_sigma(r_corr)},
        "naive": {"core_um": core_sigma(naive_um), "iqr_um": iqr_sigma(naive_um)},
        "n_tracks": int(len(r_none)),
        "corrected_vs_naive_improvement_pct": float(
            100.0 * (1.0 - core_sigma(r_corr) / core_sigma(naive_um))
        ),
    }
    key = f"residual_{axis}" + (f"_pt{pt_min:g}" if pt_min else "")
    summary[key] = s

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    lim = 120 if axis == "d0" else 350
    bins = np.linspace(-lim, lim, 61)
    hist_points(ax, r_none, bins, COLOR["none"], LABEL["none"], marker="o",
                sigma=s["none"]["core_um"])
    hist_points(ax, r_corr, bins, COLOR["corrected"], LABEL["corrected"], marker="s",
                sigma=s["corrected"]["core_um"])
    if axis == "d0":
        hist_line(ax, naive_um, bins, COLOR["naive"], LABEL["naive"],
                  sigma=s["naive"]["core_um"])
    else:
        # naive z0 is the 55.5 mm beamspot — off-scale; show it in an inset
        axins = ax.inset_axes([0.64, 0.26, 0.33, 0.30])
        bins_mm = np.linspace(-200, 200, 41)
        hist_line(axins, naive_mm, bins_mm, COLOR["naive"], None)
        axins.set_xlabel(r"naive $z_0$ res. [mm]", fontsize=8)
        axins.tick_params(labelsize=7)
        axins.set_yticks([])
        axins.text(0.06, 0.84, rf"$\sigma$ = {core_sigma(naive_mm):.0f} mm",
                   transform=axins.transAxes, fontsize=8.5, color=COLOR["naive"])
    sym = r"d_0" if axis == "d0" else r"z_0"
    ax.set_xlabel(rf"${sym}$ residual (fit $-$ truth) [$\mu$m]")
    ax.set_ylabel("Normalised tracks")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
    odd_label(ax, extra=(rf"$p_T > {pt_min:g}$ GeV" if pt_min else None))
    ax.legend(loc="upper right", bbox_to_anchor=(0.985, 0.80))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"{fname}.{ext}")
    plt.close(fig)


def fig_resolution_vs(dfs, figures_dir, summary):
    """d0 core resolution vs pT and vs |eta| for the three curves, with
    sigma/sqrt(2N) width uncertainties."""
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.3), sharey=True)
    pt_edges = np.array([1, 1.5, 2, 3, 5, 8, 13, 21, 35, 60, 100])
    eta_edges = np.linspace(0, 3.0, 11)

    improvement = {}
    for ax, var, edges, xlabel in [
        (axes[0], "t_pT", pt_edges, r"$p_T$ [GeV]"),
        (axes[1], "abs_eta", eta_edges, r"$|\eta|$"),
    ]:
        curves = {}
        for key, marker, ls in [("none", "o", "-"), ("corrected", "s", "-"),
                                ("naive", None, "--")]:
            src = dfs["none"] if key == "naive" else dfs[key]
            x = np.abs(src["t_eta"].to_numpy()) if var == "abs_eta" else src[var].to_numpy()
            y = (-src["t_d0"].to_numpy() if key == "naive"
                 else src["res_eLOC0_fit"].to_numpy()) * 1e3
            centers, widths, errs = [], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (x >= lo) & (x < hi)
                if m.sum() >= 30:
                    w = core_sigma(y[m])
                    centers.append(0.5 * (lo + hi))
                    widths.append(w)
                    errs.append(w / np.sqrt(2 * m.sum()))
            curves[key] = (np.array(centers), np.array(widths))
            if key == "naive":
                ax.plot(centers, widths, ls="--", color=COLOR[key],
                        label=LABEL[key], lw=1.8)
            else:
                ax.errorbar(centers, widths, yerr=errs, fmt=marker + ls,
                            color=COLOR[key], label=LABEL[key], ms=4.5, lw=1.2)
        if var == "t_pT":
            # constrained-vs-naive improvement per pT bin (for the summary/text)
            c_x, c_w = curves["corrected"]
            n_x, n_w = curves["naive"]
            common = np.intersect1d(c_x, n_x)
            improvement = {
                f"pt_{v:g}GeV": float(100 * (1 - c_w[c_x == v][0] / n_w[n_x == v][0]))
                for v in common
            }
            ax.set_xscale("log")
            ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel(r"$d_0$ core resolution [$\mu$m]")
    axes[0].set_yscale("log")
    ylo, yhi = axes[0].get_ylim()
    axes[0].set_ylim(ylo * 0.75, yhi * 1.8)
    odd_label(axes[0])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10.5,
               bbox_to_anchor=(0.5, -0.005), columnspacing=1.6)
    summary["d0_improvement_vs_naive_pct_by_pt"] = improvement
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"refit_d0_resolution_vs_pt_eta.{ext}")
    plt.close(fig)


def fig_xcheck(dfs, figures_dir, summary):
    """Corrected vs original ('doga') constraint matrix + pull widths."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for ax, res_branch, sym, lim in [
        (axes[0], "res_eLOC0_fit", "d_0", 120),
        (axes[1], "res_eLOC1_fit", "z_0", 350),
    ]:
        bins = np.linspace(-lim, lim, 61)
        for key, marker in [("corrected", "s"), ("doga", "D")]:
            r = dfs[key][res_branch].to_numpy() * 1e3
            hist_points(ax, r, bins, COLOR[key], LABEL[key], marker=marker,
                        sigma=core_sigma(r))
        ax.set_xlabel(rf"${sym}$ residual [$\mu$m]")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
        ax.legend(loc="upper right", fontsize=9)
    axes[0].set_ylabel("normalised tracks")
    pulls = {}
    for key in ["none", "corrected", "doga"]:
        pulls[key] = {
            "pull_d0": float(core_sigma(dfs[key]["pull_eLOC0_fit"])),
            "pull_z0": float(core_sigma(dfs[key]["pull_eLOC1_fit"])),
        }
    summary["pull_widths"] = pulls
    summary["xcheck_doga_residuals"] = {
        "d0_core_um": float(core_sigma(dfs["doga"]["res_eLOC0_fit"].to_numpy() * 1e3)),
        "z0_core_um": float(core_sigma(dfs["doga"]["res_eLOC1_fit"].to_numpy() * 1e3)),
    }
    fig.suptitle("constraint-matrix cross-check (corrected vs original)", fontsize=11)
    fig.tight_layout()
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

    out = figures_dir / "refit_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"figures -> {figures_dir}")


if __name__ == "__main__":
    main()
