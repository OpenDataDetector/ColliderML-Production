#!/usr/bin/env python3
"""Shared ODD paper-plot style — reuse this across ODD tracking / beamspot studies
so every figure has one consistent look.

What it gives you:
  - paper_style()      ATLAS-like house style (full tick box, minor ticks, no grid)
  - ensure_bold_font() self-heal for envs missing the bold DejaVu face (see below)
  - odd_label(ax)      the "ODD Simulation / ACTS <ver> / sample" experiment label
  - draw_series(...)    a series as colour-coded error-bar LINES (no markers)
  - draw_hist(...)      a density histogram in the same error-bar-line style
  - legend_lines(...)   a legend whose handles are plain colour lines
  - gauss_width(...)    ACTS ResPlotTool-style iterative Gaussian width (+ error)
  - iqr_sigma / density_hist

Design conventions baked in (kept deliberately so studies match each other):
  * Series are told apart by COLOUR ONLY — never by marker shape as well.
  * Data are drawn as error-bar lines: a horizontal segment spanning each bin
    (x err = half bin width) plus the y-error bar, with NO point marker.
  * The ODD badge is bold-italic; 'Simulation' upright.

Typical use:
    from odd_plot_style import (paper_style, odd_label, draw_series, draw_hist,
                                legend_lines, gauss_width, iqr_sigma, density_hist,
                                ACTS_VERSION)
    paper_style()
    ...
    draw_series(ax, x, w, xerr, color="#d62728", yerr=we)
    odd_label(ax)
    legend_lines(ax, [c1, c2], ["label 1", "label 2"], loc="upper right")

Font gotcha (why ensure_bold_font exists): some stripped matplotlib installs (the
NERSC hep4m2 conda env is one) ship only the *regular* DejaVu Sans face. With the
bold face missing, matplotlib silently falls back to regular, so fontweight="bold"
renders un-bold with no error. ensure_bold_font() registers the bold faces
vendored in ./fonts/ whenever the environment lacks one, so the label is bold in
any environment. paper_style() calls it for you.
"""
import warnings
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker
from scipy.optimize import curve_fit

# ACTS release baked into the sw:pr-8 container: murnanedaniel/acts fork at master
# commit 5c4c1b583 (2026-06-12), which tracks upstream acts-project/acts through PR
# #5574 -> the corresponding tagged release is v46.8.0 (2026-06-11). To re-derive
# for a new container: `git -C <acts checkout> show -s --format=%ci HEAD` for the
# date, then match against acts-project/acts GitHub releases (the fork's own
# version_number is the dev placeholder 999.999.999, not a release string).
ACTS_VERSION = "v46.8.0"

_FONT_DIR = Path(__file__).resolve().parent / "fonts"


def ensure_bold_font():
    """Register the vendored bold DejaVu faces if the environment lacks a bold face.
    Idempotent. Called by paper_style(); safe to call directly too."""
    have_bold = any("bold" in Path(f.fname).name.lower() and "dejavusans" in Path(f.fname).name.lower()
                    for f in fontManager.ttflist)
    if have_bold:
        return
    registered = {Path(f.fname).name for f in fontManager.ttflist}
    added = False
    for ttf in sorted(_FONT_DIR.glob("*.ttf")):
        if ttf.name not in registered:
            fontManager.addfont(str(ttf))
            added = True
    if not added and not have_bold:
        warnings.warn("odd_plot_style: no bold DejaVu Sans face available "
                      "(vendored fonts missing); the ODD label may render un-bold.")


def paper_style():
    """ATLAS-like tracking-plot style: full axis box, ticks inside on all four
    sides with minor ticks, no grid, sans-serif. Also self-heals the bold font."""
    ensure_bold_font()
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


def odd_label(ax, extra=None, sample=r"$t\bar{t}$, $\sqrt{s}$ = 14 TeV, $\langle\mu\rangle$ = 200",
              acts_version=None, x=0.05, y=0.965):
    """ODD experiment label:  *ODD* Simulation  /  ACTS <ver>  /  <sample>.
    'ODD' is bold-italic (needs a bold DejaVu face -> see ensure_bold_font), the
    rest upright. `extra` is appended to the sample line (e.g. a pT cut)."""
    fam = "DejaVu Sans"
    line1 = HPacker(pad=0, sep=5, align="baseline", children=[
        TextArea("ODD", textprops=dict(fontsize=16, fontweight="bold",
                                       fontstyle="italic", fontfamily=fam)),
        TextArea("Simulation", textprops=dict(fontsize=15, fontfamily=fam)),
    ])
    info = sample + (f", {extra}" if extra else "")
    box = VPacker(pad=0, sep=3, align="left", children=[
        line1,
        TextArea(f"ACTS {acts_version or ACTS_VERSION}", textprops=dict(fontsize=11, fontfamily=fam)),
        TextArea(info, textprops=dict(fontsize=11, fontfamily=fam)),
    ])
    ax.add_artist(AnchoredOffsetbox(
        loc="upper left", child=box, pad=0.0, borderpad=0.0, frameon=False,
        bbox_to_anchor=(x, y), bbox_transform=ax.transAxes))


# ---------- error-bar-line drawing ----------

def density_hist(vals, bins):
    """Return (centres, density, density_err, half_bin_width). No lines drawn."""
    h, edges = np.histogram(vals, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    norm = max(h.sum() * width, 1)
    return centres, h / norm, np.sqrt(h) / norm, np.full_like(centres, width / 2)


def draw_series(ax, x, y, xerr, color, yerr=None, elinewidth=1.8):
    """One series as error-bar lines: a colour-coded horizontal segment spanning
    each bin (x err = half bin width) plus a vertical y-error bar, NO marker.
    Series are told apart by COLOUR only (no marker-shape double-encoding)."""
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor=color,
                elinewidth=elinewidth)


def draw_hist(ax, vals, bins, color):
    """Density histogram drawn as error-bar lines (see draw_series). y-error is the
    Poisson count uncertainty per bin. Returns (centres, density, err, half_width)."""
    c, d, derr, xerr = density_hist(vals, bins)
    draw_series(ax, c, d, xerr, color, yerr=derr)
    return c, d, derr, xerr


def legend_lines(ax, colors, labels, **kw):
    """Legend with plain colour lines as handles (matches the no-marker series)."""
    handles = [Line2D([0], [0], color=c, lw=2.6) for c in colors]
    ax.legend(handles, labels, **kw)


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
