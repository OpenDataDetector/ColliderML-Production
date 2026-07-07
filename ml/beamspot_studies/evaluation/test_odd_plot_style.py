#!/usr/bin/env python3
"""Lightweight regression test for odd_plot_style.

Run:  python test_odd_plot_style.py   (exits non-zero on failure)

Guards the bold-font self-heal: simulates a matplotlib env with no bold DejaVu
face and asserts ensure_bold_font() registers the vendored bold face so the ODD
label renders bold anywhere.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
from matplotlib.font_manager import FontProperties, findfont, fontManager

import odd_plot_style


def _clear_cache():
    try:
        findfont.cache_clear()
    except AttributeError:
        pass


def test_bold_font_selfheal():
    # strip every bold DejaVu face to simulate a stripped matplotlib install
    fontManager.ttflist = [
        f for f in fontManager.ttflist
        if not ("bold" in Path(f.fname).name.lower()
                and "dejavusans" in Path(f.fname).name.lower())
    ]
    _clear_cache()
    odd_plot_style.ensure_bold_font()
    _clear_cache()
    face = findfont(FontProperties(family="DejaVu Sans", weight="bold"))
    assert "bold" in Path(face).name.lower(), f"bold face not registered (got {face})"
    print("OK bold-font self-heal ->", Path(face).name)


def test_smoke_helpers():
    import numpy as np
    odd_plot_style.paper_style()
    fig, ax = matplotlib.pyplot.subplots()
    x = np.linspace(0, 10, 11)
    odd_plot_style.draw_series(ax, x, x, np.full_like(x, 0.5), "#1f77b4", yerr=x * 0.1)
    odd_plot_style.odd_label(ax)
    odd_plot_style.legend_lines(ax, ["#1f77b4"], ["demo"], loc="upper right")
    w, we = odd_plot_style.gauss_width(np.random.default_rng(0).normal(0, 1, 5000) * 1e3)
    assert 900 < w < 1100, f"gauss_width off: {w}"
    print("OK helper smoke (gauss_width ~", round(w), "um)")


if __name__ == "__main__":
    test_bold_font_selfheal()
    test_smoke_helpers()
    print("ALL TESTS PASSED")
