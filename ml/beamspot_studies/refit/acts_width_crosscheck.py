#!/usr/bin/env python3
"""Cross-check our d0 resolution width against ACTS's OWN tooling.

Runs, via PyROOT, the exact ACTS ResPlotTool width estimator
(ActsPlugins::extractMeanWidthProfiles: iterative Gaussian fit, +-3 sigma,
3 iterations) on the ACTS-produced `res_d0_vs_eta` TH2F, hadd-merged to full
statistics. This is ACTS's own histogram + ROOT's own TF1 gaus fit — no
reimplementation.
"""
import glob
import ROOT
ROOT.gROOT.SetBatch(True)

WORK = "/pscratch/sd/d/danieltm/beamspot_refit"


def acts_width(proj, sigma_range=3.0, iters=3, min_entries=5):
    """Replicate extractMeanWidthProfiles per projection."""
    if proj.GetEntries() < min_entries:
        return None
    f = ROOT.TF1("g", "gaus", proj.GetXaxis().GetXmin(), proj.GetXaxis().GetXmax())
    if proj.Fit(f, "QN0") != 0:
        return None
    mu, s = f.GetParameter(1), f.GetParameter(2)
    for _ in range(iters):
        if s <= 0:
            break
        proj.Fit(f, "QN0", "", mu - sigma_range * s, mu + sigma_range * s)
        mu, s = f.GetParameter(1), f.GetParameter(2)
    return abs(s) * 1e3  # mm -> um


for cfg in ["none", "corrected"]:
    files = sorted(glob.glob(f"{WORK}/out/prod/*/performance_ckf_refit_{cfg}.root"))
    ch = ROOT.TFile.Open(f"{WORK}/merged_{cfg}.root")
    h2 = ch.Get("res_d0_vs_eta")   # X = eta (+-4), Y = residual d0 (+-0.5 mm)
    xa = h2.GetXaxis()
    print(f"\n=== cfg={cfg}: res_d0_vs_eta  X({xa.GetTitle()})[{xa.GetXmin()},{xa.GetXmax()}]"
          f" nbx={xa.GetNbins()}  Y[{h2.GetYaxis().GetXmin()},{h2.GetYaxis().GetXmax()}] ===")
    # eta ~ 0 slice (a few central bins)
    b0 = xa.FindBin(-0.25); b1 = xa.FindBin(0.25)
    proj0 = h2.ProjectionY("p0", b0, b1)
    w0 = acts_width(proj0)
    print(f"  eta in [-0.25,0.25]: N={proj0.GetEntries():.0f}  "
          f"ACTS gaus width = {w0:.1f} um   (RMS in +-0.5mm = {proj0.GetStdDev()*1e3:.1f} um)")
    for lo, hi in [(0.75, 1.25), (1.75, 2.25)]:
        p = h2.ProjectionY("p", xa.FindBin(lo), xa.FindBin(hi))
        print(f"  eta in [{lo},{hi}]:   N={p.GetEntries():.0f}  "
              f"ACTS width = {acts_width(p):.1f} um")
