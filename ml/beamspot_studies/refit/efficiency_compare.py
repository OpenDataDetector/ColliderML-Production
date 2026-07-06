#!/usr/bin/env python3
"""Extract tracking efficiency vs eta/pT from the ACTS EffPlotTool TEfficiency
objects (PyROOT — uproot can't read TEfficiency) in the hadd'd performance file,
and dump to JSON for matplotlib plotting against the paper reference.

Run in the container (has ROOT):
  hadd -f merged_perf_none.root out/prod/*/performance_ckf_refit_none.root
  python3 efficiency_compare.py merged_perf_none.root eff.json
"""
import json
import sys
import ROOT
ROOT.gROOT.SetBatch(True)


def teff_to_arrays(teff):
    h = teff.GetTotalHistogram()
    n = h.GetNbinsX()
    x, xe, eff, elo, ehi = [], [], [], [], []
    for b in range(1, n + 1):
        tot = h.GetBinContent(b)
        if tot <= 0:
            continue
        x.append(h.GetBinCenter(b))
        xe.append(h.GetBinWidth(b) / 2.0)
        eff.append(teff.GetEfficiency(b))
        elo.append(teff.GetEfficiencyErrorLow(b))
        ehi.append(teff.GetEfficiencyErrorUp(b))
    return dict(x=x, xerr=xe, eff=eff, elo=elo, ehi=ehi)


f = ROOT.TFile.Open(sys.argv[1])
out = {}
for name in ["trackeff_vs_eta", "trackeff_vs_pT"]:
    o = f.Get(name)
    if o:
        out[name] = teff_to_arrays(o)
        e = out[name]["eff"]
        print(f"{name}: {len(e)} bins, mean eff = {sum(e)/len(e):.4f}, "
              f"range [{min(e):.3f}, {max(e):.3f}]")
json.dump(out, open(sys.argv[2], "w"), indent=1)
print("wrote", sys.argv[2])
