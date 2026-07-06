import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":12,"xtick.direction":"in","ytick.direction":"in",
  "xtick.top":True,"ytick.right":True,"xtick.minor.visible":True,"ytick.minor.visible":True,
  "axes.linewidth":1.0})
def load(fn):
    d=json.load(open("/pscratch/sd/d/danieltm/beamspot_refit/"+fn))["trackeff_vs_eta"]
    return (np.array(d["x"]),np.array(d["xerr"]),np.array(d["eff"]),
            np.array(d["elo"]),np.array(d["ehi"]))
fig,ax=plt.subplots(figsize=(7.2,4.6))
ax.axhspan(0.93,0.99,color="#d9c7e8",alpha=0.6,label="paper reference (ATLAS approx band)")
for fn,c,mk,lab in [("eff_primary.json","#1f77b4","o","this chain, primary particles"),
                    ("eff_none.json","#d62728","s","this chain, incl. secondaries")]:
    x,xe,e,el,eh=load(fn); m=np.abs(x)<3.0
    ax.errorbar(x[m],e[m],yerr=[el[m],eh[m]],xerr=xe[m],fmt=mk,color=c,ms=4,lw=1,label=lab)
ax.set_xlabel(r"$\eta$"); ax.set_ylabel("Tracking technical efficiency")
ax.set_xlim(-3,3); ax.set_ylim(0.70,1.01)
ax.text(0.03,0.10,"OpenDataDetector Simulation",transform=ax.transAxes,fontsize=11,
        fontstyle="italic",fontweight="bold")
ax.text(0.03,0.04,r"$t\bar t$, $\langle\mu\rangle=200$, $p_T>1$ GeV — finding matched to digi_and_reco",
        transform=ax.transAxes,fontsize=8.5)
ax.legend(loc="lower center",fontsize=8.5)
ax.annotate("|η|~2 loss from seed deltaR=(1,300)\n(digi_and_reco's own flagged setting)",
            xy=(2.0,0.80),xytext=(0.2,0.76),fontsize=7.5,
            arrowprops=dict(arrowstyle="->",lw=0.8,color="0.4"))
fig.tight_layout(); fig.savefig("/pscratch/sd/d/danieltm/beamspot_refit/eff_vs_eta.png",dpi=140)
print("saved")
