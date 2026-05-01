#!/usr/bin/env python3
"""Compute and plot residuals for ODD long-strip clusters and spacepoints.

Inputs (from ACTS measurements.root + spacepoints.root + particles.root):
  - For each cluster (measurement): rec_g{x,y,z} (algorithm position, with
    loc1 set to wafer centre for 1-D strips) and true_{x,y,z} (truth hit
    position from the underlying simhit, after multi-particle averaging).
  - For each spacepoint: x, y, z (algorithm 3-D position) and the two
    contributing measurement indices.

Definitions:
  - "Cluster residual" = rec_g - true (per-axis). Captures cluster-reco
    errors and the loc1-set-to-wafer-centre approximation.
  - "Spacepoint residual" = SP_xyz - midpoint(truth1, truth2) where the
    midpoint uses the truth positions of the SHARED-particle on the two
    paired clusters. Only defined for SPs with a uniquely-shared
    primary+pT>1GeV particle (clean truth association).

Outputs PNGs into ../docs/spacepoint_phaseC/plots/.
"""
from __future__ import annotations
import argparse, sys
from collections import defaultdict
from pathlib import Path

import awkward as ak
import numpy as np
import uproot
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALLOWED = {(28, 2), (28, 4), (28, 6), (28, 8), (28, 10), (28, 12),
           (29, 2), (29, 4),
           (30, 2), (30, 4), (30, 6), (30, 8), (30, 10), (30, 12)}


def pack_bc(vp, vs, p, g, sub):
    return ((vp.astype(np.uint64) << np.uint64(52))
            | (vs.astype(np.uint64) << np.uint64(40))
            | (p.astype(np.uint64) << np.uint64(24))
            | (g.astype(np.uint64) << np.uint64(16))
            | sub.astype(np.uint64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--pt-min", type=float, default=1.0,
                    help="primary pT threshold in GeV for the SP residual")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] measurements.root", file=sys.stderr)
    m_arr = uproot.open(f"{args.run_dir}/measurements.root")["measurements"].arrays(
        ["event_nr", "volume_id", "layer_id", "surface_id",
         "rec_gx", "rec_gy", "rec_gz",
         "rec_loc0", "true_loc0",
         "true_x", "true_y", "true_z",
         "clus_size_loc0",
         "particles_vertex_primary", "particles_vertex_secondary",
         "particles_particle", "particles_generation",
         "particles_sub_particle"], library="ak")

    n_m = len(m_arr["event_nr"])
    mv = ak.to_numpy(m_arr["volume_id"]).astype(int)
    ml = ak.to_numpy(m_arr["layer_id"]).astype(int)
    m_ev = ak.to_numpy(m_arr["event_nr"]).astype(int)
    mgx = ak.to_numpy(m_arr["rec_gx"]).astype(np.float32)
    mgy = ak.to_numpy(m_arr["rec_gy"]).astype(np.float32)
    mgz = ak.to_numpy(m_arr["rec_gz"]).astype(np.float32)
    mtx = ak.to_numpy(m_arr["true_x"]).astype(np.float32)
    mty = ak.to_numpy(m_arr["true_y"]).astype(np.float32)
    mtz = ak.to_numpy(m_arr["true_z"]).astype(np.float32)

    in_pipeline = np.array(
        [(int(mv[i]), int(ml[i])) in ALLOWED for i in range(n_m)])

    # ===== Cluster residuals =====
    # The MEANINGFUL cluster residual for a 1-D strip is the across-strip
    # (loc0) precision — the strip is a line in 3-D, so loc1 is not a
    # measurement, only loc0 is. We use rec_loc0 (algorithm pitch position)
    # vs true_loc0 (truth pitch position from the simhit).
    # Δx/Δy/Δz residuals (rec_g - true) shown for completeness, but they are
    # dominated by the loc1=strip-centre approximation (the strip-end can be
    # ±half-strip-length from the wafer centre).
    print(f"[clu] computing residuals for {in_pipeline.sum()} long-strip clusters",
          file=sys.stderr)
    rl0 = ak.to_numpy(m_arr["rec_loc0"]).astype(np.float32)
    tl0 = ak.to_numpy(m_arr["true_loc0"]).astype(np.float32)
    cs0 = ak.to_numpy(m_arr["clus_size_loc0"]).astype(np.int32)
    dloc0 = (rl0 - tl0)[in_pipeline]  # mm in pitch direction — the real cluster precision
    cs0_sel = cs0[in_pipeline]
    dx = mgx[in_pipeline] - mtx[in_pipeline]
    dy = mgy[in_pipeline] - mty[in_pipeline]
    dz = mgz[in_pipeline] - mtz[in_pipeline]
    dr = np.sqrt(mgx[in_pipeline] ** 2 + mgy[in_pipeline] ** 2) \
         - np.sqrt(mtx[in_pipeline] ** 2 + mty[in_pipeline] ** 2)
    d3 = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    # Save the histograms
    def hist_panel(arr, label, fname, range_=None, bins=200, log=False, color="C0"):
        fig, ax = plt.subplots(figsize=(6.5, 4.4))
        ax.hist(arr, bins=bins, range=range_, histtype="stepfilled",
                edgecolor=color, facecolor=color, alpha=0.55)
        ax.set_xlabel(label)
        ax.set_ylabel("count")
        if log:
            ax.set_yscale("log")
        med = float(np.median(arr))
        p68 = float(np.quantile(np.abs(arr), 0.68))
        p95 = float(np.quantile(np.abs(arr), 0.95))
        rms = float(np.sqrt(np.mean(arr ** 2)))
        ax.text(0.97, 0.97,
                f"N={len(arr):,}\nmedian={med:.3f} mm\nRMS={rms:.2f} mm\n68% |·|<{p68:.2f}\n95% |·|<{p95:.2f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, family="monospace",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=130)
        plt.close(fig)

    # Pitch residual — the headline cluster-quality plot
    hist_panel(dloc0,
               r"$\Delta$loc0 = rec$_\mathrm{pitch}$ $-$ true$_\mathrm{pitch}$ [mm]   (cluster across-strip residual)",
               "cluster_resid_loc0.png", range_=(-2, 2), bins=200, color="C3")
    hist_panel(dloc0,
               r"$\Delta$loc0 [mm], log scale (cluster across-strip residual)",
               "cluster_resid_loc0_log.png", range_=(-10, 10), bins=200,
               log=True, color="C3")
    # 3-D residuals (large because rec_y == strip centre by construction)
    hist_panel(dx, r"$\Delta x$ = rec$_x$ $-$ true$_x$ [mm]   (DOMINATED by loc1=centre)",
               "cluster_resid_x.png", range_=(-100, 100))
    hist_panel(dy, r"$\Delta y$ = rec$_y$ $-$ true$_y$ [mm]   (DOMINATED by loc1=centre)",
               "cluster_resid_y.png", range_=(-100, 100))
    hist_panel(dz, r"$\Delta z$ = rec$_z$ $-$ true$_z$ [mm]",
               "cluster_resid_z.png", range_=(-100, 100))
    hist_panel(dr, r"$\Delta r$ = rec$_r$ $-$ true$_r$ [mm]",
               "cluster_resid_r.png", range_=(-100, 100))
    hist_panel(d3, r"3-D residual $\|\mathrm{rec}-\mathrm{true}\|$ [mm]   (DOMINATED by loc1=centre)",
               "cluster_resid_3d.png", range_=(0, 100))

    # Cluster pitch residual vs cluster size
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for size_bin, color in [((1, 1), "C0"), ((2, 2), "C1"), ((3, 4), "C2"),
                            ((5, 10), "C3"), ((11, 1000), "C4")]:
        sel = (cs0_sel >= size_bin[0]) & (cs0_sel <= size_bin[1])
        n = sel.sum()
        if n == 0:
            continue
        if size_bin[0] == size_bin[1]:
            label = f"loc0-size = {size_bin[0]} (n={n:,})"
        elif size_bin[1] >= 1000:
            label = f"loc0-size >= {size_bin[0]} (n={n:,})"
        else:
            label = f"loc0-size in [{size_bin[0]}, {size_bin[1]}] (n={n:,})"
        ax.hist(dloc0[sel], bins=120, range=(-2, 2), histtype="step",
                color=color, label=label, density=True)
    ax.set_xlabel(r"$\Delta$loc0 [mm] (across-strip)")
    ax.set_ylabel("density")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "cluster_resid_loc0_by_size.png", dpi=130)
    plt.close(fig)

    # Per-volume break-out (the loc1-along-strip residual is a function of
    # how long the strip is)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for vol, label, color in [(28, "vol 28 (LS endcap-)", "C0"),
                              (29, "vol 29 (LS barrel)", "C1"),
                              (30, "vol 30 (LS endcap+)", "C2")]:
        mask_v = (mv == vol) & in_pipeline
        d = d3[in_pipeline & mv[in_pipeline] == vol] if False else \
            np.sqrt(((mgx - mtx) ** 2 + (mgy - mty) ** 2 + (mgz - mtz) ** 2)[mask_v])
        ax.hist(d, bins=120, range=(0, 100), histtype="step",
                color=color, label=f"{label} (n={mask_v.sum():,})")
    ax.set_xlabel(r"3-D residual $\|\mathrm{rec}-\mathrm{true}\|$ [mm]")
    ax.set_ylabel("count")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "cluster_resid_3d_per_vol.png", dpi=130)
    plt.close(fig)

    # ===== Spacepoint residuals =====
    print(f"[sp] loading spacepoints.root", file=sys.stderr)
    sp = uproot.open(f"{args.run_dir}/spacepoints.root")["spacepoints"].arrays(
        ["event_id", "x", "y", "z", "measurement_id", "measurement_id_2"],
        library="np")
    n_sp = len(sp["event_id"])
    print(f"[sp] {n_sp} spacepoints, computing residuals", file=sys.stderr)

    # Build measurement->row map per (event, within-event-index)
    seen = defaultdict(int)
    mid_per_row = np.zeros(n_m, dtype=int)
    for i, e in enumerate(m_ev):
        mid_per_row[i] = seen[int(e)]
        seen[int(e)] += 1
    ev_mid_to_row = {(int(m_ev[i]), int(mid_per_row[i])): i for i in range(n_m)}

    # Per-measurement particle barcode set
    print(f"[sp] building particle barcode sets", file=sys.stderr)
    bc_offset = ak.num(m_arr["particles_vertex_primary"]).to_numpy()
    offsets = np.concatenate([[0], np.cumsum(bc_offset)])
    bc_flat = pack_bc(
        ak.to_numpy(ak.flatten(m_arr["particles_vertex_primary"])).astype(np.uint64),
        ak.to_numpy(ak.flatten(m_arr["particles_vertex_secondary"])).astype(np.uint64),
        ak.to_numpy(ak.flatten(m_arr["particles_particle"])).astype(np.uint64),
        ak.to_numpy(ak.flatten(m_arr["particles_generation"])).astype(np.uint64),
        ak.to_numpy(ak.flatten(m_arr["particles_sub_particle"])).astype(np.uint64))
    m_psets = [frozenset(int(b) for b in bc_flat[offsets[i]:offsets[i+1]])
               for i in range(n_m)]

    # Particle pT/primary lookup
    print(f"[sp] loading particles.root", file=sys.stderr)
    p = uproot.open(f"{args.run_dir}/particles.root")["particles"].arrays(
        ["event_id", "vertex_primary", "vertex_secondary", "particle",
         "generation", "sub_particle", "pt", "eta"], library="ak")
    p_ev_per_event = ak.to_numpy(p["event_id"]).astype(int)
    counts = ak.num(p["vertex_primary"]).to_numpy()
    p_ev = np.repeat(p_ev_per_event, counts)
    p_vs = ak.to_numpy(ak.flatten(p["vertex_secondary"])).astype(np.uint64)
    p_gg = ak.to_numpy(ak.flatten(p["generation"])).astype(np.uint64)
    p_bc = pack_bc(
        ak.to_numpy(ak.flatten(p["vertex_primary"])).astype(np.uint64),
        p_vs,
        ak.to_numpy(ak.flatten(p["particle"])).astype(np.uint64),
        p_gg,
        ak.to_numpy(ak.flatten(p["sub_particle"])).astype(np.uint64))
    p_pt = ak.to_numpy(ak.flatten(p["pt"])).astype(np.float32)
    p_eta = ak.to_numpy(ak.flatten(p["eta"])).astype(np.float32)
    particle_info = {(int(p_ev[i]), int(p_bc[i])):
                     (bool(int(p_vs[i]) == 0 and int(p_gg[i]) == 0),
                      float(p_pt[i]), float(p_eta[i]))
                     for i in range(len(p_bc))}

    # Compute SP residuals for the SUBSET that have a unique shared
    # primary+pT>min particle on both faces.
    sp_dx_clean, sp_dy_clean, sp_dz_clean = [], [], []
    sp_dx_all, sp_dy_all, sp_dz_all = [], [], []
    sp_d3_clean, sp_d3_all = [], []
    for i in range(n_sp):
        e = int(sp["event_id"][i])
        r1 = ev_mid_to_row.get((e, int(sp["measurement_id"][i])))
        r2 = ev_mid_to_row.get((e, int(sp["measurement_id_2"][i])))
        if r1 is None or r2 is None:
            continue
        if (int(mv[r1]), int(ml[r1])) not in ALLOWED:
            continue
        sx, sy, sz = float(sp["x"][i]), float(sp["y"][i]), float(sp["z"][i])
        # Truth midpoint from the shared particle's true_xyz on the two faces.
        # If multiple shared particles, ambiguous — fall back to the simple
        # midpoint of the two cluster's stored truth (which is itself
        # multi-particle averaged inside the digi).
        shared = m_psets[r1] & m_psets[r2]
        # Default truth = midpoint of the two stored true_xyz (which
        # ACTS records as the truth-hit position, possibly already
        # multi-particle averaged inside the digi for merged clusters).
        tx_mid = 0.5 * (mtx[r1] + mtx[r2])
        ty_mid = 0.5 * (mty[r1] + mty[r2])
        tz_mid = 0.5 * (mtz[r1] + mtz[r2])
        d3_sp = float(np.sqrt((sx - tx_mid) ** 2 + (sy - ty_mid) ** 2 +
                              (sz - tz_mid) ** 2))
        sp_dx_all.append(sx - tx_mid)
        sp_dy_all.append(sy - ty_mid)
        sp_dz_all.append(sz - tz_mid)
        sp_d3_all.append(d3_sp)
        # Clean subset: exactly-one shared primary+pT>min particle
        passing = [bc for bc in shared
                   if particle_info.get((e, bc), (False, 0, 0))[0]
                   and particle_info.get((e, bc), (False, 0, 0))[1] > args.pt_min]
        if len(passing) == 1:
            sp_dx_clean.append(sx - tx_mid)
            sp_dy_clean.append(sy - ty_mid)
            sp_dz_clean.append(sz - tz_mid)
            sp_d3_clean.append(d3_sp)

    sp_dx_clean = np.array(sp_dx_clean)
    sp_dy_clean = np.array(sp_dy_clean)
    sp_dz_clean = np.array(sp_dz_clean)
    sp_d3_clean = np.array(sp_d3_clean)
    sp_dx_all = np.array(sp_dx_all)
    sp_dy_all = np.array(sp_dy_all)
    sp_dz_all = np.array(sp_dz_all)
    sp_d3_all = np.array(sp_d3_all)
    print(f"[sp] N_all={len(sp_dx_all):,}, N_clean(prim+pT>{args.pt_min})={len(sp_dx_clean):,}",
          file=sys.stderr)

    hist_panel(sp_dx_clean, r"$\Delta x$ [mm]   (clean prim+pT>1 GeV SPs)",
               "sp_resid_x_clean.png", range_=(-30, 30), color="C2")
    hist_panel(sp_dy_clean, r"$\Delta y$ [mm]   (clean prim+pT>1 GeV SPs)",
               "sp_resid_y_clean.png", range_=(-30, 30), color="C2")
    hist_panel(sp_dz_clean, r"$\Delta z$ [mm]   (clean prim+pT>1 GeV SPs)",
               "sp_resid_z_clean.png", range_=(-15, 15), color="C2")
    hist_panel(sp_d3_clean, r"3-D residual [mm]   (clean prim+pT>1 GeV SPs)",
               "sp_resid_3d_clean.png", range_=(0, 50), color="C2")

    # Compare clean vs all
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.hist(sp_d3_all, bins=120, range=(0, 100), histtype="step",
            color="C0", label=f"all matched SPs (n={len(sp_d3_all):,})", density=True)
    ax.hist(sp_d3_clean, bins=120, range=(0, 100), histtype="step",
            color="C2", label=f"clean prim+pT>{args.pt_min} (n={len(sp_d3_clean):,})", density=True)
    ax.set_xlabel(r"3-D residual $\|\mathrm{SP}-\mathrm{truth}\|$ [mm]")
    ax.set_ylabel("density")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "sp_resid_3d_compare.png", dpi=130)
    plt.close(fig)

    # Print summary stats
    summary = out_dir / "residual_summary.txt"
    with open(summary, "w") as f:
        def stats(label, arr):
            return (f"{label:<40s} N={len(arr):>10,}  median={np.median(arr):>+8.3f} "
                    f"RMS={np.sqrt(np.mean(arr**2)):>7.3f} "
                    f"68%={np.quantile(np.abs(arr),0.68):>7.3f} "
                    f"95%={np.quantile(np.abs(arr),0.95):>8.3f}\n")
        f.write("Residual summary (mm)\n")
        f.write("--- Cluster: across-strip (loc0) — the meaningful precision ---\n")
        f.write(stats("Cluster Δloc0 (rec − true)", dloc0))
        f.write("--- Cluster: 3-D residuals (DOMINATED by loc1=strip-centre) ---\n")
        f.write(stats("Cluster Δx (rec_gx − true_x)", dx))
        f.write(stats("Cluster Δy (rec_gy − true_y)", dy))
        f.write(stats("Cluster Δz (rec_gz − true_z)", dz))
        f.write(stats("Cluster Δr (rec_r − true_r)", dr))
        f.write(stats("Cluster |Δ| (3-D)", d3))
        f.write("\n")
        f.write(stats("SP Δx clean prim+pT>{:.0f} GeV".format(args.pt_min), sp_dx_clean))
        f.write(stats("SP Δy clean prim+pT>{:.0f} GeV".format(args.pt_min), sp_dy_clean))
        f.write(stats("SP Δz clean prim+pT>{:.0f} GeV".format(args.pt_min), sp_dz_clean))
        f.write(stats("SP |Δ| clean prim+pT>{:.0f} GeV".format(args.pt_min), sp_d3_clean))
        f.write(stats("SP |Δ| all matched", sp_d3_all))
    print(f"[done] wrote summary to {summary}", file=sys.stderr)


if __name__ == "__main__":
    main()
