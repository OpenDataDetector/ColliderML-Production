#!/usr/bin/env python3
"""Three new diagnostic plots for the expert presentation:
  1. Endcap m diagnostic (analog to barrel)
  2. Cluster pitch residual restricted to clusters that participate in
     clean primary+pT>1 SPs (the noise THE SP CONSTRUCTION ACTUALLY SEES)
  3. Per-wafer cluster density (clusters/wafer/event for each layer)
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

ENDCAP = {(28, 2), (28, 4), (28, 6), (28, 8), (28, 10), (28, 12),
          (30, 2), (30, 4), (30, 6), (30, 8), (30, 10), (30, 12)}
BARREL = {(29, 2), (29, 4)}
ALLOWED = ENDCAP | BARREL
HALF_STRIP_BARREL = 54.0
HALF_STRIP_ENDCAP = 78.0


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
    args = ap.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[load] measurements.root", file=sys.stderr)
    m_arr = uproot.open(f"{args.run_dir}/measurements.root")["measurements"].arrays(
        ["event_nr", "volume_id", "layer_id", "surface_id",
         "rec_gx", "rec_gy", "rec_gz",
         "true_x", "true_y", "true_z",
         "rec_loc0", "true_loc0",
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
    mloc0_rec = ak.to_numpy(m_arr["rec_loc0"]).astype(np.float32)
    mloc0_true = ak.to_numpy(m_arr["true_loc0"]).astype(np.float32)
    mr_rec = np.sqrt(mgx ** 2 + mgy ** 2)
    mr_true = np.sqrt(mtx ** 2 + mty ** 2)

    counts = ak.num(m_arr["particles_vertex_primary"]).to_numpy()
    offs = np.concatenate([[0], np.cumsum(counts)])
    bc_flat = pack_bc(
        ak.to_numpy(ak.flatten(m_arr["particles_vertex_primary"])).astype(np.uint64),
        ak.to_numpy(ak.flatten(m_arr["particles_vertex_secondary"])).astype(np.uint64),
        ak.to_numpy(ak.flatten(m_arr["particles_particle"])).astype(np.uint64),
        ak.to_numpy(ak.flatten(m_arr["particles_generation"])).astype(np.uint64),
        ak.to_numpy(ak.flatten(m_arr["particles_sub_particle"])).astype(np.uint64))
    m_psets = [frozenset(int(b) for b in bc_flat[offs[i]:offs[i+1]])
               for i in range(n_m)]

    seen = defaultdict(int)
    mid_per_row = np.zeros(n_m, dtype=int)
    for i, e in enumerate(m_ev):
        mid_per_row[i] = seen[int(e)]
        seen[int(e)] += 1
    ev_mid_to_row = {(int(m_ev[i]), int(mid_per_row[i])): i for i in range(n_m)}

    p = uproot.open(f"{args.run_dir}/particles.root")["particles"].arrays(
        ["event_id", "vertex_primary", "vertex_secondary", "particle",
         "generation", "sub_particle", "pt"], library="ak")
    p_ev_per_event = ak.to_numpy(p["event_id"]).astype(int)
    p_counts = ak.num(p["vertex_primary"]).to_numpy()
    p_ev = np.repeat(p_ev_per_event, p_counts)
    p_vs = ak.to_numpy(ak.flatten(p["vertex_secondary"])).astype(np.uint64)
    p_gg = ak.to_numpy(ak.flatten(p["generation"])).astype(np.uint64)
    p_bc = pack_bc(
        ak.to_numpy(ak.flatten(p["vertex_primary"])).astype(np.uint64),
        p_vs,
        ak.to_numpy(ak.flatten(p["particle"])).astype(np.uint64),
        p_gg,
        ak.to_numpy(ak.flatten(p["sub_particle"])).astype(np.uint64))
    p_pt = ak.to_numpy(ak.flatten(p["pt"])).astype(np.float32)
    pinfo = {(int(p_ev[i]), int(p_bc[i])):
             (bool(int(p_vs[i]) == 0 and int(p_gg[i]) == 0), float(p_pt[i]))
             for i in range(len(p_bc))}

    sp = uproot.open(f"{args.run_dir}/spacepoints.root")["spacepoints"].arrays(
        ["event_id", "x", "y", "z", "measurement_id", "measurement_id_2"],
        library="np")
    sp_r = np.sqrt(sp["x"] ** 2 + sp["y"] ** 2)
    n_sp = len(sp["event_id"])

    # ===== Plot 1: Endcap m diagnostic =====
    print("[1] endcap m diagnostic", file=sys.stderr)
    endcap_m_truth, endcap_m_alg = [], []
    barrel_m_truth, barrel_m_alg = [], []
    # Also collect cluster rec_loc0 - true_loc0 for clusters that participate in clean SPs
    clean_pitch_resid = []
    for i in range(n_sp):
        e = int(sp["event_id"][i])
        r1 = ev_mid_to_row.get((e, int(sp["measurement_id"][i])))
        r2 = ev_mid_to_row.get((e, int(sp["measurement_id_2"][i])))
        if r1 is None or r2 is None:
            continue
        if (int(mv[r1]), int(ml[r1])) not in ALLOWED:
            continue
        shared = m_psets[r1] & m_psets[r2]
        passing = [bc for bc in shared
                   if pinfo.get((e, bc), (False, 0))[0]
                   and pinfo.get((e, bc), (False, 0))[1] > 1.0]
        if len(passing) != 1:
            continue
        if (int(mv[r1]), int(ml[r1])) in ENDCAP:
            wafer_r = 0.5 * (mr_rec[r1] + mr_rec[r2])
            truth_r = 0.5 * (mr_true[r1] + mr_true[r2])
            sp_r_v = float(sp_r[i])
            endcap_m_truth.append((truth_r - wafer_r) / HALF_STRIP_ENDCAP)
            endcap_m_alg.append((sp_r_v - wafer_r) / HALF_STRIP_ENDCAP)
        elif (int(mv[r1]), int(ml[r1])) in BARREL:
            wafer_z = 0.5 * (mgz[r1] + mgz[r2])
            truth_z = 0.5 * (mtz[r1] + mtz[r2])
            sp_z = float(sp["z"][i])
            barrel_m_truth.append((truth_z - wafer_z) / HALF_STRIP_BARREL)
            barrel_m_alg.append((sp_z - wafer_z) / HALF_STRIP_BARREL)
        clean_pitch_resid.append(mloc0_rec[r1] - mloc0_true[r1])
        clean_pitch_resid.append(mloc0_rec[r2] - mloc0_true[r2])

    endcap_m_truth = np.array(endcap_m_truth)
    endcap_m_alg = np.array(endcap_m_alg)
    barrel_m_truth = np.array(barrel_m_truth)
    barrel_m_alg = np.array(barrel_m_alg)
    clean_pitch_resid = np.array(clean_pitch_resid)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    axs[0].hist(endcap_m_truth, bins=80, range=(-3, 3), histtype="step",
                color="C0", lw=1.6,
                label=f"$m_\\mathrm{{truth}}$ (RMS={np.sqrt((endcap_m_truth**2).mean()):.2f})")
    axs[0].hist(endcap_m_alg, bins=80, range=(-3, 3), histtype="stepfilled",
                color="C2", alpha=0.5, edgecolor="C2", lw=1.6,
                label=f"$m_\\mathrm{{alg}}$ (RMS={np.sqrt((endcap_m_alg**2).mean()):.2f})")
    axs[0].axvline(-1, ls="--", color="black", alpha=0.6)
    axs[0].axvline(+1, ls="--", color="black", alpha=0.6)
    axs[0].set_xlabel("m (along-strip = r, normalised by half-strip)")
    axs[0].set_ylabel("count")
    axs[0].set_title(f"Endcap m parameter: alg vs truth (n={len(endcap_m_alg)})")
    axs[0].legend(loc="upper right", fontsize=9)
    axs[0].grid(True, alpha=0.3)

    axs[1].hist(barrel_m_truth, bins=80, range=(-3, 3), histtype="step",
                color="C0", lw=1.6,
                label=f"$m_\\mathrm{{truth}}$ (RMS={np.sqrt((barrel_m_truth**2).mean()):.2f})")
    axs[1].hist(barrel_m_alg, bins=80, range=(-3, 3), histtype="stepfilled",
                color="C2", alpha=0.5, edgecolor="C2", lw=1.6,
                label=f"$m_\\mathrm{{alg}}$ (RMS={np.sqrt((barrel_m_alg**2).mean()):.2f})")
    axs[1].axvline(-1, ls="--", color="black", alpha=0.6)
    axs[1].axvline(+1, ls="--", color="black", alpha=0.6)
    axs[1].set_xlabel("m (along-strip = z, normalised by half-strip)")
    axs[1].set_title(f"Barrel m parameter: alg vs truth (n={len(barrel_m_alg)})")
    axs[1].legend(loc="upper right", fontsize=9)
    axs[1].grid(True, alpha=0.3)

    fig.suptitle("Algorithm m vs truth m — narrow in endcap, broad in barrel",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "endcap_vs_barrel_m_distribution.png", dpi=140)
    plt.close(fig)
    print(f"  saved {out_dir}/endcap_vs_barrel_m_distribution.png", file=sys.stderr)
    print(f"  endcap m_truth RMS = {np.sqrt((endcap_m_truth**2).mean()):.3f}, "
          f"m_alg RMS = {np.sqrt((endcap_m_alg**2).mean()):.3f}", file=sys.stderr)
    print(f"  barrel m_truth RMS = {np.sqrt((barrel_m_truth**2).mean()):.3f}, "
          f"m_alg RMS = {np.sqrt((barrel_m_alg**2).mean()):.3f}", file=sys.stderr)

    # ===== Plot 2: Clean-primary cluster pitch residual =====
    print("[2] clean-primary pitch residual", file=sys.stderr)
    # All-cluster pitch residual for comparison
    in_pipeline = np.array([(int(mv[i]), int(ml[i])) in ALLOWED for i in range(n_m)])
    all_pitch = (mloc0_rec - mloc0_true)[in_pipeline]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(all_pitch, bins=200, range=(-2, 2), histtype="step", color="C0", lw=1.6,
            density=True,
            label=f"all clusters in pipeline (n={len(all_pitch):,})\n"
                  f"  RMS={np.sqrt((all_pitch**2).mean()):.3f} mm,  "
                  f"68%={np.quantile(np.abs(all_pitch), 0.68)*1000:.0f} µm")
    ax.hist(clean_pitch_resid, bins=200, range=(-2, 2), histtype="stepfilled",
            color="C3", alpha=0.5, edgecolor="C3", lw=1.6, density=True,
            label=f"clusters in clean prim+pT>1 SPs (n={len(clean_pitch_resid):,})\n"
                  f"  RMS={np.sqrt((clean_pitch_resid**2).mean()):.3f} mm,  "
                  f"68%={np.quantile(np.abs(clean_pitch_resid), 0.68)*1000:.0f} µm")
    ax.set_xlabel(r"$\Delta\mathrm{loc0}$ = rec − true [mm]")
    ax.set_ylabel("density")
    ax.set_title("Cluster pitch residual: all-cluster vs clean-primary subset")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_dir / "cluster_pitch_clean_vs_all.png", dpi=140)
    plt.close(fig)
    print(f"  saved {out_dir}/cluster_pitch_clean_vs_all.png", file=sys.stderr)
    print(f"  all-cluster pitch RMS = {np.sqrt((all_pitch**2).mean()):.3f} mm",
          file=sys.stderr)
    print(f"  clean-primary pitch RMS = {np.sqrt((clean_pitch_resid**2).mean()):.3f} mm",
          file=sys.stderr)

    # ===== Plot 3: Per-wafer cluster density =====
    print("[3] per-wafer density", file=sys.stderr)
    # Count clusters per (vol, layer, surface) -> per wafer (= unique stereo couple)
    n_events = len(set(m_ev.tolist()))
    density = []  # (vol, layer, label, density)
    for vol, lay in sorted(ALLOWED):
        mask = (mv == vol) & (ml == lay)
        n_clusters = mask.sum()
        # Number of unique stereo couples in this layer
        s_arr = ak.to_numpy(m_arr["surface_id"])[mask].astype(int)
        unique_surfs = set(s_arr.tolist())
        # Count odd surfaces that have odd+1 partner
        n_couples = sum(1 for s in unique_surfs if s % 2 == 1 and (s + 1) in unique_surfs)
        if n_couples == 0:
            continue
        # clusters per wafer per event = n_clusters / (n_couples * 2) / n_events
        # (factor 2 because each wafer has 2 sensor faces)
        d = n_clusters / (n_couples * 2 * n_events)
        density.append((vol, lay, n_clusters, n_couples, d))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = [f"({v},{l})" for v, l, _, _, _ in density]
    ds = [d for _, _, _, _, d in density]
    colors = ["C0" if v == 28 else ("C1" if v == 29 else "C2")
              for v, _, _, _, _ in density]
    ax.bar(labels, ds, color=colors)
    ax.set_ylabel("clusters per sensor-face per event")
    ax.set_xlabel("(volume, layer)")
    ax.set_title("Per-sensor-face cluster density: endcap > barrel (corrected!)")
    ax.grid(True, alpha=0.3, axis="y")
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color="C0", label="vol 28 (LS endcap −)"),
               mpatches.Patch(color="C1", label="vol 29 (LS barrel)"),
               mpatches.Patch(color="C2", label="vol 30 (LS endcap +)")]
    ax.legend(handles=patches, loc="upper right", fontsize=9)
    for x, d in enumerate(ds):
        ax.text(x, d + 0.3, f"{d:.1f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "per_wafer_density.png", dpi=140)
    plt.close(fig)
    print(f"  saved {out_dir}/per_wafer_density.png", file=sys.stderr)
    for v, l, nc, np_, d in density:
        print(f"  vol={v:>3} lay={l:>3}: n_clusters={nc:>7,}  n_couples={np_:>4}  "
              f"density={d:>5.2f}/sensor/event", file=sys.stderr)


if __name__ == "__main__":
    main()
