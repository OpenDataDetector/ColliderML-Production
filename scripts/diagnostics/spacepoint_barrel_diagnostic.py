#!/usr/bin/env python3
"""Diagnose why barrel SP z resolution is bad.

For each clean (prim+pT>1 GeV, single shared particle) barrel SP, compute:
  - SP_z         : algorithm SP global z
  - wafer_z      : average of the two cluster rec_gz (= wafer centre z)
  - truth_z      : midpoint of the two simhit true_z
  - m_implied    : (SP_z - wafer_z) / half_strip   (where the SP sits ALONG the strip)
  - m_truth      : (truth_z - wafer_z) / half_strip (where the particle ACTUALLY was)

A perfect algorithm would have m_implied = m_truth (with small noise).
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

ALLOWED_BARREL = {(29, 2), (29, 4)}
HALF_STRIP_BARREL = 54.0  # mm (vol 29: strip = 108mm long)


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

    print(f"[load] measurements.root", file=sys.stderr)
    m_arr = uproot.open(f"{args.run_dir}/measurements.root")["measurements"].arrays(
        ["event_nr", "volume_id", "layer_id",
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
    mgz = ak.to_numpy(m_arr["rec_gz"]).astype(np.float32)
    mtz = ak.to_numpy(m_arr["true_z"]).astype(np.float32)
    mtx = ak.to_numpy(m_arr["true_x"]).astype(np.float32)
    mty = ak.to_numpy(m_arr["true_y"]).astype(np.float32)
    mloc0_rec = ak.to_numpy(m_arr["rec_loc0"]).astype(np.float32)
    mloc0_true = ak.to_numpy(m_arr["true_loc0"]).astype(np.float32)

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
         "generation", "sub_particle", "pt", "eta",
         "vx", "vy", "vz"], library="ak")
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
    p_eta = ak.to_numpy(ak.flatten(p["eta"])).astype(np.float32)
    p_vx = ak.to_numpy(ak.flatten(p["vx"])).astype(np.float32)
    p_vy = ak.to_numpy(ak.flatten(p["vy"])).astype(np.float32)
    p_vz = ak.to_numpy(ak.flatten(p["vz"])).astype(np.float32)
    pinfo = {(int(p_ev[i]), int(p_bc[i])):
             (bool(int(p_vs[i]) == 0 and int(p_gg[i]) == 0),
              float(p_pt[i]), float(p_eta[i]),
              float(p_vx[i]), float(p_vy[i]), float(p_vz[i]))
             for i in range(len(p_bc))}

    sp = uproot.open(f"{args.run_dir}/spacepoints.root")["spacepoints"].arrays(
        ["event_id", "x", "y", "z", "measurement_id", "measurement_id_2"],
        library="np")
    n_sp = len(sp["event_id"])
    print(f"[sp] N={n_sp}", file=sys.stderr)

    # Collect per-clean-barrel-SP diagnostic
    SP_z = []          # algorithm SP z
    wafer_z = []       # avg(rec_gz of two clusters) = wafer center
    truth_z = []       # avg(true_z of two clusters) = truth midpoint
    m_implied = []     # (SP_z - wafer_z) / half_strip
    m_truth = []       # (truth_z - wafer_z) / half_strip
    delta_loc0 = []    # u_B - u_F (rec)
    delta_loc0_true = []  # u_B - u_F (truth)
    eta_p = []
    pt_p = []
    vz_p = []          # particle vertex z
    z_w_global = []    # wafer center as a function of where the wafer is

    for i in range(n_sp):
        e = int(sp["event_id"][i])
        r1 = ev_mid_to_row.get((e, int(sp["measurement_id"][i])))
        r2 = ev_mid_to_row.get((e, int(sp["measurement_id_2"][i])))
        if r1 is None or r2 is None:
            continue
        if (int(mv[r1]), int(ml[r1])) not in ALLOWED_BARREL:
            continue
        shared = m_psets[r1] & m_psets[r2]
        passing = [bc for bc in shared
                   if pinfo.get((e, bc), (False, 0, 0, 0, 0, 0))[0]
                   and pinfo.get((e, bc), (False, 0, 0, 0, 0, 0))[1] > 1.0]
        if len(passing) != 1:
            continue
        bc = passing[0]
        info = pinfo[(e, bc)]
        sp_z_val = float(sp["z"][i])
        w_z = 0.5 * (mgz[r1] + mgz[r2])
        t_z = 0.5 * (mtz[r1] + mtz[r2])
        SP_z.append(sp_z_val)
        wafer_z.append(w_z)
        truth_z.append(t_z)
        m_implied.append((sp_z_val - w_z) / HALF_STRIP_BARREL)
        m_truth.append((t_z - w_z) / HALF_STRIP_BARREL)
        delta_loc0.append(mloc0_rec[r2] - mloc0_rec[r1])
        delta_loc0_true.append(mloc0_true[r2] - mloc0_true[r1])
        eta_p.append(info[2])
        pt_p.append(info[1])
        vz_p.append(info[5])
        z_w_global.append(w_z)

    SP_z = np.array(SP_z); wafer_z = np.array(wafer_z); truth_z = np.array(truth_z)
    m_implied = np.array(m_implied); m_truth = np.array(m_truth)
    delta_loc0 = np.array(delta_loc0); delta_loc0_true = np.array(delta_loc0_true)
    eta_p = np.array(eta_p); pt_p = np.array(pt_p); vz_p = np.array(vz_p)
    z_w_global = np.array(z_w_global)

    print(f"[diag] N={len(SP_z)} clean barrel SPs", file=sys.stderr)
    print(f"[diag] m_implied stats:  mean={m_implied.mean():+.3f}  RMS={np.sqrt((m_implied**2).mean()):.3f}", file=sys.stderr)
    print(f"[diag] m_truth   stats:  mean={m_truth.mean():+.3f}  RMS={np.sqrt((m_truth**2).mean()):.3f}", file=sys.stderr)

    # Plot 1: m_implied vs m_truth (THE diagnostic — should be y=x line if working)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(m_truth, m_implied, s=2, alpha=0.3, color="C0")
    lim = 4
    ax.plot([-lim, lim], [-lim, lim], color="red", ls="--", lw=1, label="ideal y=x")
    ax.axhline(0, ls=":", color="gray", alpha=0.5)
    ax.axvline(0, ls=":", color="gray", alpha=0.5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("$m_\\mathrm{truth}$ = (truth_z - wafer_z) / half_strip")
    ax.set_ylabel("$m_\\mathrm{algorithm}$ = (SP_z - wafer_z) / half_strip")
    ax.set_title("Barrel SP: algorithm m vs truth m   (clean prim+pT>1 GeV)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "barrel_m_implied_vs_truth.png", dpi=140)
    plt.close(fig)
    print(f"saved {out_dir}/barrel_m_implied_vs_truth.png")

    # Plot 2: histogram of m_implied vs m_truth for direct comparison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(m_truth, bins=80, range=(-3, 3), histtype="step",
            color="C0", lw=1.6, label=f"$m_\\mathrm{{truth}}$ (truth particle z within wafer)\n  RMS={np.sqrt((m_truth**2).mean()):.2f}")
    ax.hist(m_implied, bins=80, range=(-3, 3), histtype="stepfilled",
            color="C2", alpha=0.5, edgecolor="C2", lw=1.6,
            label=f"$m_\\mathrm{{alg}}$ (algorithm SP_z relative to wafer)\n  RMS={np.sqrt((m_implied**2).mean()):.2f}")
    ax.axvline(-1, color="black", ls="--", lw=0.7, alpha=0.7)
    ax.axvline(+1, color="black", ls="--", lw=0.7, alpha=0.7)
    ax.set_xlabel("m  (SP/truth z relative to wafer centre, normalised by half-strip)")
    ax.set_ylabel("count")
    ax.set_title(f"Barrel: m parameter — algorithm vs truth (n={len(m_implied)})")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "barrel_m_distribution.png", dpi=140)
    plt.close(fig)
    print(f"saved {out_dir}/barrel_m_distribution.png")

    # Plot 3: residual m_implied - m_truth vs |truth wafer offset| (does it depend on where in the wafer)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    resid = m_implied - m_truth
    ax.scatter(m_truth, resid, s=2, alpha=0.3, color="C0")
    ax.axhline(0, color="red", ls="--", lw=1)
    ax.set_xlabel("$m_\\mathrm{truth}$")
    ax.set_ylabel("$m_\\mathrm{alg} - m_\\mathrm{truth}$")
    ax.set_title("Barrel SP: m residual vs truth m")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-3, 3)
    fig.tight_layout()
    fig.savefig(out_dir / "barrel_m_residual.png", dpi=140)
    plt.close(fig)
    print(f"saved {out_dir}/barrel_m_residual.png")

    # Plot 4: Δloc0 (rec) vs truth (proportional to truth_z if geometry works)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(delta_loc0_true, delta_loc0, s=2, alpha=0.3, color="C0", label="rec vs true")
    lim = 5
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=1, label="rec = true")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("$\\Delta\\mathrm{loc0}^\\mathrm{truth}$ [mm] = $u_B^\\mathrm{truth} - u_F^\\mathrm{truth}$")
    ax.set_ylabel("$\\Delta\\mathrm{loc0}^\\mathrm{rec}$ [mm]")
    ax.set_title("Barrel: Δloc0 reco vs truth (clean prim+pT>1 GeV)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "barrel_delta_loc0.png", dpi=140)
    plt.close(fig)
    print(f"saved {out_dir}/barrel_delta_loc0.png")

    # Plot 5: SP residual vs particle vertex_z (does the IP-vertex assumption fail for displaced particles?)
    sp_dz = SP_z - truth_z
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    axs[0].scatter(eta_p, sp_dz, s=2, alpha=0.3)
    axs[0].axhline(0, color="red", ls="--")
    axs[0].set_xlabel("particle η"); axs[0].set_ylabel("SP_z − truth_z [mm]")
    axs[0].set_ylim(-100, 100)
    axs[0].grid(True, alpha=0.3)
    axs[0].set_title("SP Δz vs η")
    axs[1].scatter(vz_p, sp_dz, s=2, alpha=0.3)
    axs[1].axhline(0, color="red", ls="--")
    axs[1].set_xlabel("particle vertex_z [mm]"); axs[1].set_ylabel("SP_z − truth_z [mm]")
    axs[1].set_ylim(-100, 100); axs[1].set_xlim(-200, 200)
    axs[1].grid(True, alpha=0.3)
    axs[1].set_title("SP Δz vs particle origin vertex z")
    fig.suptitle("Barrel SP Δz vs particle kinematics (clean prim+pT>1 GeV)")
    fig.tight_layout()
    fig.savefig(out_dir / "barrel_sp_dz_vs_kin.png", dpi=140)
    plt.close(fig)
    print(f"saved {out_dir}/barrel_sp_dz_vs_kin.png")


if __name__ == "__main__":
    main()
