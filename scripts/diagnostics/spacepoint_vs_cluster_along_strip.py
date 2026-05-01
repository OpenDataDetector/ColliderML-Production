#!/usr/bin/env python3
"""Compare the along-strip residual: spacepoint (2 clusters + stereo geometry)
vs single cluster (along-strip coord = wafer centre).

Question: does using the stereo SP buy you anything along the strip,
relative to just taking the cluster centroid?

For ODD long strips:
  - LS barrel  (vol 29): along-strip direction is global z.
  - LS endcap  (vol 28, 30): along-strip direction is radial r.

For each "clean" prim+pT>1 GeV SP, compute the along-strip residual using:
  (a) SP_along       (algorithm SP coord projected on strip direction)
  (b) cluster_along  (single cluster's rec position projected on strip dir;
                     same particle, on either of the two paired faces — we
                     average over the front/back face for symmetry).

Truth = the same particle's true_xyz on each face, averaged.
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
ALLOWED_ENDCAP = {(28, 2), (28, 4), (28, 6), (28, 8), (28, 10), (28, 12),
                  (30, 2), (30, 4), (30, 6), (30, 8), (30, 10), (30, 12)}
ALLOWED = ALLOWED_BARREL | ALLOWED_ENDCAP


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
    ap.add_argument("--pt-min", type=float, default=1.0)
    args = ap.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] measurements.root", file=sys.stderr)
    m_arr = uproot.open(f"{args.run_dir}/measurements.root")["measurements"].arrays(
        ["event_nr", "volume_id", "layer_id",
         "rec_gx", "rec_gy", "rec_gz",
         "true_x", "true_y", "true_z",
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
    mr_rec = np.sqrt(mgx ** 2 + mgy ** 2)
    mr_true = np.sqrt(mtx ** 2 + mty ** 2)

    # Particle bc-set per measurement
    print(f"[idx] building barcode sets", file=sys.stderr)
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

    # within-event -> row map
    seen = defaultdict(int)
    mid_per_row = np.zeros(n_m, dtype=int)
    for i, e in enumerate(m_ev):
        mid_per_row[i] = seen[int(e)]
        seen[int(e)] += 1
    ev_mid_to_row = {(int(m_ev[i]), int(mid_per_row[i])): i for i in range(n_m)}

    # Particle pT/primary lookup
    print(f"[idx] particles", file=sys.stderr)
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
    particle_info = {(int(p_ev[i]), int(p_bc[i])):
                     (bool(int(p_vs[i]) == 0 and int(p_gg[i]) == 0),
                      float(p_pt[i]))
                     for i in range(len(p_bc))}

    # Spacepoints
    print(f"[load] spacepoints.root", file=sys.stderr)
    sp = uproot.open(f"{args.run_dir}/spacepoints.root")["spacepoints"].arrays(
        ["event_id", "x", "y", "z", "measurement_id", "measurement_id_2"],
        library="np")
    sp_r = np.sqrt(sp["x"] ** 2 + sp["y"] ** 2)
    n_sp = len(sp["event_id"])
    print(f"[sp] N={n_sp}", file=sys.stderr)

    # For each clean SP, compute along-strip residuals for both SP and the
    # average single-cluster position.
    barrel_sp_dz, barrel_clu_dz = [], []
    endcap_sp_dr, endcap_clu_dr = [], []

    for i in range(n_sp):
        e = int(sp["event_id"][i])
        r1 = ev_mid_to_row.get((e, int(sp["measurement_id"][i])))
        r2 = ev_mid_to_row.get((e, int(sp["measurement_id_2"][i])))
        if r1 is None or r2 is None:
            continue
        v, l = int(mv[r1]), int(ml[r1])
        if (v, l) not in ALLOWED:
            continue
        # Clean: exactly one shared primary+pT>min particle
        shared = m_psets[r1] & m_psets[r2]
        passing = [bc for bc in shared
                   if particle_info.get((e, bc), (False, 0))[0]
                   and particle_info.get((e, bc), (False, 0))[1] > args.pt_min]
        if len(passing) != 1:
            continue
        # Truth = midpoint of the two true positions
        t_z = 0.5 * (mtz[r1] + mtz[r2])
        t_r = 0.5 * (mr_true[r1] + mr_true[r2])
        # Cluster along-strip = average of the two clusters' rec positions
        clu_z = 0.5 * (mgz[r1] + mgz[r2])
        clu_r = 0.5 * (mr_rec[r1] + mr_rec[r2])
        sp_z = float(sp["z"][i])
        sp_r_v = float(sp_r[i])

        if (v, l) in ALLOWED_BARREL:
            barrel_sp_dz.append(sp_z - t_z)
            barrel_clu_dz.append(clu_z - t_z)
        elif (v, l) in ALLOWED_ENDCAP:
            endcap_sp_dr.append(sp_r_v - t_r)
            endcap_clu_dr.append(clu_r - t_r)

    barrel_sp_dz = np.array(barrel_sp_dz)
    barrel_clu_dz = np.array(barrel_clu_dz)
    endcap_sp_dr = np.array(endcap_sp_dr)
    endcap_clu_dr = np.array(endcap_clu_dr)

    def stats(arr):
        return (np.median(arr),
                np.sqrt(np.mean(arr ** 2)),
                np.quantile(np.abs(arr), 0.68),
                np.quantile(np.abs(arr), 0.95))

    print(f"[barrel n={len(barrel_sp_dz)}]")
    print(f"  SP  Δz : median={stats(barrel_sp_dz)[0]:+.2f} RMS={stats(barrel_sp_dz)[1]:.2f} 68%={stats(barrel_sp_dz)[2]:.2f} 95%={stats(barrel_sp_dz)[3]:.2f}")
    print(f"  CLU Δz : median={stats(barrel_clu_dz)[0]:+.2f} RMS={stats(barrel_clu_dz)[1]:.2f} 68%={stats(barrel_clu_dz)[2]:.2f} 95%={stats(barrel_clu_dz)[3]:.2f}")

    print(f"[endcap n={len(endcap_sp_dr)}]")
    print(f"  SP  Δr : median={stats(endcap_sp_dr)[0]:+.2f} RMS={stats(endcap_sp_dr)[1]:.2f} 68%={stats(endcap_sp_dr)[2]:.2f} 95%={stats(endcap_sp_dr)[3]:.2f}")
    print(f"  CLU Δr : median={stats(endcap_clu_dr)[0]:+.2f} RMS={stats(endcap_clu_dr)[1]:.2f} 68%={stats(endcap_clu_dr)[2]:.2f} 95%={stats(endcap_clu_dr)[3]:.2f}")

    # Plots: side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    bins = 200
    rng_b = (-100, 100)
    ax1.hist(barrel_clu_dz, bins=bins, range=rng_b, histtype="step",
             color="C0", label=f"single-cluster $\\Delta z$ (rec_z = wafer centre)\n"
                               f"  RMS={stats(barrel_clu_dz)[1]:.1f}, 68%<{stats(barrel_clu_dz)[2]:.1f}",
             linewidth=1.6)
    ax1.hist(barrel_sp_dz, bins=bins, range=rng_b, histtype="stepfilled",
             color="C2", alpha=0.5,
             label=f"two-cluster SP $\\Delta z$ (stereo intersection)\n"
                   f"  RMS={stats(barrel_sp_dz)[1]:.1f}, 68%<{stats(barrel_sp_dz)[2]:.1f}",
             edgecolor="C2", linewidth=1.6)
    ax1.set_title(f"LS barrel (vol 29) — along-strip is global z   n={len(barrel_sp_dz):,}")
    ax1.set_xlabel(r"$\Delta z$ [mm]")
    ax1.set_ylabel("count")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    rng_e = (-100, 100)
    ax2.hist(endcap_clu_dr, bins=bins, range=rng_e, histtype="step",
             color="C0",
             label=f"single-cluster $\\Delta r$ (rec_r = wafer centre)\n"
                   f"  RMS={stats(endcap_clu_dr)[1]:.1f}, 68%<{stats(endcap_clu_dr)[2]:.1f}",
             linewidth=1.6)
    ax2.hist(endcap_sp_dr, bins=bins, range=rng_e, histtype="stepfilled",
             color="C2", alpha=0.5,
             label=f"two-cluster SP $\\Delta r$ (stereo intersection)\n"
                   f"  RMS={stats(endcap_sp_dr)[1]:.1f}, 68%<{stats(endcap_sp_dr)[2]:.1f}",
             edgecolor="C2", linewidth=1.6)
    ax2.set_title(f"LS endcap (vol 28/30) — along-strip is radial r   n={len(endcap_sp_dr):,}")
    ax2.set_xlabel(r"$\Delta r$ [mm]")
    ax2.set_ylabel("count")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Along-strip residual: 2-cluster SP vs single-cluster (clean prim+pT>{args.pt_min:.0f} GeV)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "sp_vs_cluster_along_strip.png", dpi=140)
    plt.close(fig)
    print(f"saved {out_dir}/sp_vs_cluster_along_strip.png", file=sys.stderr)

    # Also a zoomed-in view to see SP precision better
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    rng_zoom = (-15, 15)
    ax1.hist(barrel_clu_dz, bins=200, range=rng_zoom, histtype="step",
             color="C0", linewidth=1.6,
             label=f"single-cluster (RMS={stats(barrel_clu_dz)[1]:.1f}\\,mm)")
    ax1.hist(barrel_sp_dz, bins=200, range=rng_zoom, histtype="stepfilled",
             color="C2", alpha=0.5, edgecolor="C2", linewidth=1.6,
             label=f"two-cluster SP (RMS={stats(barrel_sp_dz)[1]:.1f}\\,mm)")
    ax1.set_title(f"LS barrel — zoomed to ±15\\,mm")
    ax1.set_xlabel(r"$\Delta z$ [mm]"); ax1.set_ylabel("count")
    ax1.legend(loc="upper right", fontsize=9); ax1.grid(True, alpha=0.3)
    ax2.hist(endcap_clu_dr, bins=200, range=rng_zoom, histtype="step",
             color="C0", linewidth=1.6,
             label=f"single-cluster (RMS={stats(endcap_clu_dr)[1]:.1f}\\,mm)")
    ax2.hist(endcap_sp_dr, bins=200, range=rng_zoom, histtype="stepfilled",
             color="C2", alpha=0.5, edgecolor="C2", linewidth=1.6,
             label=f"two-cluster SP (RMS={stats(endcap_sp_dr)[1]:.1f}\\,mm)")
    ax2.set_title(f"LS endcap — zoomed to ±15\\,mm")
    ax2.set_xlabel(r"$\Delta r$ [mm]"); ax2.set_ylabel("count")
    ax2.legend(loc="upper right", fontsize=9); ax2.grid(True, alpha=0.3)
    fig.suptitle(f"Along-strip residual (zoomed): 2-cluster SP vs single-cluster",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "sp_vs_cluster_along_strip_zoomed.png", dpi=140)
    plt.close(fig)
    print(f"saved {out_dir}/sp_vs_cluster_along_strip_zoomed.png", file=sys.stderr)

    # Append to summary
    summary = out_dir / "residual_summary.txt"
    with open(summary, "a") as f:
        def stat_line(label, arr):
            med, rms, p68, p95 = stats(arr)
            return f"{label:<48s} N={len(arr):>8,} median={med:>+8.3f}  RMS={rms:>7.3f}  68%={p68:>7.3f}  95%={p95:>8.3f}\n"
        f.write("\n--- Along-strip residual: 2-cluster SP vs single-cluster ---\n")
        f.write(stat_line("Barrel (vol 29) cluster Δz (rec - true)", barrel_clu_dz))
        f.write(stat_line("Barrel (vol 29) SP      Δz (rec - true)", barrel_sp_dz))
        f.write(stat_line("Endcap (vol 28/30) cluster Δr (rec - true)", endcap_clu_dr))
        f.write(stat_line("Endcap (vol 28/30) SP      Δr (rec - true)", endcap_sp_dr))


if __name__ == "__main__":
    main()
