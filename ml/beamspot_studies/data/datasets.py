"""
PyTorch datasets for the beam spot study.

Pre-processes all tracks into flat tensors at init time, so __getitem__
is a single tensor index with zero overhead during training.
"""

import glob
import hashlib
import math
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

PARAM_NAMES_RAW = ["d0", "z0", "phi", "theta", "qop"]
PARAM_NAMES_MODEL = ["d0", "z0", "sin_phi", "cos_phi", "cot_theta", "qop"]
N_OUTPUT = 6
N_HIT_FEATURES = 12  # r, phi, z, vol, lay, det, dr, dphi, dr_dphi, dz_dr, u_conf, v_conf
N_CLS_FEATURES = 8   # z_intercept, dz_dr_slope, curvature, sagitta, delta_r, delta_phi, delta_z, n_hits


def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def raw_to_model_params(params_raw):
    """[d0, z0, phi, theta, qop] -> [d0, z0, sin(phi), cos(phi), theta, qop]."""
    d0, z0, phi, theta, qop = params_raw
    return np.array([d0, z0, np.sin(phi), np.cos(phi), theta, qop], dtype=np.float32)


def model_to_raw_params(params_model):
    """[d0, z0, sin_phi, cos_phi, cot_theta, qop] -> [d0, z0, phi, theta, qop]."""
    d0, z0, sin_phi, cos_phi, cot_theta, qop = params_model
    phi = np.arctan2(sin_phi, cos_phi)
    theta = np.arctan2(1.0, cot_theta)
    return np.array([d0, z0, phi, theta, qop], dtype=np.float32)


class TrackHitDataset(Dataset):
    """Pre-processed dataset: all tracks loaded into tensors at init.

    __getitem__ is a simple tensor index — zero overhead during training.

    Caches the pre-processed tensors to disk as .pt files so subsequent
    runs with the same data skip the processing step.
    """

    def __init__(self, parquet_base, max_hits=20, max_files=None,
                 cache_dir=None):
        self.max_hits = max_hits
        parquet_base = Path(parquet_base)

        track_files = sorted(glob.glob(str(parquet_base / "reco/tracks/*.parquet")))
        hit_files = sorted(glob.glob(str(parquet_base / "reco/tracker_hits/*.parquet")))
        particle_files = sorted(glob.glob(str(parquet_base / "truth/particles/*.parquet")))

        if not track_files:
            raise FileNotFoundError(f"No track files in {parquet_base / 'reco/tracks/'}")

        if max_files is not None:
            track_files = track_files[:max_files]
            hit_files = hit_files[:max_files]
            particle_files = particle_files[:max_files]

        # Check for cached tensors
        cache_path = self._get_cache_path(track_files, max_hits, cache_dir or parquet_base)
        if cache_path.exists():
            print(f"Loading cached tensors from {cache_path}")
            t0 = time.time()
            cached = torch.load(cache_path, weights_only=False)
            self.hit_features = cached["hit_features"]
            self.truth_params = cached["truth_params"]
            self.reco_params = cached["reco_params"]
            self.padding_mask = cached["padding_mask"]
            self.n_hits = cached["n_hits"]
            self.cls_features = cached["cls_features"]
            self._input_std = cached["input_std"]
            self._cls_std = cached["cls_std"]
            self._output_scales = cached["output_scales"]
            print(f"Loaded {len(self)} tracks in {time.time()-t0:.1f}s")
            return

        # Pre-process all files
        print(f"Pre-processing {len(track_files)} files...")
        t0 = time.time()

        all_feats, all_truth, all_reco, all_mask, all_nhits, all_cls = [], [], [], [], [], []

        for fi in range(len(track_files)):
            tracks_tbl = pq.read_table(track_files[fi])
            hits_tbl = pq.read_table(hit_files[fi])
            particles_tbl = pq.read_table(particle_files[fi])

            for ei in range(len(tracks_tbl)):
                self._process_event(
                    tracks_tbl, hits_tbl, particles_tbl, ei,
                    all_feats, all_truth, all_reco, all_mask, all_nhits, all_cls,
                )

            if (fi + 1) % 5 == 0 or fi == len(track_files) - 1:
                print(f"  File {fi+1}/{len(track_files)}: {len(all_feats)} tracks so far")

        n = len(all_feats)
        print(f"Processed {n} valid tracks in {time.time()-t0:.1f}s")

        # Stack into tensors
        self.hit_features = torch.from_numpy(np.stack(all_feats))   # (N, max_hits, 12)
        self.truth_params = torch.from_numpy(np.stack(all_truth))   # (N, 6)
        self.reco_params = torch.from_numpy(np.stack(all_reco))     # (N, 6)
        self.padding_mask = torch.from_numpy(np.stack(all_mask))    # (N, max_hits)
        self.n_hits = torch.tensor(all_nhits, dtype=torch.long)     # (N,)
        self.cls_features = torch.from_numpy(np.stack(all_cls))     # (N, N_CLS_FEATURES)

        # Compute normalization (scale only, no mean shift)
        self._compute_normalization()

        # Apply input normalization: divide by std only (no mean subtraction)
        mask_expanded = self.padding_mask.unsqueeze(-1).expand_as(self.hit_features)
        input_scale = torch.from_numpy(self._input_std).unsqueeze(0).unsqueeze(0)
        normalized = self.hit_features / input_scale
        self.hit_features = torch.where(mask_expanded, normalized, torch.zeros_like(normalized))

        # Normalize CLS features by their std
        cls_std = self.cls_features.std(dim=0).numpy().astype(np.float32)
        cls_std[cls_std < 1e-6] = 1.0
        self._cls_std = cls_std
        self.cls_features = self.cls_features / torch.from_numpy(cls_std).unsqueeze(0)

        # Apply output normalization: divide by scale
        output_scales = torch.from_numpy(self._output_scales).unsqueeze(0)
        self.truth_params = self.truth_params / output_scales

        # Cache to disk
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "hit_features": self.hit_features,
                "truth_params": self.truth_params,
                "reco_params": self.reco_params,
                "padding_mask": self.padding_mask,
                "n_hits": self.n_hits,
                "cls_features": self.cls_features,
                "input_std": self._input_std,
                "cls_std": self._cls_std,
                "output_scales": self._output_scales,
            }, cache_path)
            print(f"Cached to {cache_path}")
        except Exception as e:
            print(f"Warning: could not save cache: {e}")

    # Bump this when normalization or feature computation changes
    CACHE_VERSION = 4  # v4: CLS features (z_intercept, curvature, etc.)

    def _get_cache_path(self, track_files, max_hits, cache_dir):
        """Deterministic cache path based on file list, max_hits, and code version."""
        key = f"v{self.CACHE_VERSION}_{[str(f) for f in track_files]}_{max_hits}"
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        return Path(cache_dir) / f".track_cache_{h}.pt"

    def _process_event(self, tracks_tbl, hits_tbl, particles_tbl, ei,
                       out_feats, out_truth, out_reco, out_mask, out_nhits, out_cls):
        """Extract all valid tracks from one event, appending to output lists."""
        def to_np(col, idx, dtype=np.float32):
            return np.array(col[idx].as_py(), dtype=dtype)

        hit_x = to_np(hits_tbl["x"], ei)
        hit_y = to_np(hits_tbl["y"], ei)
        hit_z = to_np(hits_tbl["z"], ei)
        hit_vol = to_np(hits_tbl["volume_id"], ei)
        hit_lay = to_np(hits_tbl["layer_id"], ei)
        hit_det = to_np(hits_tbl["detector"], ei)
        n_total_hits = len(hit_x)

        part_ids = to_np(particles_tbl["particle_id"], ei, np.int64)
        part_px = to_np(particles_tbl["px"], ei)
        part_py = to_np(particles_tbl["py"], ei)
        part_pz = to_np(particles_tbl["pz"], ei)
        part_charge = to_np(particles_tbl["charge"], ei)
        part_d0 = to_np(particles_tbl["perigee_d0"], ei)
        part_z0 = to_np(particles_tbl["perigee_z0"], ei)
        pid_to_idx = {int(pid): i for i, pid in enumerate(part_ids)}

        track_d0 = tracks_tbl["d0"][ei].as_py()
        track_z0 = tracks_tbl["z0"][ei].as_py()
        track_phi = tracks_tbl["phi"][ei].as_py()
        track_theta = tracks_tbl["theta"][ei].as_py()
        track_qop = tracks_tbl["qop"][ei].as_py()
        track_majpid = tracks_tbl["majority_particle_id"][ei].as_py()
        track_hit_ids = tracks_tbl["hit_ids"][ei].as_py()

        for ti in range(len(track_d0)):
            hids = track_hit_ids[ti]
            if not hids:
                continue
            hids = [h for h in hids if h < n_total_hits]
            if not hids:
                continue

            maj_pid = int(track_majpid[ti])
            if maj_pid not in pid_to_idx:
                continue
            pi = pid_to_idx[maj_pid]

            if np.isnan(part_d0[pi]) or np.isnan(part_z0[pi]) or part_charge[pi] == 0:
                continue

            # Truth — use cot(theta) = pz/pt as the angular parameter
            p_tot = math.sqrt(float(part_px[pi])**2 + float(part_py[pi])**2 + float(part_pz[pi])**2)
            pt = math.sqrt(float(part_px[pi])**2 + float(part_py[pi])**2)
            phi = math.atan2(float(part_py[pi]), float(part_px[pi]))
            cot_theta = float(part_pz[pi]) / (pt + 1e-10)
            qop = float(part_charge[pi]) / p_tot if p_tot > 0 else 0.0
            truth = np.array([float(part_d0[pi]), float(part_z0[pi]),
                              math.sin(phi), math.cos(phi), cot_theta, qop], dtype=np.float32)

            # Reco — convert theta to cot(theta)
            reco_phi = float(track_phi[ti])
            reco_theta = float(track_theta[ti])
            reco_cot_theta = math.cos(reco_theta) / (math.sin(reco_theta) + 1e-10)
            reco = np.array([float(track_d0[ti]), float(track_z0[ti]),
                             math.sin(reco_phi), math.cos(reco_phi),
                             reco_cot_theta, float(track_qop[ti])], dtype=np.float32)

            # Hit features
            hids_arr = np.array(hids)
            x = hit_x[hids_arr]; y = hit_y[hids_arr]; z = hit_z[hids_arr]
            r = np.sqrt(x**2 + y**2)
            phi_hit = np.arctan2(y, x)

            sort_idx = np.argsort(r)
            r = r[sort_idx]; phi_hit = phi_hit[sort_idx]; z = z[sort_idx]
            vol = hit_vol[hids_arr[sort_idx]]
            lay = hit_lay[hids_arr[sort_idx]]
            det = hit_det[hids_arr[sort_idx]]

            nh = len(r)
            dr = np.zeros(nh, dtype=np.float32)
            dphi = np.zeros(nh, dtype=np.float32)
            dr_dphi = np.zeros(nh, dtype=np.float32)
            dz_dr = np.zeros(nh, dtype=np.float32)
            if nh > 1:
                eps = 1e-6
                dr[1:] = r[1:] - r[:-1]
                dphi[1:] = wrap_to_pi(phi_hit[1:] - phi_hit[:-1])
                dr_dphi[1:] = np.clip(dr[1:] / (dphi[1:] + np.sign(dphi[1:] + eps) * eps), -1000, 1000)
                dz_dr[1:] = np.clip((z[1:] - z[:-1]) / (dr[1:] + eps), -100, 100)

            # Conformal coordinates: linearize helical trajectories
            x_sorted = r * np.cos(phi_hit)
            y_sorted = r * np.sin(phi_hit)
            r2 = x_sorted**2 + y_sorted**2 + 1e-10
            u_conf = x_sorted / r2
            v_conf = y_sorted / r2

            features = np.stack([r, phi_hit, z, vol, lay, det, dr, dphi, dr_dphi, dz_dr,
                                 u_conf, v_conf], axis=1)

            n_hits = min(nh, self.max_hits)
            padded = np.zeros((self.max_hits, N_HIT_FEATURES), dtype=np.float32)
            mask = np.zeros(self.max_hits, dtype=bool)
            padded[:n_hits] = features[:n_hits]
            mask[:n_hits] = True

            # Track-level summary features for CLS token
            eps = 1e-10
            delta_r_total = r[-1] - r[0]
            delta_phi_total = wrap_to_pi(phi_hit[-1] - phi_hit[0])
            delta_z_total = z[-1] - z[0]
            dz_dr_slope = delta_z_total / (delta_r_total + eps)
            z_intercept = z[0] - dz_dr_slope * r[0]

            # Sagitta from middle hit
            if nh >= 3:
                mid = nh // 2
                frac = (r[mid] - r[0]) / (delta_r_total + eps)
                phi_exp = phi_hit[0] + frac * delta_phi_total
                sagitta = wrap_to_pi(phi_hit[mid] - phi_exp)
            else:
                sagitta = 0.0

            # Menger curvature from first 3 hits
            if nh >= 3:
                x0, y0 = r[0]*np.cos(phi_hit[0]), r[0]*np.sin(phi_hit[0])
                x1, y1 = r[1]*np.cos(phi_hit[1]), r[1]*np.sin(phi_hit[1])
                x2, y2 = r[2]*np.cos(phi_hit[2]), r[2]*np.sin(phi_hit[2])
                area2 = abs((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0))
                d01 = np.sqrt((x1-x0)**2 + (y1-y0)**2) + eps
                d12 = np.sqrt((x2-x1)**2 + (y2-y1)**2) + eps
                d02 = np.sqrt((x2-x0)**2 + (y2-y0)**2) + eps
                curvature = np.clip(4*area2 / (d01*d12*d02 + eps), -1, 1)
            else:
                curvature = 0.0

            cls_feats = np.array([
                z_intercept, dz_dr_slope, curvature, sagitta,
                delta_r_total, delta_phi_total, delta_z_total, float(nh),
            ], dtype=np.float32)

            out_feats.append(padded)
            out_truth.append(truth)
            out_reco.append(reco)
            out_mask.append(mask)
            out_nhits.append(n_hits)
            out_cls.append(cls_feats)

    def _compute_normalization(self):
        """Compute input/output normalization scales from the full dataset.

        Scale-only normalization (no mean subtraction). Fully vectorized.
        """
        # Extract all real (non-padded) hits using mask indexing
        mask_3d = self.padding_mask.unsqueeze(-1).expand_as(self.hit_features)
        real_hits = self.hit_features[mask_3d].reshape(-1, N_HIT_FEATURES)
        self._input_std = real_hits.std(dim=0).numpy().astype(np.float32)
        self._input_std[self._input_std < 1e-6] = 1.0

        truth = self.truth_params.numpy()
        self._output_scales = np.array([
            max(np.std(truth[:, 0]), 1e-6),  # d0
            max(np.std(truth[:, 1]), 1e-6),  # z0
            1.0,                              # sin_phi
            1.0,                              # cos_phi
            max(np.std(truth[:, 4]), 1e-6),  # cot_theta (was pi for theta)
            max(np.std(truth[:, 5]), 1e-6),  # qop
        ], dtype=np.float32)

    def get_norm_stats(self):
        return {
            "input_std": self._input_std,
            "cls_std": self._cls_std,
            "output_scales": self._output_scales,
        }

    def __len__(self):
        return len(self.hit_features)

    def __getitem__(self, idx):
        return {
            "hit_features": self.hit_features[idx],
            "cls_features": self.cls_features[idx],
            "truth_params": self.truth_params[idx],
            "reco_params": self.reco_params[idx],
            "padding_mask": self.padding_mask[idx],
            "n_hits": self.n_hits[idx].item(),
        }
