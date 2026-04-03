"""
PyTorch datasets for the beam spot study.

Loads ColliderML parquet files lazily with LRU file cache.
Converts hits to cylindrical coordinates with inter-hit delta features.
Normalizes inputs and outputs for stable training.
"""

import glob
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

# Output parameter indices (after sin/cos phi parameterization)
# [d0, z0, sin_phi, cos_phi, theta, qop]
PARAM_NAMES_RAW = ["d0", "z0", "phi", "theta", "qop"]
PARAM_NAMES_MODEL = ["d0", "z0", "sin_phi", "cos_phi", "theta", "qop"]
N_OUTPUT = 6
N_HIT_FEATURES = 10  # r, phi, z, vol, lay, det, dr, dphi, dr_dphi, dz_dr


def wrap_to_pi(x):
    """Wrap angle to [-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def compute_truth_params_raw(px, py, pz, charge, perigee_d0, perigee_z0):
    """Compute truth track parameters in raw format [d0, z0, phi, theta, qop]."""
    pt = math.sqrt(px**2 + py**2)
    p = math.sqrt(px**2 + py**2 + pz**2)
    phi = math.atan2(py, px)
    theta = math.atan2(pt, pz)
    qop = charge / p if p > 0 else 0.0
    return np.array([perigee_d0, perigee_z0, phi, theta, qop], dtype=np.float32)


def raw_to_model_params(params_raw):
    """Convert [d0, z0, phi, theta, qop] -> [d0, z0, sin(phi), cos(phi), theta, qop]."""
    d0, z0, phi, theta, qop = params_raw
    return np.array([d0, z0, np.sin(phi), np.cos(phi), theta, qop], dtype=np.float32)


def model_to_raw_params(params_model):
    """Convert [d0, z0, sin_phi, cos_phi, theta, qop] -> [d0, z0, phi, theta, qop]."""
    d0, z0, sin_phi, cos_phi, theta, qop = params_model
    phi = np.arctan2(sin_phi, cos_phi)
    return np.array([d0, z0, phi, theta, qop], dtype=np.float32)


def compute_hit_features(hit_x, hit_y, hit_z, hit_vol, hit_lay, hit_det, hit_indices):
    """Convert Cartesian hits to cylindrical + compute inter-hit deltas.

    Returns (positions, features) both of shape (n_hits, ...) sorted by radius.
    Full feature vector per hit: [r, phi, z, vol, lay, det, dr, dphi, dr_dphi, dz_dr]
    """
    x = hit_x[hit_indices]
    y = hit_y[hit_indices]
    z = hit_z[hit_indices]
    vol = hit_vol[hit_indices]
    lay = hit_lay[hit_indices]
    det = hit_det[hit_indices]

    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    # Sort by radius
    sort_idx = np.argsort(r)
    r = r[sort_idx]
    phi = phi[sort_idx]
    z = z[sort_idx]
    vol = vol[sort_idx]
    lay = lay[sort_idx]
    det = det[sort_idx]

    n = len(r)

    # Inter-hit deltas (first hit gets zeros)
    dr = np.zeros(n, dtype=np.float32)
    dphi = np.zeros(n, dtype=np.float32)
    dr_dphi = np.zeros(n, dtype=np.float32)
    dz_dr = np.zeros(n, dtype=np.float32)

    if n > 1:
        dr[1:] = r[1:] - r[:-1]
        dphi[1:] = wrap_to_pi(phi[1:] - phi[:-1])

        eps = 1e-6
        dr_dphi[1:] = dr[1:] / (dphi[1:] + np.sign(dphi[1:] + eps) * eps)
        dz_dr[1:] = (z[1:] - z[:-1]) / (dr[1:] + eps)

    # Stack all 10 features
    features = np.stack([r, phi, z, vol, lay, det, dr, dphi, dr_dphi, dz_dr], axis=1)
    return features  # (n_hits, 10)


class TrackHitDataset(Dataset):
    """Lazy-loading dataset with cylindrical coordinates, deltas, and normalization.

    Each sample:
        hit_features:  (max_hits, 10) float32 — normalized [r, phi, z, vol, lay, det, dr, dphi, dr/dphi, dz/dr]
        truth_params:  (6,) float32 — normalized [d0, z0, sin_phi, cos_phi, theta, qop]
        reco_params:   (6,) float32 — NOT normalized [d0, z0, sin_phi, cos_phi, theta, qop]
        padding_mask:  (max_hits,) bool — True for real hits
        n_hits:        int
    """

    def __init__(self, parquet_base, max_hits=20, max_files=None, file_cache_size=4):
        self.max_hits = max_hits
        parquet_base = Path(parquet_base)

        self.track_files = sorted(glob.glob(str(parquet_base / "reco/tracks/*.parquet")))
        self.hit_files = sorted(glob.glob(str(parquet_base / "reco/tracker_hits/*.parquet")))
        self.particle_files = sorted(glob.glob(str(parquet_base / "truth/particles/*.parquet")))

        if not self.track_files:
            raise FileNotFoundError(f"No track files in {parquet_base / 'reco/tracks/'}")

        if max_files is not None:
            self.track_files = self.track_files[:max_files]
            self.hit_files = self.hit_files[:max_files]
            self.particle_files = self.particle_files[:max_files]

        # Build lightweight index: (file_idx, event_idx, track_idx)
        self.index = []
        for file_idx, tf in enumerate(self.track_files):
            tbl = pq.read_table(tf, columns=["track_id"])
            for event_idx in range(len(tbl)):
                n_tracks = len(tbl["track_id"][event_idx].as_py())
                for track_idx in range(n_tracks):
                    self.index.append((file_idx, event_idx, track_idx))

        # Simple dict-based file cache (pickle-safe for DataLoader workers)
        self._file_cache_size = file_cache_size
        self._file_cache = {}
        self._file_cache_order = []

        # Compute normalization stats
        self._input_mean = None
        self._input_std = None
        self._output_scales = None
        self._compute_normalization()

    def _load_file(self, file_idx):
        """Load file with simple LRU dict cache (pickle-safe)."""
        if file_idx in self._file_cache:
            return self._file_cache[file_idx]
        # Evict oldest if full
        while len(self._file_cache) >= self._file_cache_size:
            oldest = self._file_cache_order.pop(0)
            self._file_cache.pop(oldest, None)
        result = (
            pq.read_table(self.track_files[file_idx]),
            pq.read_table(self.hit_files[file_idx]),
            pq.read_table(self.particle_files[file_idx]),
        )
        self._file_cache[file_idx] = result
        self._file_cache_order.append(file_idx)
        return result

    def _get_event_arrays(self, file_idx, event_idx):
        """Cached per-event numpy arrays."""
        key = (file_idx, event_idx)
        if getattr(self, "_event_cache_key", None) == key:
            return self._event_cache_val

        tracks_tbl, hits_tbl, particles_tbl = self._load_file(file_idx)

        def to_np(col, idx, dtype=np.float32):
            return np.array(col[idx].as_py(), dtype=dtype)

        part_ids = to_np(particles_tbl["particle_id"], event_idx, np.int64)
        val = {
            "hit_x": to_np(hits_tbl["x"], event_idx),
            "hit_y": to_np(hits_tbl["y"], event_idx),
            "hit_z": to_np(hits_tbl["z"], event_idx),
            "hit_vol": to_np(hits_tbl["volume_id"], event_idx),
            "hit_lay": to_np(hits_tbl["layer_id"], event_idx),
            "hit_det": to_np(hits_tbl["detector"], event_idx),
            "part_px": to_np(particles_tbl["px"], event_idx),
            "part_py": to_np(particles_tbl["py"], event_idx),
            "part_pz": to_np(particles_tbl["pz"], event_idx),
            "part_charge": to_np(particles_tbl["charge"], event_idx),
            "part_d0": to_np(particles_tbl["perigee_d0"], event_idx),
            "part_z0": to_np(particles_tbl["perigee_z0"], event_idx),
            "pid_to_idx": {int(pid): i for i, pid in enumerate(part_ids)},
            "track_d0": tracks_tbl["d0"][event_idx].as_py(),
            "track_z0": tracks_tbl["z0"][event_idx].as_py(),
            "track_phi": tracks_tbl["phi"][event_idx].as_py(),
            "track_theta": tracks_tbl["theta"][event_idx].as_py(),
            "track_qop": tracks_tbl["qop"][event_idx].as_py(),
            "track_majpid": tracks_tbl["majority_particle_id"][event_idx].as_py(),
            "track_hit_ids": tracks_tbl["hit_ids"][event_idx].as_py(),
        }
        self._event_cache_key = key
        self._event_cache_val = val
        return val

    def _extract_track(self, file_idx, event_idx, track_idx):
        """Extract a single track. Returns dict or None if invalid."""
        ev = self._get_event_arrays(file_idx, event_idx)
        n_total_hits = len(ev["hit_x"])

        hids = ev["track_hit_ids"][track_idx]
        if not hids:
            return None
        hids = [h for h in hids if h < n_total_hits]
        if not hids:
            return None

        maj_pid = int(ev["track_majpid"][track_idx])
        if maj_pid not in ev["pid_to_idx"]:
            return None
        p_idx = ev["pid_to_idx"][maj_pid]

        if np.isnan(ev["part_d0"][p_idx]) or np.isnan(ev["part_z0"][p_idx]):
            return None
        if ev["part_charge"][p_idx] == 0:
            return None

        # Truth params (raw then model format)
        truth_raw = compute_truth_params_raw(
            ev["part_px"][p_idx], ev["part_py"][p_idx], ev["part_pz"][p_idx],
            ev["part_charge"][p_idx], ev["part_d0"][p_idx], ev["part_z0"][p_idx],
        )
        truth_model = raw_to_model_params(truth_raw)

        # Reco params (model format, NOT normalized — for comparison)
        reco_raw = np.array([
            ev["track_d0"][track_idx], ev["track_z0"][track_idx],
            ev["track_phi"][track_idx], ev["track_theta"][track_idx],
            ev["track_qop"][track_idx],
        ], dtype=np.float32)
        reco_model = raw_to_model_params(reco_raw)

        # Hit features: cylindrical + deltas (10 features)
        hids_arr = np.array(hids)
        features = compute_hit_features(
            ev["hit_x"], ev["hit_y"], ev["hit_z"],
            ev["hit_vol"], ev["hit_lay"], ev["hit_det"],
            hids_arr,
        )

        # Pad or truncate
        n_hits = min(len(features), self.max_hits)
        padded = np.zeros((self.max_hits, N_HIT_FEATURES), dtype=np.float32)
        mask = np.zeros(self.max_hits, dtype=bool)
        padded[:n_hits] = features[:n_hits]
        mask[:n_hits] = True

        return {
            "hit_features": padded,
            "truth_params": truth_model,
            "reco_params": reco_model,
            "padding_mask": mask,
            "n_hits": n_hits,
        }

    def _compute_normalization(self, n_samples=2000):
        """Compute input/output normalization from first n_samples tracks."""
        n = min(n_samples, len(self))
        all_features, all_truth = [], []

        for idx in range(n):
            fi, ei, ti = self.index[idx]
            sample = self._extract_track(fi, ei, ti)
            if sample is None:
                continue
            nh = sample["n_hits"]
            if nh > 0:
                all_features.append(sample["hit_features"][:nh])
            all_truth.append(sample["truth_params"])

        if not all_features:
            self._input_mean = np.zeros(N_HIT_FEATURES, dtype=np.float32)
            self._input_std = np.ones(N_HIT_FEATURES, dtype=np.float32)
            self._output_scales = np.ones(N_OUTPUT, dtype=np.float32)
            return

        feat = np.concatenate(all_features, axis=0)
        self._input_mean = feat.mean(axis=0).astype(np.float32)
        self._input_std = feat.std(axis=0).astype(np.float32)
        self._input_std[self._input_std < 1e-6] = 1.0  # prevent div by zero

        truth = np.stack(all_truth)
        # Output scales: [d0_std, z0_std, 1(sin), 1(cos), pi(theta), qop_std]
        self._output_scales = np.array([
            max(np.std(truth[:, 0]), 1e-6),  # d0
            max(np.std(truth[:, 1]), 1e-6),  # z0
            1.0,                              # sin_phi (already [-1,1])
            1.0,                              # cos_phi (already [-1,1])
            np.pi,                            # theta
            max(np.std(truth[:, 5]), 1e-6),  # qop
        ], dtype=np.float32)

    def get_norm_stats(self):
        """Return normalization stats for saving with checkpoint."""
        return {
            "input_mean": self._input_mean,
            "input_std": self._input_std,
            "output_scales": self._output_scales,
        }

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, event_idx, track_idx = self.index[idx]
        sample = self._extract_track(file_idx, event_idx, track_idx)

        if sample is None:
            for offset in range(1, 100):
                alt_idx = (idx + offset) % len(self.index)
                fi, ei, ti = self.index[alt_idx]
                sample = self._extract_track(fi, ei, ti)
                if sample is not None:
                    break
            if sample is None:
                sample = {
                    "hit_features": np.zeros((self.max_hits, N_HIT_FEATURES), dtype=np.float32),
                    "truth_params": np.zeros(N_OUTPUT, dtype=np.float32),
                    "reco_params": np.zeros(N_OUTPUT, dtype=np.float32),
                    "padding_mask": np.zeros(self.max_hits, dtype=bool),
                    "n_hits": 0,
                }

        # Normalize inputs
        feats = sample["hit_features"].copy()
        mask = sample["padding_mask"]
        feats[mask] = (feats[mask] - self._input_mean) / self._input_std

        # Normalize truth outputs
        truth_norm = sample["truth_params"] / self._output_scales

        return {
            "hit_features": torch.from_numpy(feats),
            "truth_params": torch.from_numpy(truth_norm),
            "reco_params": torch.from_numpy(sample["reco_params"]),  # NOT normalized
            "padding_mask": torch.from_numpy(mask),
            "n_hits": sample["n_hits"],
        }
