"""
PyTorch datasets for the beam spot study.

Loads ColliderML parquet files lazily — only the requested file is read per
batch, with an LRU cache to avoid reloading the same file repeatedly.
"""

import glob
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


def compute_truth_params(px, py, pz, charge, perigee_d0, perigee_z0):
    """Compute truth track parameters from particle kinematics.

    Returns (d0, z0, phi, theta, qop) matching the ACTS convention.
    """
    pt = math.sqrt(px**2 + py**2)
    p = math.sqrt(px**2 + py**2 + pz**2)
    phi = math.atan2(py, px)
    theta = math.atan2(pt, pz)
    qop = charge / p if p > 0 else 0.0
    return np.array([perigee_d0, perigee_z0, phi, theta, qop], dtype=np.float32)


class TrackHitDataset(Dataset):
    """Lazy-loading dataset yielding per-track samples.

    Only builds a lightweight index at init (reads just the track_id column
    from each file). Actual data is loaded on demand per file, with an LRU
    cache so consecutive accesses to the same file are free.

    Each sample contains:
        hit_positions: (max_hits, 3) float32
        hit_features:  (max_hits, 3) float32
        truth_params:  (5,) float32
        reco_params:   (5,) float32
        padding_mask:  (max_hits,) bool
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

        # Build index: (file_idx, event_idx, track_idx_in_event)
        # Only reads the track_id column — very fast.
        self.index = []
        for file_idx, tf in enumerate(self.track_files):
            tbl = pq.read_table(tf, columns=["track_id"])
            for event_idx in range(len(tbl)):
                n_tracks = len(tbl["track_id"][event_idx].as_py())
                for track_idx in range(n_tracks):
                    self.index.append((file_idx, event_idx, track_idx))

        # Set up LRU cache with requested size
        self._load_file = lru_cache(maxsize=file_cache_size)(self._load_file_uncached)

    def _load_file_uncached(self, file_idx):
        """Load the three parquet tables for a given file index."""
        tracks = pq.read_table(self.track_files[file_idx])
        hits = pq.read_table(self.hit_files[file_idx])
        particles = pq.read_table(self.particle_files[file_idx])
        return tracks, hits, particles

    @staticmethod
    def _arrow_list_to_numpy(col, event_idx, dtype=np.float32):
        """Convert an Arrow list-column element to numpy without going through Python."""
        arr = col[event_idx]
        return np.array(arr.as_py(), dtype=dtype)

    def _get_event_arrays(self, file_idx, event_idx):
        """Get pre-parsed numpy arrays for an event, with caching."""
        key = (file_idx, event_idx)
        if getattr(self, "_event_cache_key", None) == key:
            return self._event_cache_val

        tracks_tbl, hits_tbl, particles_tbl = self._load_file(file_idx)
        to_np = self._arrow_list_to_numpy

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
        """Extract a single track sample.

        Returns a dict, or None if the track is invalid (no truth match, etc.).
        """
        ev = self._get_event_arrays(file_idx, event_idx)
        n_total_hits = len(ev["hit_x"])

        # Extract this track
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

        truth = compute_truth_params(
            ev["part_px"][p_idx], ev["part_py"][p_idx], ev["part_pz"][p_idx],
            ev["part_charge"][p_idx], ev["part_d0"][p_idx], ev["part_z0"][p_idx],
        )

        reco = np.array([
            ev["track_d0"][track_idx], ev["track_z0"][track_idx],
            ev["track_phi"][track_idx], ev["track_theta"][track_idx],
            ev["track_qop"][track_idx],
        ], dtype=np.float32)

        hids_arr = np.array(hids)
        positions = np.stack([
            ev["hit_x"][hids_arr], ev["hit_y"][hids_arr], ev["hit_z"][hids_arr],
        ], axis=1)
        radii = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
        sort_idx = np.argsort(radii)
        positions = positions[sort_idx]
        features = np.stack([
            ev["hit_vol"][hids_arr[sort_idx]],
            ev["hit_lay"][hids_arr[sort_idx]],
            ev["hit_det"][hids_arr[sort_idx]],
        ], axis=1)

        n_hits = min(len(positions), self.max_hits)
        padded_pos = np.zeros((self.max_hits, 3), dtype=np.float32)
        padded_feat = np.zeros((self.max_hits, 3), dtype=np.float32)
        mask = np.zeros(self.max_hits, dtype=bool)
        padded_pos[:n_hits] = positions[:n_hits]
        padded_feat[:n_hits] = features[:n_hits]
        mask[:n_hits] = True

        return {
            "hit_positions": padded_pos,
            "hit_features": padded_feat,
            "truth_params": truth,
            "reco_params": reco,
            "padding_mask": mask,
            "n_hits": n_hits,
        }

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, event_idx, track_idx = self.index[idx]
        sample = self._extract_track(file_idx, event_idx, track_idx)

        # If this track is invalid, try neighbours until we find a valid one
        if sample is None:
            for offset in range(1, 100):
                alt_idx = (idx + offset) % len(self.index)
                fi, ei, ti = self.index[alt_idx]
                sample = self._extract_track(fi, ei, ti)
                if sample is not None:
                    break
            if sample is None:
                # Last resort: return zeros (should essentially never happen)
                sample = {
                    "hit_positions": np.zeros((self.max_hits, 3), dtype=np.float32),
                    "hit_features": np.zeros((self.max_hits, 3), dtype=np.float32),
                    "truth_params": np.zeros(5, dtype=np.float32),
                    "reco_params": np.zeros(5, dtype=np.float32),
                    "padding_mask": np.zeros(self.max_hits, dtype=bool),
                    "n_hits": 0,
                }

        return {
            "hit_positions": torch.from_numpy(sample["hit_positions"]),
            "hit_features": torch.from_numpy(sample["hit_features"]),
            "truth_params": torch.from_numpy(sample["truth_params"]),
            "reco_params": torch.from_numpy(sample["reco_params"]),
            "padding_mask": torch.from_numpy(sample["padding_mask"]),
            "n_hits": sample["n_hits"],
        }

    def get_normalization_stats(self, n_samples=2000):
        """Estimate normalization stats from a subset of tracks.

        Samples sequentially from the start of the index for cache-friendliness.
        """
        n = min(n_samples, len(self))
        # Sequential sampling is much faster due to event cache locality
        indices = range(n)

        truths, recos, positions = [], [], []
        for idx in indices:
            s = self[idx]
            truths.append(s["truth_params"].numpy())
            recos.append(s["reco_params"].numpy())
            nh = s["n_hits"]
            if nh > 0:
                positions.append(s["hit_positions"][:nh].numpy())

        all_truth = np.stack(truths)
        all_reco = np.stack(recos)
        all_pos = np.concatenate(positions, axis=0)

        return {
            "truth_mean": all_truth.mean(axis=0),
            "truth_std": all_truth.std(axis=0),
            "reco_mean": all_reco.mean(axis=0),
            "reco_std": all_reco.std(axis=0),
            "hit_pos_mean": all_pos.mean(axis=0),
            "hit_pos_std": all_pos.std(axis=0),
        }
