"""
PyTorch datasets for the beam spot study.

Loads ColliderML parquet files (event-grouped list columns) and produces
per-track samples suitable for transformer-based track parameter regression.
"""

import glob
import math
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
    d0 = perigee_d0
    z0 = perigee_z0
    return np.array([d0, z0, phi, theta, qop], dtype=np.float32)


def load_parquet_dir(directory, pattern="*.parquet"):
    """Load and concatenate all parquet files in a directory."""
    files = sorted(glob.glob(str(Path(directory) / pattern)))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {directory}")
    tables = [pq.read_table(f) for f in files]
    return pq.concat_tables(tables)


class TrackHitDataset(Dataset):
    """Dataset yielding per-track samples with hit positions and truth parameters.

    Each sample contains:
        hit_positions: (max_hits, 3) float32 — x, y, z of digitized hits
        hit_features:  (max_hits, 3) float32 — volume_id, layer_id, detector (normalized)
        truth_params:  (5,) float32 — d0, z0, phi, theta, qop from truth
        reco_params:   (5,) float32 — d0, z0, phi, theta, qop from ACTS KF
        padding_mask:  (max_hits,) bool — True for real hits, False for padding
        n_hits:        int — number of real hits

    Args:
        parquet_base: Path to parquet directory (e.g., .../v1/parquet/)
        max_hits: Maximum number of hits per track (padded/truncated)
        file_indices: Optional list of file indices to load (for train/val/test splits)
    """

    def __init__(self, parquet_base, max_hits=20, file_indices=None):
        self.max_hits = max_hits
        self.samples = []
        self._load_data(parquet_base, file_indices)

    def _load_data(self, parquet_base, file_indices):
        parquet_base = Path(parquet_base)

        # Get file lists
        track_files = sorted(glob.glob(str(parquet_base / "reco/tracks/*.parquet")))
        hit_files = sorted(glob.glob(str(parquet_base / "reco/tracker_hits/*.parquet")))
        particle_files = sorted(glob.glob(str(parquet_base / "truth/particles/*.parquet")))

        if not track_files:
            raise FileNotFoundError(f"No track files in {parquet_base / 'reco/tracks/'}")
        if not hit_files:
            raise FileNotFoundError(f"No hit files in {parquet_base / 'reco/tracker_hits/'}")
        if not particle_files:
            raise FileNotFoundError(f"No particle files in {parquet_base / 'truth/particles/'}")

        # Filter by file index if specified
        if file_indices is not None:
            track_files = [track_files[i] for i in file_indices if i < len(track_files)]
            hit_files = [hit_files[i] for i in file_indices if i < len(hit_files)]
            particle_files = [particle_files[i] for i in file_indices if i < len(particle_files)]

        # Load all files
        for tf, hf, pf in zip(track_files, hit_files, particle_files):
            self._process_file(tf, hf, pf)

    def _process_file(self, track_file, hit_file, particle_file):
        """Process a single set of parquet files (tracks, hits, particles)."""
        tracks_tbl = pq.read_table(track_file)
        hits_tbl = pq.read_table(hit_file)
        particles_tbl = pq.read_table(particle_file)

        n_events = len(tracks_tbl)

        for evt_idx in range(n_events):
            self._process_event(tracks_tbl, hits_tbl, particles_tbl, evt_idx)

    def _process_event(self, tracks_tbl, hits_tbl, particles_tbl, evt_idx):
        """Extract per-track samples from a single event."""
        # Get hit arrays for this event
        hit_x = np.array(hits_tbl["x"][evt_idx].as_py(), dtype=np.float32)
        hit_y = np.array(hits_tbl["y"][evt_idx].as_py(), dtype=np.float32)
        hit_z = np.array(hits_tbl["z"][evt_idx].as_py(), dtype=np.float32)
        hit_vol = np.array(hits_tbl["volume_id"][evt_idx].as_py(), dtype=np.float32)
        hit_lay = np.array(hits_tbl["layer_id"][evt_idx].as_py(), dtype=np.float32)
        hit_det = np.array(hits_tbl["detector"][evt_idx].as_py(), dtype=np.float32)
        n_total_hits = len(hit_x)

        # Build particle lookup: particle_id -> index
        part_ids = np.array(particles_tbl["particle_id"][evt_idx].as_py())
        part_px = np.array(particles_tbl["px"][evt_idx].as_py(), dtype=np.float32)
        part_py = np.array(particles_tbl["py"][evt_idx].as_py(), dtype=np.float32)
        part_pz = np.array(particles_tbl["pz"][evt_idx].as_py(), dtype=np.float32)
        part_charge = np.array(particles_tbl["charge"][evt_idx].as_py(), dtype=np.float32)
        part_d0 = np.array(particles_tbl["perigee_d0"][evt_idx].as_py(), dtype=np.float32)
        part_z0 = np.array(particles_tbl["perigee_z0"][evt_idx].as_py(), dtype=np.float32)
        pid_to_idx = {int(pid): i for i, pid in enumerate(part_ids)}

        # Get track arrays for this event
        track_d0 = np.array(tracks_tbl["d0"][evt_idx].as_py(), dtype=np.float32)
        track_z0 = np.array(tracks_tbl["z0"][evt_idx].as_py(), dtype=np.float32)
        track_phi = np.array(tracks_tbl["phi"][evt_idx].as_py(), dtype=np.float32)
        track_theta = np.array(tracks_tbl["theta"][evt_idx].as_py(), dtype=np.float32)
        track_qop = np.array(tracks_tbl["qop"][evt_idx].as_py(), dtype=np.float32)
        track_majpid = np.array(tracks_tbl["majority_particle_id"][evt_idx].as_py())
        track_hit_ids = tracks_tbl["hit_ids"][evt_idx].as_py()
        n_tracks = len(track_d0)

        for trk_idx in range(n_tracks):
            # Get hit indices for this track
            hids = track_hit_ids[trk_idx]
            if not hids:
                continue

            # Filter valid hit indices
            hids = [h for h in hids if h < n_total_hits]
            if not hids:
                continue

            # Get truth particle
            maj_pid = int(track_majpid[trk_idx])
            if maj_pid not in pid_to_idx:
                continue
            p_idx = pid_to_idx[maj_pid]

            # Skip if truth perigee params are NaN
            if np.isnan(part_d0[p_idx]) or np.isnan(part_z0[p_idx]):
                continue
            if part_charge[p_idx] == 0:
                continue

            # Compute truth parameters
            truth = compute_truth_params(
                part_px[p_idx], part_py[p_idx], part_pz[p_idx],
                part_charge[p_idx], part_d0[p_idx], part_z0[p_idx]
            )

            # Reco parameters
            reco = np.array([
                track_d0[trk_idx], track_z0[trk_idx],
                track_phi[trk_idx], track_theta[trk_idx], track_qop[trk_idx]
            ], dtype=np.float32)

            # Extract hit positions and features
            hids_arr = np.array(hids)
            positions = np.stack([hit_x[hids_arr], hit_y[hids_arr], hit_z[hids_arr]], axis=1)

            # Sort by radius (innermost first)
            radii = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
            sort_idx = np.argsort(radii)
            positions = positions[sort_idx]

            features = np.stack([
                hit_vol[hids_arr[sort_idx]],
                hit_lay[hids_arr[sort_idx]],
                hit_det[hids_arr[sort_idx]],
            ], axis=1)

            # Pad or truncate
            n_hits = min(len(positions), self.max_hits)
            padded_pos = np.zeros((self.max_hits, 3), dtype=np.float32)
            padded_feat = np.zeros((self.max_hits, 3), dtype=np.float32)
            mask = np.zeros(self.max_hits, dtype=bool)

            padded_pos[:n_hits] = positions[:n_hits]
            padded_feat[:n_hits] = features[:n_hits]
            mask[:n_hits] = True

            self.samples.append({
                "hit_positions": padded_pos,
                "hit_features": padded_feat,
                "truth_params": truth,
                "reco_params": reco,
                "padding_mask": mask,
                "n_hits": n_hits,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "hit_positions": torch.from_numpy(s["hit_positions"]),
            "hit_features": torch.from_numpy(s["hit_features"]),
            "truth_params": torch.from_numpy(s["truth_params"]),
            "reco_params": torch.from_numpy(s["reco_params"]),
            "padding_mask": torch.from_numpy(s["padding_mask"]),
            "n_hits": s["n_hits"],
        }

    def get_normalization_stats(self):
        """Compute per-feature mean and std from the dataset."""
        all_truth = np.stack([s["truth_params"] for s in self.samples])
        all_reco = np.stack([s["reco_params"] for s in self.samples])

        # Hit position stats (only non-padded)
        all_pos = []
        for s in self.samples:
            n = s["n_hits"]
            all_pos.append(s["hit_positions"][:n])
        all_pos = np.concatenate(all_pos, axis=0)

        return {
            "truth_mean": all_truth.mean(axis=0),
            "truth_std": all_truth.std(axis=0),
            "reco_mean": all_reco.mean(axis=0),
            "reco_std": all_reco.std(axis=0),
            "hit_pos_mean": all_pos.mean(axis=0),
            "hit_pos_std": all_pos.std(axis=0),
        }
