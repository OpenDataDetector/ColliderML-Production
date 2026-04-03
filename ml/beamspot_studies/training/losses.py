"""Loss functions for track parameter regression."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrackHuberLoss(nn.Module):
    """Huber (smooth L1) loss for normalized track parameters.

    Since inputs and outputs are pre-normalized to ~unit scale, plain Huber
    with delta=1.0 gives roughly equal weighting across parameters.
    """

    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target):
        return F.smooth_l1_loss(pred, target, beta=self.delta)


class NormalizedMSELoss(nn.Module):
    """MSE loss normalized by per-parameter variance (legacy)."""

    def __init__(self, target_std):
        super().__init__()
        self.register_buffer("target_var", torch.tensor(target_std, dtype=torch.float32) ** 2)

    def forward(self, pred, target):
        diff_sq = (pred - target) ** 2
        normalized = diff_sq / self.target_var.unsqueeze(0).clamp(min=1e-8)
        return normalized.mean()
