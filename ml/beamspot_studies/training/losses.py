"""Loss functions for track parameter regression."""

import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    """MSE loss with per-parameter weighting.

    The 5 track parameters (d0, z0, phi, theta, qop) have very different
    scales. This loss applies configurable weights to balance their
    contributions.
    """

    def __init__(self, weights=None):
        super().__init__()
        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, pred, target):
        """
        Args:
            pred: (B, 5) predicted parameters
            target: (B, 5) truth parameters
        Returns:
            scalar loss
        """
        diff_sq = (pred - target) ** 2  # (B, 5)
        weighted = diff_sq * self.weights.unsqueeze(0)  # (B, 5)
        return weighted.mean()


class NormalizedMSELoss(nn.Module):
    """MSE loss normalized by per-parameter variance from training data.

    This automatically balances the loss across parameters with different
    dynamic ranges.
    """

    def __init__(self, target_std):
        super().__init__()
        self.register_buffer("target_var", torch.tensor(target_std, dtype=torch.float32) ** 2)

    def forward(self, pred, target):
        diff_sq = (pred - target) ** 2
        normalized = diff_sq / self.target_var.unsqueeze(0).clamp(min=1e-8)
        return normalized.mean()
