"""
Encoder-only transformer for track parameter regression from hits.

Takes only hit positions and detector features as input (no reconstructed
track parameters) for a fair comparison with the Kalman filter.
"""

import math

import torch
import torch.nn as nn


class TrackTransformer(nn.Module):
    """Transformer encoder that regresses track parameters from hit sequences.

    Architecture:
        1. Hit embedding: Linear(input_dim, d_model)
        2. Learned positional encoding (by radius-sorted index)
        3. Learnable [CLS] token prepended
        4. Transformer encoder: n_layers, d_model, n_heads, d_ff
        5. MLP head on [CLS] output -> (5,) track parameters

    Args:
        d_model: Transformer hidden dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer encoder layers
        d_ff: Feedforward dimension
        max_hits: Maximum sequence length
        hit_pos_dim: Dimension of hit positions (default: 3 for x,y,z)
        hit_feat_dim: Dimension of hit features (default: 3 for vol,lay,det)
        output_dim: Number of output parameters (default: 5 for d0,z0,phi,theta,qop)
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model=64,
        n_heads=4,
        n_layers=4,
        d_ff=256,
        max_hits=20,
        hit_pos_dim=3,
        hit_feat_dim=3,
        output_dim=5,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_hits = max_hits

        # Hit embedding: project (pos + features) to d_model
        input_dim = hit_pos_dim + hit_feat_dim
        self.hit_embed = nn.Linear(input_dim, d_model)

        # Learned positional encoding (max_hits + 1 for CLS token)
        self.pos_embed = nn.Embedding(max_hits + 1, d_model)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Output MLP head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, output_dim),
        )

    def forward(self, hit_positions, hit_features, padding_mask):
        """Forward pass.

        Args:
            hit_positions: (B, max_hits, 3) float — hit x, y, z
            hit_features: (B, max_hits, 3) float — volume_id, layer_id, detector
            padding_mask: (B, max_hits) bool — True for real hits

        Returns:
            (B, 5) float — predicted track parameters [d0, z0, phi, theta, qop]
        """
        B = hit_positions.shape[0]

        # Concatenate position and features
        x = torch.cat([hit_positions, hit_features], dim=-1)  # (B, max_hits, 6)
        x = self.hit_embed(x)  # (B, max_hits, d_model)

        # Add positional encoding (positions 1..max_hits for hits)
        hit_pos_ids = torch.arange(1, self.max_hits + 1, device=x.device)
        x = x + self.pos_embed(hit_pos_ids).unsqueeze(0)

        # Prepend CLS token (position 0)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        cls = cls + self.pos_embed(torch.zeros(1, dtype=torch.long, device=x.device))
        x = torch.cat([cls, x], dim=1)  # (B, 1+max_hits, d_model)

        # Build attention mask: CLS can attend to everything, hits respect padding
        # PyTorch TransformerEncoder uses key_padding_mask where True = ignore
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        key_padding_mask = torch.cat([cls_mask, ~padding_mask], dim=1)  # (B, 1+max_hits)

        # Transformer
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        # Extract CLS output
        cls_out = x[:, 0]  # (B, d_model)

        # Predict parameters
        return self.head(cls_out)  # (B, 5)
