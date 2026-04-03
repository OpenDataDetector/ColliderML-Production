"""
Encoder-only transformer for track parameter regression from hits.

Takes cylindrical hit coordinates + inter-hit deltas as input.
Outputs normalized [d0, z0, sin(phi), cos(phi), theta, qop].
"""

import torch
import torch.nn as nn


class TrackTransformer(nn.Module):
    """Transformer encoder that regresses track parameters from hit sequences.

    Args:
        d_model: Transformer hidden dimension (default: 128)
        n_heads: Number of attention heads (default: 8)
        n_layers: Number of encoder layers (default: 6)
        d_ff: Feedforward dimension (default: 512)
        max_hits: Maximum sequence length (default: 20)
        input_dim: Features per hit (default: 10)
        output_dim: Number of output parameters (default: 6)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        d_model=128,
        n_heads=8,
        n_layers=6,
        d_ff=512,
        max_hits=20,
        input_dim=10,
        output_dim=6,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_hits = max_hits

        self.hit_embed = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Embedding(max_hits + 1, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

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

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, output_dim),
        )

    def forward(self, hit_features, padding_mask):
        """
        Args:
            hit_features: (B, max_hits, input_dim) — normalized hit features
            padding_mask: (B, max_hits) bool — True for real hits

        Returns:
            (B, output_dim) — predicted normalized track parameters
        """
        B = hit_features.shape[0]

        x = self.hit_embed(hit_features)

        hit_pos_ids = torch.arange(1, self.max_hits + 1, device=x.device)
        x = x + self.pos_embed(hit_pos_ids).unsqueeze(0)

        cls = self.cls_token.expand(B, -1, -1)
        cls = cls + self.pos_embed(torch.zeros(1, dtype=torch.long, device=x.device))
        x = torch.cat([cls, x], dim=1)

        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        key_padding_mask = torch.cat([cls_mask, ~padding_mask], dim=1)

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        return self.head(x[:, 0])
