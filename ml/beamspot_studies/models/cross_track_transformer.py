"""Cross-track transformer that wraps a per-track TrackTransformer.

Architecture (Phase 6 of the beam spot study):

    Per-track encoder (existing TrackTransformer, warm-started)
        -> (B_ev, T, d_model) per-track CLS embeddings
        |
        v
    [EVT] token prepended + learnable cross-track positional encoding
        |
        v
    Cross-track TransformerEncoder (2 layers, 8 heads, d_ff=1024)
        -> (B_ev, 1+T, d_model)
        |
        v
    Extract updated per-track embeddings (drop [EVT])
        -> (B_ev, T, d_model)
        |
        v
    Per-track head (reused from TrackTransformer)
        -> (B_ev, T, output_dim)

The goal is for the cross-track encoder to discover the primary vertex from
the ensemble of tracks in the event and use it to condition the per-track
predictions — an implicit beam-spot constraint learned from data.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .track_transformer import TrackTransformer


class EventTrackTransformer(nn.Module):
    """Wraps a per-track TrackTransformer with a cross-track attention block.

    Args:
        track_model: an existing TrackTransformer whose encoder (and head) are
            reused. The cross-track block operates on its CLS outputs.
        n_cross_layers: number of TransformerEncoder layers in the cross-track
            block (default: 2).
        n_cross_heads: attention heads in the cross-track block (default: 8).
        cross_d_ff: FFN dim in the cross-track block (default: 4 * d_model).
        cross_dropout: dropout in the cross-track block (default: 0.1).
        max_tracks: max tracks per event for positional embedding sizing
            (default: 128).
    """

    def __init__(
        self,
        track_model: TrackTransformer,
        n_cross_layers: int = 2,
        n_cross_heads: int = 8,
        cross_d_ff: int | None = None,
        cross_dropout: float = 0.1,
        max_tracks: int = 128,
    ):
        super().__init__()
        self.track_model = track_model
        self.d_model = track_model.d_model
        self.max_tracks = max_tracks

        if cross_d_ff is None:
            cross_d_ff = 4 * self.d_model

        # Learnable [EVT] token prepended to the per-track embedding sequence.
        self.evt_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)

        # Positional embedding across tracks (slot 0 = EVT, slots 1..T = tracks).
        self.cross_pos_embed = nn.Embedding(max_tracks + 1, self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_cross_heads,
            dim_feedforward=cross_d_ff,
            dropout=cross_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.cross_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_cross_layers
        )

    def forward(
        self,
        hit_features: torch.Tensor,
        padding_mask: torch.Tensor,
        cls_features: torch.Tensor | None,
        track_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hit_features: (B_ev, T, H, F) event-level batched hit features
            padding_mask: (B_ev, T, H) bool — per-hit validity
            cls_features: (B_ev, T, C) per-track CLS summary features (or None)
            track_mask:   (B_ev, T) bool — True for real tracks, False for pad

        Returns:
            predictions: (B_ev, T, output_dim)
            track_mask:  (B_ev, T) — the same mask passed in, for convenience
        """
        B_ev, T, H, F = hit_features.shape
        assert T <= self.max_tracks, (
            f"max_tracks={self.max_tracks} but batch has T={T}"
        )

        # --- 1. Per-track encoder on flattened (B_ev*T, H, F) ---------------
        flat_hits = hit_features.reshape(B_ev * T, H, F)
        flat_mask = padding_mask.reshape(B_ev * T, H)
        flat_cls = cls_features.reshape(B_ev * T, -1) if cls_features is not None else None

        # Padded track slots have all-zero hits and all-False padding_mask.
        # `TrackTransformer.encode` handles any inputs — we'll mask them out
        # below before the cross-track encoder sees them.
        # To avoid nan from the per-track transformer attending to a sequence
        # with no real hits, we force at least one unmasked position for
        # padded slots (the CLS token is always real anyway). Concretely, we
        # set padding_mask[..., 0] = True for padded tracks so the encoder
        # sees a valid key_padding_mask. The output embedding of those slots
        # will be garbage, but we overwrite them with zeros before the
        # cross-track encoder runs.
        flat_track_valid = track_mask.reshape(B_ev * T)  # (B_ev*T,)
        safe_mask = flat_mask.clone()
        # For padded tracks, force at least one "real" hit slot so attention
        # doesn't degenerate. We use slot 0 as the synthetic real hit.
        safe_mask[~flat_track_valid, 0] = True

        track_embeddings = self.track_model.encode(
            flat_hits, safe_mask, flat_cls
        )  # (B_ev*T, d_model)

        # --- 2. Reshape back and zero padded track slots --------------------
        track_embeddings = track_embeddings.reshape(B_ev, T, self.d_model)
        track_embeddings = track_embeddings * track_mask.unsqueeze(-1).float()

        # --- 3. Prepend [EVT] token and add cross positional encoding ------
        evt = self.evt_token.expand(B_ev, -1, -1)  # (B_ev, 1, d_model)
        seq = torch.cat([evt, track_embeddings], dim=1)  # (B_ev, 1+T, d_model)

        pos_ids = torch.arange(seq.shape[1], device=seq.device)  # (1+T,)
        seq = seq + self.cross_pos_embed(pos_ids).unsqueeze(0)

        # Key padding mask: EVT token is always real (False = attend), track
        # slots follow track_mask.
        evt_mask = torch.zeros(B_ev, 1, dtype=torch.bool, device=seq.device)
        cross_kpm = torch.cat([evt_mask, ~track_mask], dim=1)  # True = ignore

        # --- 4. Cross-track transformer encoder -----------------------------
        seq = self.cross_encoder(seq, src_key_padding_mask=cross_kpm)

        # --- 5. Per-track head on updated embeddings -----------------------
        updated_track_emb = seq[:, 1:, :]  # drop [EVT] token
        flat_updated = updated_track_emb.reshape(B_ev * T, self.d_model)
        flat_pred = self.track_model.head(flat_updated)  # (B_ev*T, output_dim)
        output_dim = flat_pred.shape[-1]
        predictions = flat_pred.reshape(B_ev, T, output_dim)

        return predictions, track_mask

    @torch.no_grad()
    def extract_evt_embedding(
        self,
        hit_features: torch.Tensor,
        padding_mask: torch.Tensor,
        cls_features: torch.Tensor | None,
        track_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run forward up to the cross encoder and return the [EVT] token output.

        Used for diagnostic probes (e.g., does the [EVT] token linearly predict
        the true primary vertex?). Returns shape (B_ev, d_model).
        """
        B_ev, T, H, F = hit_features.shape
        flat_hits = hit_features.reshape(B_ev * T, H, F)
        flat_mask = padding_mask.reshape(B_ev * T, H)
        flat_cls = cls_features.reshape(B_ev * T, -1) if cls_features is not None else None

        flat_track_valid = track_mask.reshape(B_ev * T)
        safe_mask = flat_mask.clone()
        safe_mask[~flat_track_valid, 0] = True

        track_embeddings = self.track_model.encode(flat_hits, safe_mask, flat_cls)
        track_embeddings = track_embeddings.reshape(B_ev, T, self.d_model)
        track_embeddings = track_embeddings * track_mask.unsqueeze(-1).float()

        evt = self.evt_token.expand(B_ev, -1, -1)
        seq = torch.cat([evt, track_embeddings], dim=1)
        pos_ids = torch.arange(seq.shape[1], device=seq.device)
        seq = seq + self.cross_pos_embed(pos_ids).unsqueeze(0)

        evt_mask = torch.zeros(B_ev, 1, dtype=torch.bool, device=seq.device)
        cross_kpm = torch.cat([evt_mask, ~track_mask], dim=1)
        seq = self.cross_encoder(seq, src_key_padding_mask=cross_kpm)

        return seq[:, 0, :]  # (B_ev, d_model)
