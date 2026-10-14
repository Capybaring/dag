"""Learn an implicit delay pattern from ordered causal patch windows."""

import torch
from torch import nn


class DelayEncoder(nn.Module):
    """Encode [q-S+1, ..., q], including the current patch.

    tokens: [B, N_x, K, d_model].
    patch_mask: [B, N_x, K], True for patches containing observations.
    Returns representations [B, N_x, K, d_model] and window_valid [B, N_x, K].
    The same parameters are applied to every historical and future window.
    This is a learned window representation, not an explicit lag estimate.
    """

    def __init__(self, d_model, delay_context=4, hidden_dim=None, dropout=0.0):
        super().__init__()
        if not isinstance(delay_context, int) or delay_context < 1:
            raise ValueError("delay_context must be a positive integer")
        if d_model < 1:
            raise ValueError("d_model must be positive")
        self.d_model = d_model
        self.delay_context = delay_context
        hidden_dim = hidden_dim if hidden_dim is not None else 2 * d_model
        # Flattening preserves the order of patch offsets within the window.
        # An extra validity flag distinguishes a missing patch from zero data.
        self.network = nn.Sequential(
            nn.Linear(delay_context * (d_model + 1), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )

    def build_windows(self, tokens, patch_mask):
        if tokens.ndim != 4 or tokens.shape[-1] != self.d_model:
            raise ValueError("tokens must have shape [B, N_x, K, d_model]")
        if min(tokens.shape[:3]) < 1:
            raise ValueError("Batch, channel and patch counts must be positive")
        if patch_mask.shape != tokens.shape[:3]:
            raise ValueError("patch_mask must have shape [B, N_x, K]")
        patch_mask = patch_mask.to(device=tokens.device, dtype=torch.bool)
        tokens = torch.where(
            patch_mask.unsqueeze(-1), tokens, torch.zeros_like(tokens)
        )
        if not torch.isfinite(tokens).all():
            raise ValueError("Valid patch tokens must be finite")

        batch, channels, _, hidden = tokens.shape
        pad = self.delay_context - 1
        tokens = torch.cat(
            [tokens.new_zeros(batch, channels, pad, hidden), tokens], dim=2
        )
        patch_mask = torch.cat(
            [patch_mask.new_zeros(batch, channels, pad), patch_mask], dim=2
        )
        windows = tokens.unfold(2, self.delay_context, 1)
        windows = windows.permute(0, 1, 2, 4, 3)  # [B, N_x, K, S, d]
        window_mask = patch_mask.unfold(2, self.delay_context, 1)
        return windows, window_mask

    def forward(self, tokens, patch_mask):
        windows, window_mask = self.build_windows(tokens, patch_mask)
        features = torch.cat(
            [windows, window_mask.unsqueeze(-1).to(windows.dtype)], dim=-1
        ).flatten(start_dim=-2)
        representation = self.network(features)
        window_valid = window_mask.any(dim=-1)
        representation = representation * window_valid.unsqueeze(-1).to(
            representation.dtype
        )
        return representation, window_valid
