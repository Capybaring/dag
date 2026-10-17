import torch
from torch import nn
from torch.nn import functional as F


class IrregularPatchEncoder(nn.Module):
    """Encode observations on retained regular time slots into patch tokens.

    values: [B, T, C], floating point values, normalized outside this module.
    times: [B, T] or [T], relative times in one shared coordinate system.
    mask: [B, T, C], True/1 for observed values; None means all observed.

    Returns tokens [B, C, K, d_model] and boolean patch_mask [B, C, K].
    Uses t-PatchGNN's learnable time embedding and TTCN aggregation.
    Patch boundaries depend on retained time slots, not observation counts.
    """

    def __init__(self, d_model=64, te_dim=10, patch_len=48, stride=48):
        super().__init__()
        if d_model < 2 or te_dim < 2:
            raise ValueError("d_model and te_dim must be at least 2")
        if not isinstance(patch_len, int) or not isinstance(stride, int):
            raise ValueError("patch_len and stride must be integers")
        if not 1 <= stride <= patch_len:
            raise ValueError("Require 1 <= stride <= patch_len")

        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride
        self.ttcn_dim = d_model - 1
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, te_dim - 1)
        input_dim = 1 + te_dim
        self.Filter_Generators = nn.Sequential(
            nn.Linear(input_dim, self.ttcn_dim),
            nn.ReLU(),
            nn.Linear(self.ttcn_dim, self.ttcn_dim),
            nn.ReLU(),
            nn.Linear(self.ttcn_dim, input_dim * self.ttcn_dim),
        )
        self.T_bias = nn.Parameter(torch.randn(1, self.ttcn_dim))

    def LearnableTE(self, times):
        linear = self.te_scale(times)
        periodic = torch.sin(self.te_periodic(times))
        return torch.cat([linear, periodic], dim=-1)

    def TTCN(self, features, mask):
        # features: [B*C*K, L, 1+te_dim]; mask: [B*C*K, L, 1]
        n, length, _ = features.shape
        logits = self.Filter_Generators(features)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

        # Finite masked logits avoid NaNs for empty patches. Explicitly
        # remove masked weights and renormalize, including empty patches.
        weights = F.softmax(logits, dim=1, dtype=torch.float32)
        weights = weights * mask.to(weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        weights = weights.to(features.dtype)
        weights = weights.reshape(n, length, self.ttcn_dim, -1)

        aggregated = (features.unsqueeze(-2) * weights).sum(dim=(1, 3))
        encoded = F.relu(aggregated + self.T_bias)
        patch_valid = mask.any(dim=1)
        encoded = encoded * patch_valid.to(encoded.dtype)
        return encoded, patch_valid

    def forward(self, values, times, mask=None):
        if values.ndim != 3 or not values.is_floating_point():
            raise ValueError("values must be floating point [B, T, C]")
        batch, length, channels = values.shape
        if min(batch, length, channels) < 1:
            raise ValueError("B, T and C must be positive")

        times = times.to(device=values.device, dtype=values.dtype)
        if times.ndim == 1 and times.shape[0] == length:
            times = times.unsqueeze(0).expand(batch, -1)
        if times.shape != (batch, length):
            raise ValueError("times must have shape [T] or [B, T]")
        if not torch.isfinite(times).all():
            raise ValueError("times must be finite, including missing slots")

        if mask is None:
            mask = torch.ones_like(values, dtype=torch.bool)
        else:
            if mask.shape != values.shape:
                raise ValueError("mask must have the same shape as values")
            mask = mask.to(device=values.device, dtype=torch.bool)

        # torch.where, rather than multiplication, also removes masked NaNs.
        values = torch.where(mask, values, torch.zeros_like(values))
        if not torch.isfinite(values).all():
            raise ValueError("Observed values must be finite")

        # Cover the final partial patch; padding does not count as observed.
        patches = (max(length - self.patch_len, 0) + self.stride - 1)
        patches = patches // self.stride + 1
        total_length = (patches - 1) * self.stride + self.patch_len
        pad = total_length - length

        x = F.pad(values.transpose(1, 2), (0, pad))
        m = F.pad(mask.transpose(1, 2), (0, pad), value=False)
        t = F.pad(times, (0, pad))
        x = x.unfold(-1, self.patch_len, self.stride)
        m = m.unfold(-1, self.patch_len, self.stride)
        t = t.unfold(-1, self.patch_len, self.stride)
        t = t.unsqueeze(1).expand(-1, channels, -1, -1)

        x = x.reshape(-1, self.patch_len, 1)
        m = m.reshape(-1, self.patch_len, 1)
        t = t.reshape(-1, self.patch_len, 1)
        features = torch.cat([x, self.LearnableTE(t)], dim=-1)
        encoded, patch_valid = self.TTCN(features, m)
        tokens = torch.cat([encoded, patch_valid.to(encoded.dtype)], dim=-1)

        tokens = tokens.reshape(batch, channels, patches, self.d_model)
        patch_mask = patch_valid.reshape(batch, channels, patches)
        return tokens, patch_mask

    def forward_patches(self, values, times, mask):
        """Encode t-PatchGNN-packed [B,K,L,C] variable observation lists.

        Patching is done using actual time intervals by the collator; the
        regular-slot forward path and all learnable parameters are unchanged.
        """
        if values.ndim != 4 or times.shape != values.shape or mask.shape != values.shape:
            raise ValueError('Packed values, times and mask must share [B,K,L,C]')
        if min(values.shape) < 1 or not values.is_floating_point():
            raise ValueError('Packed dimensions must be positive floating point values')
        mask = mask.to(device=values.device, dtype=torch.bool)
        times = times.to(values)
        values = torch.where(mask, values, torch.zeros_like(values))
        times = torch.where(mask, times, torch.zeros_like(times))
        if not torch.isfinite(values).all() or not torch.isfinite(times).all():
            raise ValueError('Observed values and times must be finite')
        b, k, length, c = values.shape
        x = values.permute(0, 3, 1, 2).reshape(-1, length, 1)
        t = times.permute(0, 3, 1, 2).reshape(-1, length, 1)
        m = mask.permute(0, 3, 1, 2).reshape(-1, length, 1)
        encoded, valid = self.TTCN(torch.cat([x, self.LearnableTE(t)], dim=-1), m)
        tokens = torch.cat([encoded, valid.to(encoded.dtype)], dim=-1)
        return tokens.reshape(b, c, k, self.d_model), valid.reshape(b, c, k)
