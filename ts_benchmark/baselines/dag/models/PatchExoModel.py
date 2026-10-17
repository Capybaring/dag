"""Patch + exogenous fusion baseline; explicit delay learning comes later."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ts_benchmark.baselines.dag.layers.IrregularPatchEncoder import (
    IrregularPatchEncoder,
)


class PatchExoModel(nn.Module):
    """Predict endogenous values using historical data and known future X.

    input: [B, T_history, series_dim + exog_dim], ordered [Y_history, X_history].
    exog_future: [B, T_future, exog_dim].
    Optional masks follow the corresponding value tensor shapes.
    Optional times: [T] or [B, T], in the same coordinate system and scale.

    Returns (prediction [B, T_future, series_dim], scalar zero auxiliary loss),
    matching the calling convention of the existing DAG adapter.
    Inputs and predictions use the dataset scaler's scale; no extra RevIN.
    """

    def __init__(self, config):
        super().__init__()
        self.series_dim = config.series_dim
        self.enc_in = config.enc_in
        self.exog_dim = self.enc_in - self.series_dim
        self.d_model = config.d_model
        self.patch_len = config.patch_len
        self.stride = config.stride
        if self.series_dim < 1 or self.exog_dim < 1:
            raise ValueError("At least one endogenous and exogenous variable required")

        te_dim = getattr(config, "te_dim", 10)
        dropout = getattr(config, "dropout", 0.0)
        d_ff = getattr(config, "d_ff", 2 * self.d_model)
        patch_args = dict(
            d_model=self.d_model,
            te_dim=te_dim,
            patch_len=self.patch_len,
            stride=self.stride,
        )
        # X history and future share this exact encoder instance.
        self.x_patch_encoder = IrregularPatchEncoder(**patch_args)
        self.y_patch_encoder = IrregularPatchEncoder(**patch_args)

        # Two causal temporal convolutions over historical Y patch tokens.
        self.y_conv1 = nn.Conv1d(self.d_model, self.d_model, kernel_size=3)
        self.y_conv2 = nn.Conv1d(
            self.d_model, self.d_model, kernel_size=3, dilation=2
        )
        self.dropout = nn.Dropout(dropout)
        self.target_embedding = nn.Embedding(self.series_dim, self.d_model)
        self.exog_embedding = nn.Embedding(self.exog_dim, self.d_model)
        self.x_history_projection = nn.Linear(
            self.d_model, self.d_model, bias=False
        )
        self.time_projection = nn.Linear(te_dim, self.d_model)
        self.query_projection = nn.Linear(self.d_model, self.d_model)
        self.key_projection = nn.Linear(self.d_model, self.d_model)
        self.value_projection = nn.Linear(self.d_model, self.d_model)
        self.forecast_head = nn.Sequential(
            nn.Linear(3 * self.d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1),
        )

    @staticmethod
    def _masked_patch_mean(tokens, patch_mask):
        weights = patch_mask.unsqueeze(-1).to(tokens.dtype)
        count = weights.sum(dim=2).clamp_min(1)
        return (tokens * weights).sum(dim=2) / count

    def _encode_y_history(self, tokens, patch_mask):
        batch, channels, patches, hidden = tokens.shape
        valid = patch_mask.reshape(batch * channels, 1, patches)
        x = tokens.reshape(batch * channels, patches, hidden).transpose(1, 2)
        x = self.dropout(F.gelu(self.y_conv1(F.pad(x, (2, 0)))))
        x = x * valid.to(x.dtype)
        x = self.dropout(F.gelu(self.y_conv2(F.pad(x, (4, 0)))))
        x = x * valid.to(x.dtype)
        x = x.transpose(1, 2).reshape(batch, channels, patches, hidden)
        return self._masked_patch_mean(x, patch_mask)

    @staticmethod
    def _prepare_times(times, batch, length, reference):
        times = times.to(device=reference.device, dtype=reference.dtype)
        if times.ndim == 1 and times.shape[0] == length:
            times = times.unsqueeze(0).expand(batch, -1)
        if times.shape != (batch, length):
            raise ValueError("Time tensor must have shape [T] or [B, T]")
        return times

    def _future_patch_indices(self, future_len, num_patches, device):
        indices = torch.arange(future_len, device=device) // self.stride
        return indices.clamp_max(num_patches - 1)

    @staticmethod
    def _select_query_patches(tokens, indices):
        if indices.ndim == 1:
            return tokens.index_select(2, indices)
        shape = [tokens.shape[0], tokens.shape[1], indices.shape[1]]
        index = indices[:, None, :].expand(*shape)
        if tokens.ndim == 4:
            index = index.unsqueeze(-1).expand(*shape, tokens.shape[-1])
        return tokens.gather(2, index)

    def _build_exogenous_key_value(self, H_x, M_x, H_future, M_future, future_len,
                                   query_indices=None):
        """Baseline fusion; subclasses can add a delay representation here."""
        x_state = self._masked_patch_mean(H_x, M_x)
        patch_index = query_indices if query_indices is not None else self._future_patch_indices(
            future_len, H_future.shape[2], H_future.device
        )
        x_at_query = self._select_query_patches(H_future, patch_index)
        x_available = self._select_query_patches(M_future, patch_index)
        x_at_query = (
            x_at_query
            + self.x_history_projection(x_state).unsqueeze(2)
            + self.exog_embedding.weight.unsqueeze(0).unsqueeze(2)
        )
        return (
            self.key_projection(x_at_query),
            self.value_projection(x_at_query),
            x_available,
        )

    def forward(
        self,
        input,
        exog_future,
        t_history=None,
        t_future=None,
        history_mask=None,
        future_mask=None,
    ):
        if input.ndim != 3 or input.shape[-1] != self.enc_in:
            raise ValueError("input must have shape [B, T_history, enc_in]")
        if not input.is_floating_point():
            raise ValueError("input must be floating point")
        batch, history_len, _ = input.shape
        if batch < 1 or history_len < 1:
            raise ValueError("Batch size and history length must be positive")
        if exog_future is None:
            raise ValueError("This baseline requires known future exogenous values")
        if (
            exog_future.ndim != 3
            or exog_future.shape[0] != batch
            or exog_future.shape[2] != self.exog_dim
            or exog_future.shape[1] < 1
        ):
            raise ValueError("exog_future must have shape [B, T_future, exog_dim]")
        exog_future = exog_future.to(device=input.device, dtype=input.dtype)
        future_len = exog_future.shape[1]

        # Automatic times are for the current retained regular-slot dataset.
        # Supply BOTH time tensors when using your own shared time scale.
        if (t_history is None) != (t_future is None):
            raise ValueError("Provide both t_history and t_future, or neither")
        if t_history is None:
            t_history = torch.arange(
                history_len, device=input.device, dtype=input.dtype
            ) / history_len
            t_future = torch.arange(
                history_len, history_len + future_len,
                device=input.device, dtype=input.dtype,
            ) / history_len
        t_history = self._prepare_times(t_history, batch, history_len, input)
        t_future = self._prepare_times(t_future, batch, future_len, input)

        if history_mask is None:
            history_mask = torch.ones_like(input, dtype=torch.bool)
        elif history_mask.shape != input.shape:
            raise ValueError("history_mask must match input shape")

        # The existing DAG data pipeline already orders history as [Y, X].
        y_history = input[:, :, :self.series_dim]
        x_history = input[:, :, self.series_dim:]
        mask_y = history_mask[:, :, :self.series_dim]
        mask_x = history_mask[:, :, self.series_dim:]
        H_y, M_y = self.y_patch_encoder(y_history, t_history, mask_y)
        H_x, M_x = self.x_patch_encoder(x_history, t_history, mask_x)
        H_future, M_future = self.x_patch_encoder(
            exog_future, t_future, future_mask
        )

        return self._predict_from_tokens(H_y, M_y, H_x, M_x, H_future, M_future, t_future)

    def _predict_from_tokens(self, H_y, M_y, H_x, M_x, H_future, M_future,
                             t_future, query_indices=None):
        """Shared DAG exogenous fusion/head for regular and packed IMTS input."""
        future_len = t_future.shape[1]
        y_state = self._encode_y_history(H_y, M_y)
        y_state = y_state + self.target_embedding.weight.unsqueeze(0)
        key, value, x_available = self._build_exogenous_key_value(
            H_x, M_x, H_future, M_future, future_len, query_indices=query_indices
        )

        time_features = self.time_projection(
            self.y_patch_encoder.LearnableTE(t_future.unsqueeze(-1))
        )
        y_at_query = y_state.unsqueeze(2).expand(-1, -1, future_len, -1)
        time_at_query = time_features.unsqueeze(1).expand(
            -1, self.series_dim, -1, -1
        )
        query = self.query_projection(y_at_query + time_at_query)

        # X-to-Y channel attention; key/value may include a subclass's delay.
        scores = torch.einsum("bjhd,bihd->bhji", query, key)
        scores = scores / math.sqrt(self.d_model)
        available = x_available.permute(0, 2, 1).unsqueeze(2)
        scores = scores.masked_fill(~available, torch.finfo(scores.dtype).min)
        attention = F.softmax(scores, dim=-1, dtype=torch.float32)
        attention = attention * available.to(attention.dtype)
        attention = attention / attention.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        exog_context = torch.einsum(
            "bhji,bihd->bjhd", attention.to(value.dtype), value
        )

        features = torch.cat(
            [y_at_query, exog_context, time_at_query], dim=-1
        )
        prediction = self.forecast_head(features).squeeze(-1).transpose(1, 2)
        return prediction.contiguous(), prediction.new_zeros(())
