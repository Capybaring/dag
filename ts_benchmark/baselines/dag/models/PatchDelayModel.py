"""Add causal exogenous patch-window representations to PatchExoModel."""

import torch
from torch import nn

from ts_benchmark.baselines.dag.layers.DelayEncoder import DelayEncoder
from ts_benchmark.baselines.dag.models.PatchExoModel import PatchExoModel


class PatchDelayModel(PatchExoModel):
    """Patch + implicit delay + known future exogenous prediction.

    Keeps the baseline's history encoders, queries, channel attention and head.
    Adds shared window encoding to BOTH exogenous keys and values.
    All parameters are trained jointly using the existing Y prediction loss.

    This first version requires non-overlapping patches and a history length
    divisible by patch_len, so history/future patches share one time-slot grid.
    The final future patch may be partial. Default DIST settings (720/48/48)
    satisfy these requirements. Actual-time irregular patching is a later step.
    """

    def __init__(self, config):
        super().__init__(config)
        if self.stride != self.patch_len:
            raise ValueError("PatchDelay currently requires stride == patch_len")
        self.use_delay = bool(getattr(config, "use_delay", True))
        self.delay_encoder = DelayEncoder(
            self.d_model,
            delay_context=getattr(config, "delay_context", 4),
            hidden_dim=getattr(config, "delay_hidden_dim", 2 * self.d_model),
            dropout=getattr(config, "dropout", 0.0),
        )
        self.delay_key_projection = nn.Linear(
            self.d_model, self.d_model, bias=False
        )
        self.delay_value_projection = nn.Linear(
            self.d_model, self.d_model, bias=False
        )

    def _build_exogenous_key_value(self, H_x, M_x, H_future, M_future, future_len,
                                   query_indices=None):
        key, value, available = super()._build_exogenous_key_value(
            H_x, M_x, H_future, M_future, future_len, query_indices=query_indices
        )
        if not self.use_delay:
            return key, value, available

        # Sliding windows cross the history/future boundary without resetting.
        all_tokens = torch.cat([H_x, H_future], dim=2)
        all_masks = torch.cat([M_x, M_future], dim=2)
        delay, delay_valid = self.delay_encoder(all_tokens, all_masks)
        history_patches = H_x.shape[2]
        future_delay = delay[:, :, history_patches:]
        future_valid = delay_valid[:, :, history_patches:]
        patch_index = query_indices if query_indices is not None else self._future_patch_indices(
            future_len, H_future.shape[2], H_future.device
        )
        at_query = self._select_query_patches(future_delay, patch_index)
        valid_at_query = self._select_query_patches(future_valid, patch_index)

        key = key + self.delay_key_projection(at_query)
        value = value + self.delay_value_projection(at_query)
        # A past observed X can contribute even when the current patch is empty.
        available = available | valid_at_query
        return key, value, available

    def forward(
        self, input, exog_future, t_history=None, t_future=None,
        history_mask=None, future_mask=None,
    ):
        if input.ndim == 3 and input.shape[1] % self.patch_len != 0:
            raise ValueError("History length must be divisible by patch_len")
        return super().forward(
            input, exog_future, t_history=t_history, t_future=t_future,
            history_mask=history_mask, future_mask=future_mask,
        )
