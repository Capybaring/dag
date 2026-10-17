"""t-PatchGNN data interface with the existing DAG/PatchDelay predictor."""
from ts_benchmark.baselines.dag.models.PatchDelayModel import PatchDelayModel


class IrregularPatchDelayModel(PatchDelayModel):
    def forward(self, batch):
        values, times, mask = (batch[k] for k in ('observed_data', 'observed_tp', 'observed_mask'))
        d = self.series_dim
        H_y, M_y = self.y_patch_encoder.forward_patches(values[..., :d], times[..., :d], mask[..., :d])
        H_x, M_x = self.x_patch_encoder.forward_patches(values[..., d:], times[..., d:], mask[..., d:])
        H_future, M_future = self.x_patch_encoder.forward_patches(
            batch['future_exog'], batch['future_exog_tp'], batch['future_exog_mask'])
        return self._predict_from_tokens(H_y, M_y, H_x, M_x, H_future, M_future,
                                         batch['tp_to_predict'], batch['query_patch_indices'])
