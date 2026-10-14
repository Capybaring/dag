"""Benchmark adapter for the Patch + exogenous fusion baseline."""

from ts_benchmark.baselines.dag.dag import DAG
from ts_benchmark.baselines.dag.models.PatchExoModel import PatchExoModel


MODEL_HYPER_PARAMS = {
    "d_model": 64,
    "d_ff": 128,
    "te_dim": 10,
    "patch_len": 48,
    "stride": 48,
    "dropout": 0.0,
    "batch_size": 16,
    "lr": 0.001,
    "lradj": "type3",
    "num_epochs": 50,
    "patience": 5,
    "loss": "MAE",
    "norm": True,
    "fusion_method": "",
    "infer_use_future": True,
}


class PatchExo(DAG):
    """Reuse DAG's training, scalers, losses and forecast evaluation.

    Selected by --model-name dag.PatchExo.
    The current adapter is for complete observations on regular time slots.
    PatchExoModel itself performs exogenous fusion and requires known future X.
    """

    def __init__(self, **kwargs):
        parameters = {**MODEL_HYPER_PARAMS, **kwargs}
        if not parameters["infer_use_future"]:
            raise ValueError("PatchExo requires known future exogenous values")
        if parameters["fusion_method"]:
            raise ValueError("Set fusion_method='' because PatchExo already fuses X")
        super().__init__(**parameters)

    @property
    def model_name(self):
        return "PatchExo"

    def _init_model(self):
        # series_dim and enc_in are set by forecast_fit after splitting Y/X.
        return PatchExoModel(self.config)

    def _process(self, input, target, input_mark, target_mark, exog_future=None):
        # The target Y and calendar marks are not passed to the predictor.
        # Regular-slot relative times are constructed inside PatchExoModel.
        output, auxiliary_loss = self.model(input, exog_future)
        result = {"output": output}
        if self.model.training:
            result["additional_loss"] = auxiliary_loss
        return result
