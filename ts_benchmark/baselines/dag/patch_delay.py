"""Benchmark adapter for PatchDelayModel, reusing the PatchExo pipeline."""

from ts_benchmark.baselines.dag.patch_exo import PatchExo
from ts_benchmark.baselines.dag.models.PatchDelayModel import PatchDelayModel


class PatchDelay(PatchExo):
    def __init__(self, **kwargs):
        parameters = {
            "delay_context": 4,
            "use_delay": True,
            **kwargs,
        }
        super().__init__(**parameters)

    @property
    def model_name(self):
        return "PatchDelay"

    def _init_model(self):
        return PatchDelayModel(self.config)
