from unittest.mock import MagicMock

import pytest
import torch

from mlmodule.v2.base.predictions import BatchModelPrediction
from mlmodule.v2.states import StateType
from mlmodule.v2.torch.datasets import ListDataset
from mlmodule.v2.torch.modules import TorchMlModule
from mlmodule.v2.torch.options import TorchRunnerOptions
from mlmodule.v2.torch.runners import TorchInferenceRunner


class TorchTestFeaturesModule(TorchMlModule[torch.Tensor, torch.Tensor]):
    @property
    def state_type(self) -> StateType:
        return StateType(backend="pytorch", architecture="test-module")

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return batch

    def forward_predictions(
        self, batch: torch.Tensor
    ) -> BatchModelPrediction[torch.Tensor]:
        return BatchModelPrediction(features=self.forward(batch))


@pytest.fixture
def tensor_features_dataset() -> ListDataset[torch.Tensor]:
    return ListDataset([torch.rand(512) for _ in range(10)])


def test_on_runner_end_callback(tensor_features_dataset: ListDataset[torch.Tensor]):
    # Getting a test model
    model = TorchTestFeaturesModule()

    # A mock callback
    callback = MagicMock()

    # Inference runner
    runner = TorchInferenceRunner(
        model=model,
        callbacks=[callback],
        dataset=tensor_features_dataset,
        options=TorchRunnerOptions(),
    )
    runner.run()

    assert callback.on_runner_end.called
