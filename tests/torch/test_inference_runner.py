from unittest.mock import MagicMock

import pytest
import torch

from mozuma.predictions import BatchModelPrediction
from mozuma.states import StateType
from mozuma.torch.datasets import ListDataset
from mozuma.torch.modules import TorchMlModule
from mozuma.torch.options import TorchRunnerOptions
from mozuma.torch.runners import TorchInferenceRunner


class TorchTestFeaturesModule(TorchMlModule[torch.Tensor, torch.Tensor]):
    @property
    def state_type(self) -> StateType:
        return StateType(backend="pytorch", architecture="test-module")

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return batch

    def to_predictions(
        self, forward_output: torch.Tensor
    ) -> BatchModelPrediction[torch.Tensor]:
        return BatchModelPrediction(features=forward_output)


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


def test_validate_data_loader_options_copy(
    tensor_features_dataset: ListDataset[torch.Tensor],
):
    """Makes sure that the options are not directly modified by the inference runner"""
    # Getting a test model
    model = TorchTestFeaturesModule()

    # Mock callback
    callback = MagicMock()

    # Options
    options = TorchRunnerOptions()
    assert options.data_loader_options == {}

    # Inference runner
    runner = TorchInferenceRunner(
        model=model,
        callbacks=[callback],
        dataset=tensor_features_dataset,
        options=options,
    )
    runner.run()

    assert options.data_loader_options == {}
