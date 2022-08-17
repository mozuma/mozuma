from unittest.mock import MagicMock

import onnx
import pytest
import torch
import tempfile

from mozuma.predictions import BatchModelPrediction
from mozuma.states import StateType
from mozuma.torch.datasets import ListDataset
from mozuma.torch.modules import TorchModel
from mozuma.torch.options import TorchRunnerOptions
from mozuma.callbacks import CollectFeaturesInMemory
from mozuma.torch.runners import TorchInferenceRunner
from mozuma.onnx_runner import ONNXInferenceRunner


class TorchTestFeaturesModule(TorchModel[torch.Tensor, torch.Tensor]):
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


def test_export_to_onnx(tensor_features_dataset: ListDataset[torch.Tensor]):
    # Getting a test model
    model = TorchTestFeaturesModule()

    # A mock callback
    callback = MagicMock()

    # Inference runner
    runner = ONNXInferenceRunner(
        model=model,
        callbacks=[callback],
        dataset=tensor_features_dataset,
        options=TorchRunnerOptions(),
    )

    # export to tempfile
    with tempfile.NamedTemporaryFile() as tmp_onnx:
      runner.transfer(tmp_onnx.name)
      onnx_model = onnx.load(tmp_onnx.name)
      
      assert not onnx.checker.check_model(onnx_model)


def test_output_consistency(tensor_features_dataset: ListDataset[torch.Tensor]):
    # Getting a test model
    model = TorchTestFeaturesModule()

    # features callback
    features = CollectFeaturesInMemory()
    features_onnx = CollectFeaturesInMemory()

    # Inference runner
    runner = TorchInferenceRunner(
        model=model,
        callbacks=[features],
        dataset=tensor_features_dataset,
        options=TorchRunnerOptions(),
    )

    # ONNX Inference runner
    runner_onnx = ONNXInferenceRunner(
        model=model,
        callbacks=[features_onnx],
        dataset=tensor_features_dataset,
        options=TorchRunnerOptions(),
    )

    runner.run()
    runner_onnx.run()

    consistency = 0
    for feature, feature_onnx in zip(features.features, features_onnx.features):
      if feature.all() == feature_onnx.all():
        consistency += 1

    consistency_percent = int((consistency/len(features.features))*100)
  
    assert consistency_percent == 100
