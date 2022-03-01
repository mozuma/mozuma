import torch

from mlmodule.v2.testing import ModuleTestConfiguration
from mlmodule.v2.torch.modules import TorchMlModule


def test_forward_random_tensor(torch_ml_module: ModuleTestConfiguration[TorchMlModule]):
    """Sends a random tensor to forward and makes sure it run without error"""
    model = torch_ml_module.get_module()
    batch = torch_ml_module.batch_input_type(*torch_ml_module.batch_input_shape)  # type: ignore
    assert model.forward(batch) is not None


def test_forward_predictions_random_tensor(
    torch_ml_module: ModuleTestConfiguration[TorchMlModule],
):
    """Sends a random tensor to forward and makes sure it run without error"""
    model = torch_ml_module.get_module()
    batch = torch_ml_module.batch_input_type(*torch_ml_module.batch_input_shape)  # type: ignore
    assert model.forward_predictions(batch) is not None
