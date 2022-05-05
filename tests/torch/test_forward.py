import torch

from mlmodule.testing import ModuleTestConfiguration
from mlmodule.torch.modules import TorchMlModule


def test_forward_random_tensor(torch_ml_module: ModuleTestConfiguration[TorchMlModule]):
    """Sends a random tensor to forward and makes sure it run without error"""
    model = torch_ml_module.get_module()
    model.eval()
    batch = torch_ml_module.batch_factory()  # type: ignore
    with torch.no_grad():
        assert model.forward(batch) is not None


def test_forward_predictions_random_tensor(
    torch_ml_module: ModuleTestConfiguration[TorchMlModule],
):
    """Sends a random tensor to forward and makes sure it run without error"""
    model = torch_ml_module.get_module()
    model.eval()
    batch = torch_ml_module.batch_factory()  # type: ignore
    with torch.no_grad():
        assert model.to_predictions(model.forward(batch)) is not None
