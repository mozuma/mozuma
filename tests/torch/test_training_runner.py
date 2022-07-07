import logging

import pytest
import torch

from mozuma.torch.modules import TorchModel
from mozuma.torch.runners import TorchTrainingRunner

LOGGER = logging.getLogger(__name__)


class TorchTestModule(TorchModel[torch.Tensor, torch.Tensor]):
    def __init__(self, is_trainable: bool = True):
        super().__init__(is_trainable=is_trainable)


@pytest.mark.parametrize("is_trainable", [False, True])
def test_warning_if_not_trainable(caplog, is_trainable):
    caplog.set_level(logging.WARNING)

    # Getting a test model
    model = TorchTestModule(is_trainable)

    # Check warning if model not trainable
    _ = TorchTrainingRunner(model=model, dataset=(), callbacks=[], options=None)
    if not is_trainable:
        assert "is not trainable!" in caplog.text
    else:
        assert caplog.text == ""
