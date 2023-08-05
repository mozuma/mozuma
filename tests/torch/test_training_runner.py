import logging
from unittest.mock import MagicMock

import pytest
import torch
from test_inference_runner import TorchTestFeaturesModule

from mozuma.callbacks.states import SaveModelState
from mozuma.stores.local import LocalStateStore
from mozuma.torch.datasets import ListDataset, TorchTrainingDataset
from mozuma.torch.modules import TorchModel
from mozuma.torch.options import TorchTrainingOptions
from mozuma.torch.runners import TorchTrainingRunner

LOGGER = logging.getLogger(__name__)


class TorchTestModule(TorchModel[torch.Tensor, torch.Tensor]):
    def __init__(self, is_trainable: bool = True):
        super().__init__(is_trainable=is_trainable)


@pytest.fixture
def tensor_training_dataset() -> TorchTrainingDataset[int, torch.Tensor, torch.Tensor]:
    size = 10
    dataset = ListDataset([torch.rand(512) for _ in range(size)])
    return TorchTrainingDataset(
        dataset=dataset, targets=[torch.tensor([0]) for _ in range(size)]
    )


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


@pytest.mark.parametrize(
    "checkpoint_every", [0, 1], ids=["no_checkpoint_every", "checkpoint_every"]
)
def test_save_model_is_called(tensor_training_dataset, checkpoint_every):
    # Getting a test model
    model = TorchTestFeaturesModule()

    # Define dataset
    train_ds = tensor_training_dataset
    test_ds = tensor_training_dataset

    # Define callback
    store = LocalStateStore("")
    store.exists = MagicMock(return_value=False)

    callback = SaveModelState(store=store, state_key=MagicMock())
    callback.save_model_state = MagicMock()

    # Options
    options = TorchTrainingOptions(
        criterion=MagicMock(),
        optimizer=MagicMock(),
        num_epoch=1,
        validate_every=2,
        metrics={},
        checkpoint_every=checkpoint_every,
    )

    # Training runner
    runner = TorchTrainingRunner(
        model=model,
        dataset=(train_ds, test_ds),
        callbacks=[callback],
        options=options,
    )
    runner.run()

    # Check callback is called only once when `checkpoint_every` is not set
    # and times=checkpoint_every+1 otherwise
    if not checkpoint_every:
        callback.save_model_state.assert_called_once()
    else:
        callback.save_model_state.assert_called()
        assert callback.save_model_state.call_count == checkpoint_every + 1
