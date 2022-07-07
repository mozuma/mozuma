import dataclasses
from logging import getLogger
from typing import Any

import ignite.distributed as idist
from ignite.engine import Engine

from mozuma.states import StateKey
from mozuma.stores.abstract import AbstractStateStore

_logger = getLogger()


@dataclasses.dataclass
class SaveModelState:
    """Simple callback to save model state during training.

    If state are saved during training (every X epochs,
    see [`TorchTrainingOptions`][mozuma.torch.options.TorchTrainingOptions])
    the current epoch number is appended to the `state_key.training_id` in the
    following way: `<state_key.training_id>-e<num_epoch>`.
    When the training is complete, just the `state_key.training_id` is used.

    Attributes:
        store (AbstractStateStore): Object to handle model state saving
        state_key (StateKey): State identifier for the training activity.

    Warning:
        This callback only saves the model state, thus does not create a whole
        training checkpoint (optimizer state, loss, etc..).
    """

    store: AbstractStateStore = dataclasses.field()
    state_key: StateKey = dataclasses.field()

    def __post_init__(self) -> None:
        if self.store.exists(self.state_key):
            raise ValueError("Model state already exists!")

    @idist.one_rank_only()
    def save_model_state(self, engine: Engine, model: Any) -> None:
        """Save model state by calling the state store

        Arguments:
            model (Any): The model to save
        """
        epoch = engine.state.epoch
        _logger.debug(f"Calling store.save for epoch {epoch}")

        # Append current epoch number to training_id
        new_training_id = f"{self.state_key.training_id}-e{epoch}"

        # If the training is done instead, use the pure training_id,
        # without epoch information
        is_done_epochs = (
            engine.state.max_epochs is not None
            and engine.state.epoch >= engine.state.max_epochs
        )

        if is_done_epochs:
            new_training_id = self.state_key.training_id

            if not self.store.exists(self.state_key):
                self.store.save(model, new_training_id)

        else:
            self.store.save(model, new_training_id)
