import dataclasses
from logging import getLogger
from typing import Any, Optional

import ignite.distributed as idist

from mozuma.states import StateKey
from mozuma.stores.abstract import AbstractStateStore

_logger = getLogger(__name__)


@dataclasses.dataclass
class SaveModelState:
    """Simple callback to save model state.

    Attributes:
        store (AbstractStateStore): Object to handle model state saving
        state_key (StateKey): State identifier for the training activity.

    Warning:
        The use of this callback is limited to runners with distributed
        capabilities, such as `TorchTrainingRunner`.
    """

    store: AbstractStateStore = dataclasses.field()
    state_key: StateKey = dataclasses.field()

    def __post_init__(self) -> None:
        if self.store.exists(self.state_key):
            raise ValueError("Model state already exists!")

    @idist.one_rank_only()
    def save_model_state(
        self, model: Any, training_id_suffix: Optional[str] = None
    ) -> None:
        """Save model state by calling the state store

        Arguments:
            model (Any): The MoZuMa model to save
            training_id_suffix (str | None): Optional string to append to
                `state_key.training_id`
        """
        new_training_id = self.state_key.training_id
        if training_id_suffix:
            new_training_id += training_id_suffix

        _logger.debug("Call store save utility")
        self.store.save(model, new_training_id)
