import abc
from typing import List, NoReturn, TypeVar

from mlmodule.v2.base.models import ModelWithState
from mlmodule.v2.states import StateKey, StateType
from mlmodule.v2.stores.abstract import AbstractStateStore

_ModelType = TypeVar("_ModelType", bound=ModelWithState)


class ListStateStore(AbstractStateStore[_ModelType]):
    @abc.abstractproperty
    def available_state_keys(self) -> List[StateKey]:
        """List of available state keys"""

    @abc.abstractmethod
    def state_downloader(self, model: _ModelType, state_key: StateKey) -> None:
        """Downloads and applies a state to a model"""

    def get_state_keys(self, state_type: StateType) -> List[StateKey]:
        return [
            sk
            for sk in self.available_state_keys
            if sk.state_type.is_compatible_with(state_type)
        ]

    def load(self, model: _ModelType, state_key: StateKey) -> None:
        super().load(model, state_key)
        if state_key not in self.available_state_keys:
            raise ValueError("This state key does not exists on the store")

        self.state_downloader(model, state_key)

    def save(self, model: _ModelType, training_id: str) -> NoReturn:
        raise NotImplementedError(
            "This state store does not support saving new model states"
        )
