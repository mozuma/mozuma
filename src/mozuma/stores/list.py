import abc
from typing import List, NoReturn, TypeVar

from mozuma.models.types import ModelWithState
from mozuma.states import StateKey, StateType
from mozuma.stores.abstract import AbstractStateStore

_ModelType = TypeVar("_ModelType", bound=ModelWithState)


class AbstractListStateStore(AbstractStateStore[_ModelType]):
    """Helper to define a store from a fixed list of state keys.

    The subclasses should implement the following:

    * [`available_state_keys`][mozuma.stores.list.AbstractListStateStore.available_state_keys]
    * [`state_downloader`][mozuma.stores.list.AbstractListStateStore.state_downloader]
    """

    @abc.abstractproperty
    def available_state_keys(self) -> List[StateKey]:
        """List of available state keys for this store

        Returns:
            list(StateKey): All available state keys in the store
        """

    @abc.abstractmethod
    def state_downloader(self, model: _ModelType, state_key: StateKey) -> None:
        """Downloads and applies a state to a model

        Args:
            model (_ModelType): The model that will be used to load the state
            state_key (StateKey): The state key identifier
        """

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
