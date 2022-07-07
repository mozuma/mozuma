import dataclasses
import os
from typing import List, TypeVar

from mozuma.models.types import ModelWithState
from mozuma.states import StateKey, StateType
from mozuma.stores.abstract import AbstractStateStore

_ModelType = TypeVar("_ModelType", bound=ModelWithState)


@dataclasses.dataclass
class LocalStateStore(AbstractStateStore[_ModelType]):
    """Local filebased store

    Attributes:
        folder (str): Path to the folder to save model's state
    """

    folder: str

    def _get_state_type_prefix(self, state_type: StateType) -> str:
        return f"{state_type.backend}.{state_type.architecture}"

    def _get_state_type_prefix_with_extra(self, state_type: StateType) -> str:
        if not state_type.extra:
            return self._get_state_type_prefix(state_type)
        extra = ".".join(state_type.extra)
        return f"{self._get_state_type_prefix(state_type)}.{extra}"

    def get_filename(self, state_key: StateKey) -> str:
        state_type_prefix = self._get_state_type_prefix_with_extra(state_key.state_type)
        return os.path.join(
            self.folder,
            f"{state_type_prefix}.{state_key.training_id}.pt",
        )

    def _list_files(self, prefix: str) -> List[str]:
        return [f for f in os.listdir(self.folder) if f.startswith(prefix)]

    def _filename_to_state_key(self, filename: str) -> StateKey:
        name_parts = filename[:-3].split(".")
        backend, architecture = name_parts[:2]
        extras = tuple(name_parts[2:-1])
        training_id = name_parts[-1]
        return StateKey(
            state_type=StateType(
                backend=backend, architecture=architecture, extra=extras
            ),
            training_id=training_id,
        )

    def get_state_keys(self, state_type: StateType) -> List[StateKey]:
        return [
            self._filename_to_state_key(f)
            for f in self._list_files(self._get_state_type_prefix(state_type))
        ]

    def save(self, model: _ModelType, training_id: str) -> StateKey:
        """Saves the model state to the local file

        Attributes:
            model (ModelWithState): Model to save
            training_id (str): Identifier for the training activity

        Returns:
            StateKey: The identifier for the model state that has been saved
        """
        filename = self.get_filename(
            StateKey(state_type=model.state_type, training_id=training_id)
        )
        if os.path.exists(filename):
            raise ValueError(f"File {filename} already exists.")

        with open(filename, mode="wb") as f:
            f.write(model.get_state())
        return super().save(model, training_id)

    def load(self, model: _ModelType, state_key: StateKey) -> None:
        """Loads the models weights from the local file

        Attributes:
            model (ModelWithState): Model to update
            state_key (StateKey): The state identifier to load
        """
        super().load(model, state_key)

        filename = self.get_filename(state_key)
        with open(filename, mode="rb") as f:
            model.set_state(f.read())
