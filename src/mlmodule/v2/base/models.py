from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class ModelWithState(Protocol):
    """Identifies a model that has state that can be set and gotten"""

    # Unique identifier for the model
    @property
    def mlmodule_model_uri(self) -> str:
        ...

    def set_state(self, state: bytes) -> None:
        ...

    def get_state(self) -> bytes:
        ...


class ModelWithStateFromProvider(ModelWithState):
    """Set the model state from data provided by the model author."""

    def set_state_from_provider(self) -> None:
        ...
