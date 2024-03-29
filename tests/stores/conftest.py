import dataclasses

import pytest

from mozuma.models.types import ModelWithState
from mozuma.states import StateType


@dataclasses.dataclass
class _DummyModelWithState:
    state: bytes = b"default"
    _state_type: StateType = StateType(backend="test", architecture="dummy")

    @property
    def state_type(self) -> StateType:
        return self._state_type

    def set_state(self, state: bytes) -> None:
        self.state = state

    def get_state(self) -> bytes:
        return self.state


@pytest.fixture
def model_with_state() -> ModelWithState:
    """Fixture that returns a model with state for testing purposes"""
    return _DummyModelWithState()
