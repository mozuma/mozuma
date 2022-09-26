from unittest.mock import MagicMock

import pytest

from mozuma.callbacks.states import SaveModelState
from mozuma.states import StateKey, StateType
from mozuma.stores.local import LocalStateStore


@pytest.fixture()
def state_key():
    return StateKey(
        state_type=StateType(
            backend="test",
            architecture="test",
        ),
        training_id="test",
    )


@pytest.mark.parametrize("state_exist", [True, False])
def test_raise_error_if_state_exist(state_key, state_exist):
    # A mock store
    store = LocalStateStore(folder="")
    store.exists = MagicMock(return_value=state_exist)

    if state_exist:
        # If model state exists, check exception is raised
        # at initialisation
        with pytest.raises(ValueError) as exc_info:
            _ = SaveModelState(store=store, state_key=state_key)
            assert exc_info.type is ValueError
    else:
        # If not, check an exception is not raised
        try:
            callback = SaveModelState(store=store, state_key=state_key)
            assert callback is not None
        except ValueError as exc:
            assert False, f"SaveModelState raised an exception {exc}"


def test_store_is_called(state_key):
    # A mock store
    store = LocalStateStore(folder="")
    store.exists = MagicMock(return_value=False)
    store.save = MagicMock()

    callback = SaveModelState(store=store, state_key=state_key)

    # Call callback
    callback.save_model_state(None, None)

    # Check store `save` method is called
    assert store.save.called
