import os
from unittest.mock import MagicMock

import pytest

from mlmodule.v2.states import StateKey, StateType
from mlmodule.v2.stores.local import LocalStateStore


@pytest.fixture
def local_store() -> LocalStateStore:
    return LocalStateStore(folder="test")


def test_parse_state_key(local_store: LocalStateStore):
    filename = "pytorch.resnet18.cls1000.imagenet.pt"

    assert local_store._filename_to_state_key(filename) == StateKey(
        state_type=StateType(
            backend="pytorch", architecture="resnet18", extra=("cls1000",)
        ),
        training_id="imagenet",
    )


def test_state_type_prefix(local_store: LocalStateStore):
    state_type = StateType(
        backend="pytorch", architecture="resnet18", extra=("cls1000",)
    )

    assert local_store._get_state_type_prefix(state_type) == "pytorch.resnet18"
    assert (
        local_store._get_state_type_prefix_with_extra(state_type)
        == "pytorch.resnet18.cls1000"
    )


def test_get_filename(local_store: LocalStateStore):
    state_key = StateKey(
        state_type=StateType(
            backend="pytorch", architecture="resnet18", extra=("cls1000",)
        ),
        training_id="imagenet",
    )
    assert local_store.get_filename(state_key) == os.path.join(
        "test", "pytorch.resnet18.cls1000.imagenet.pt"
    )


def test_get_state_keys(local_store: LocalStateStore):

    with MagicMock() as m:
        # Patching the os listdir function
        from mlmodule.v2.stores.local import os

        os.listdir = m
        m.return_value = [
            "pytorch.resnet18.cls1000.imagenet.pt",
            "pytorch.resnet18.cls2.imagenet.pt",
            "pytorch.clip.cls2.imagenet.pt",
        ]
        assert local_store.get_state_keys(
            StateType(backend="pytorch", architecture="resnet18", extra=("cls1000",))
        ) == [
            StateKey(
                StateType(
                    backend="pytorch", architecture="resnet18", extra=("cls1000",)
                ),
                training_id="imagenet",
            ),
            StateKey(
                StateType(backend="pytorch", architecture="resnet18", extra=("cls2",)),
                training_id="imagenet",
            ),
        ]
