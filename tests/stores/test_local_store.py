import os
from unittest.mock import MagicMock

import pytest

from mlmodule.v2.states import StateKey, StateType
from mlmodule.v2.stores.local import LocalStateStore


@pytest.fixture
def local_store() -> LocalStateStore:
    return LocalStateStore(folder="test")


@pytest.mark.parametrize(
    ("filename", "state_key"),
    [
        (
            "pytorch.resnet18.cls1000.imagenet.pt",
            StateKey(
                state_type=StateType(
                    backend="pytorch", architecture="resnet18", extra=("cls1000",)
                ),
                training_id="imagenet",
            ),
        ),
        (
            "pytorch.resnet18.imagenet.pt",
            StateKey(
                state_type=StateType(
                    backend="pytorch", architecture="resnet18", extra=None
                ),
                training_id="imagenet",
            ),
        ),
        (
            "pytorch.resnet18.cls1000.extra1.imagenet.pt",
            StateKey(
                state_type=StateType(
                    backend="pytorch",
                    architecture="resnet18",
                    extra=("cls1000", "extra1"),
                ),
                training_id="imagenet",
            ),
        ),
    ],
)
def test_parse_state_key(
    local_store: LocalStateStore, filename: str, state_key: StateKey
):
    assert local_store._filename_to_state_key(filename) == state_key


@pytest.mark.parametrize(
    ("state_type", "prefix", "extra_str"),
    [
        (
            StateType(backend="pytorch", architecture="resnet18", extra=("cls1000",)),
            "pytorch.resnet18",
            ".cls1000",
        ),
        (
            StateType(
                backend="pytorch",
                architecture="resnet18",
                extra=(
                    "cls1000",
                    "extra1",
                ),
            ),
            "pytorch.resnet18",
            ".cls1000.extra1",
        ),
        (
            StateType(backend="pytorch", architecture="resnet18", extra=None),
            "pytorch.resnet18",
            "",
        ),
    ],
)
def test_state_type_prefix(
    local_store: LocalStateStore, state_type: StateType, prefix: str, extra_str: str
):
    assert local_store._get_state_type_prefix(state_type) == prefix
    assert (
        local_store._get_state_type_prefix_with_extra(state_type)
        == f"{prefix}{extra_str}"
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
