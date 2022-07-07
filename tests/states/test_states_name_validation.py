from typing import Tuple

import pytest

from mozuma.states import StateKey, StateType


def test_state_key_valid_name():
    StateKey(StateType("pytorch", "torchresnet18"), "imagenet1")


def test_state_key_invalid_name():
    with pytest.raises(ValueError):
        StateKey(StateType("pytorch", "torchresnet18"), "imagenet/1")


def test_state_type_valid_name():
    StateType("pytorch", "torchresnet18", ("cls-1000",))


@pytest.mark.parametrize(
    ("backend", "architecture", "extra"),
    [
        ("pytorch/", "torchresnet18", ("cls1000",)),
        ("pytorch", "torchresnet18/", ("cls1000",)),
        ("pytorch", "torchresnet18", ("cls1000/",)),
    ],
)
def test_state_type_invalid_name(backend: str, architecture: str, extra: Tuple[str]):
    with pytest.raises(ValueError):
        StateType(backend, architecture, extra)
