from typing import cast
from unittest.mock import MagicMock

import pytest

from mlmodule.v2.states import StateKey, StateType
from mlmodule.v2.stores.s3 import S3StateStore


@pytest.fixture
def s3_store() -> S3StateStore:
    return S3StateStore(bucket="bucket", base_path="pretrained_models/")


def test_parse_state_key(s3_store: S3StateStore):
    state_key = "pretrained_models/pytorch/resnet18.cls1000.imagenet.pt"

    assert s3_store._parse_state_key(state_key) == StateKey(
        state_type=StateType(
            backend="pytorch", architecture="resnet18", extra=("cls1000",)
        ),
        training_id="imagenet",
    )


def test_state_type_prefix(s3_store: S3StateStore):
    state_type = StateType(
        backend="pytorch", architecture="resnet18", extra=("cls1000",)
    )

    assert s3_store._state_type_prefix(state_type) == "pytorch/resnet18"
    assert (
        s3_store._state_type_prefix_with_extra(state_type) == "pytorch/resnet18.cls1000"
    )


def test_get_state_keys(s3_store: S3StateStore):
    with MagicMock() as m:
        # Mocking the function
        s3_store._list_bucket_keys_by_prefix = m  # type: ignore
        m.return_value = []

        s3_store.get_state_keys(
            StateType(backend="pytorch", architecture="resnet18", extra=("cls1000",))
        )

        cast(MagicMock, s3_store._list_bucket_keys_by_prefix).assert_called_once_with(
            "pretrained_models/pytorch/resnet18"
        )
