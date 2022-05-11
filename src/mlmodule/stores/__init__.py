import functools
import os
from typing import TypeVar

from mlmodule.models.types import ModelWithState
from mlmodule.states import StateKey
from mlmodule.stores.s3 import S3StateStore

_T = TypeVar("_T", bound=ModelWithState)


@functools.lru_cache(1)  # There is only one entry
def Store() -> S3StateStore:
    """MlModule model state store.

    Example:
        The store can be used to list available pre-trained states for a model

        ```python
        store = Store()
        states = store.get_state_keys(model.state_type)
        ```

        And load a given state to a model

        ```python
        store.load(model, state_key=states[0])
        ```
    """
    return S3StateStore(
        bucket="lsir-public-assets",
        session_kwargs=dict(
            aws_access_key_id=os.environ.get("MLMODULE_AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("MLMODULE_AWS_SECRET_ACCESS_KEY"),
            profile_name=os.environ.get("MLMODULE_AWS_PROFILE_NAME"),
        ),
        s3_endpoint_url="https://sos-ch-gva-2.exo.io",
        base_path="pretrained-models/",
    )


def load_pretrained_model(model: _T, training_id: str) -> _T:
    """Loads model state from MLModule store with the given `training_id`"""
    Store().load(model, StateKey(state_type=model.state_type, training_id=training_id))
    return model
