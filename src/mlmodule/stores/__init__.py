import functools
from typing import TypeVar

from mlmodule.models.types import ModelWithState
from mlmodule.states import StateKey
from mlmodule.stores.github import GitHUBReleaseStore

_T = TypeVar("_T", bound=ModelWithState)


@functools.lru_cache(1)  # There is only one entry
def Store() -> GitHUBReleaseStore:
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
    return GitHUBReleaseStore("LSIR", "mlmodule")


def load_pretrained_model(model: _T, training_id: str) -> _T:
    """Loads model state from MLModule store with the given `training_id`"""
    Store().load(model, StateKey(state_type=model.state_type, training_id=training_id))
    return model
