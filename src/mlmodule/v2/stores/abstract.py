import abc
import os
import warnings
from io import BytesIO
from typing import Generic, List, NoReturn, TypeVar

import boto3

from mlmodule.models.types import ModelWithState
from mlmodule.v2.states import StateKey, StateType

_ModelType = TypeVar("_ModelType", bound=ModelWithState)


class AbstractStateStore(abc.ABC, Generic[_ModelType]):
    """Interface to handle model state loading and saving

    See [states reference](states.md) for more information on state management.
    """

    @abc.abstractmethod
    def save(self, model: _ModelType, training_id: str) -> StateKey:
        """Saves the model state to the store

        Attributes:
            model (ModelWithState): Model to save
            training_id (str): Identifier for the training activity

        Returns:
            StateKey: The identifier for the state that has been created
        """
        return StateKey(state_type=model.state_type, training_id=training_id)

    @abc.abstractmethod
    def load(self, model: _ModelType, state_key: StateKey) -> None:
        """Loads the models weights from the store

        Attributes:
            model (ModelWithState): Model to update
            state_key (StateKey): The identifier for the state to load
        """
        if not model.state_type.is_compatible_with(state_key.state_type):
            warnings.warn(
                "The model state type is incompatible with the state key to load "
                f"{model.state_type} is not compatible with {state_key.state_type}.",
                RuntimeWarning,
            )

    @abc.abstractmethod
    def get_state_keys(self, state_type: StateType) -> List[StateKey]:
        """Lists the available states that are compatible with the given state type.

        Attributes:
            state_type (StateType): Used to filter the compatible state keys

        Example:
            This is used to list the pretrained weights for a given model.
            The following code gives all available state keys in `store` for the `model`.

            ```python
            keys = store.get_state_keys(model.state_type)
            ```
        """

    def exists(self, state_key: StateKey) -> bool:
        """Tests whether the state key exists in the current store

        Args:
            state_key (StateKey): The state key to test

        Returns:
            bool: `True` if state key exists or `False` otherwise
        """
        return state_key in self.get_state_keys(state_key.state_type)


class MLModuleModelStore(AbstractStateStore):
    """Default MLModule store with pretrained model states"""

    def _get_bucket(self):
        session = boto3.session.Session(
            aws_access_key_id=os.environ.get("MLMODULE_AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("MLMODULE_AWS_SECRET_ACCESS_KEY"),
            profile_name=os.environ.get("MLMODULE_AWS_PROFILE_NAME"),
        )
        s3 = session.resource("s3", endpoint_url="https://sos-ch-gva-2.exo.io")
        # Select lsir-public-assets bucket
        return s3.Bucket("lsir-public-assets")

    def save(self, model: _ModelType, training_id: str) -> NoReturn:
        raise NotImplementedError("MLModuleStore states are read-only")

    def load(self, model: ModelWithState, state_key: StateKey) -> None:
        """Loads the models weights from the store

        Attributes:
            model (ModelWithState): Model to update
            state_key (StateKey): The state identifier to load.

        Warning:
            Setting the `training_id` in the argument `state_id` is not supported and will raise an error.
        """
        # Making sure state_id is compatible with the model
        super().load(model, state_key=state_key)

        # S3 Bucket
        bucket = self._get_bucket()

        # Download state dict into BytesIO file
        f = BytesIO()
        bucket.Object(f"pretrained-models/{state_key.state_type}.pt").download_fileobj(
            f
        )

        # Set the model state
        f.seek(0)
        model.set_state(f.read())
