import abc
import dataclasses
import os
from io import BytesIO
from typing import NoReturn, Optional

import boto3

from mlmodule.v2.base.models import ModelWithState
from mlmodule.v2.states import StateIdentifier


class AbstractModelStore(abc.ABC):
    """Interface between model state store and the model architecture"""

    def save(
        self, model: ModelWithState, training_id: Optional[str] = None
    ) -> StateIdentifier:
        """Saves the model state to the store

        Attributes:
            model (ModelWithState): Model to save
            training_id (Optional[str]): Identifier for the training activity

        Returns:
            StateIdentifier: The identifier for the model state that has been saved
        """
        return StateIdentifier(
            state_architecture=model.state_architecture(), training_id=training_id
        )

    @abc.abstractmethod
    def load(
        self, model: ModelWithState, state_id: Optional[StateIdentifier] = None
    ) -> None:
        """Loads the models weights from the store

        Attributes:
            model (ModelWithState): Model to update
            state_id (Optional[StateIdentifier]): Optionally pass the state identifier to load
        """
        if (
            state_id is not None
            and state_id.state_architecture
            not in model.compatible_state_architectures()
        ):
            raise ValueError(
                f"The state with architecture {state_id.state_architecture} cannot be loaded on {model}."
            )


class MLModuleModelStore(AbstractModelStore):
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

    def save(
        self, model: ModelWithState, training_id: Optional[str] = None
    ) -> NoReturn:
        raise ValueError("MLModuleStore states are read-only")

    def load(
        self, model: ModelWithState, state_id: Optional[StateIdentifier] = None
    ) -> None:
        """Loads the models weights from the store

        Attributes:
            model (ModelWithState): Model to update
            state_id (Optional[StateIdentifier]): Optionally pass the state identifier to load.

        Warning:
            Setting the `training_id` in the argument `state_id` is not supported and will raise an error.
        """
        # Making sure state_id is compatible with the model
        super().load(model, state_id=state_id)
        # Getting a default value for state id if not defined
        state_id = state_id or StateIdentifier(
            state_architecture=model.state_architecture()
        )
        if state_id.training_id is not None:
            raise ValueError(
                "Setting the training_id to load in MLModule store is not supported"
            )

        # S3 Bucket
        bucket = self._get_bucket()

        # Download state dict into BytesIO file
        f = BytesIO()
        bucket.Object(
            f"pretrained-models/{state_id.state_architecture}.pt"
        ).download_fileobj(f)

        # Set the model state
        f.seek(0)
        model.set_state(f.read())


@dataclasses.dataclass
class LocalFileModelStore(AbstractModelStore):
    """Local filebased store

    Attributes:
        folder (str): Path to the folder to save model's state
    """

    folder: str

    def get_filename(self, state_arch: str, training_id: Optional[str] = None) -> str:
        filename = os.path.join(self.folder, f"{state_arch}")
        if training_id is not None:
            return f"{filename}-{training_id}.pt"
        else:
            return f"{filename}.pt"

    def save(
        self, model: ModelWithState, training_id: Optional[str] = None
    ) -> StateIdentifier:
        """Saves the model state to the local file

        Attributes:
            model (ModelWithState): Model to save
            training_id (Optional[str]): Identifier for the training activity

        Returns:
            StateIdentifier: The identifier for the model state that has been saved
        """
        filename = self.get_filename(model.state_architecture(), training_id)
        if os.path.exists(filename):
            raise ValueError(f"File {filename} already exists.")

        with open(filename, mode="wb") as f:
            f.write(model.get_state())
        return super().save(model, training_id)

    def load(
        self, model: ModelWithState, state_id: Optional[StateIdentifier] = None
    ) -> None:
        """Loads the models weights from the local file

        Attributes:
            model (ModelWithState): Model to update
            state_id (Optional[StateIdentifier]): Optionally pass the state identifier to load
        """
        super().load(model, state_id)
        state_id = state_id or StateIdentifier(
            state_architecture=model.state_architecture()
        )

        filename = self.get_filename(state_id.state_architecture, state_id.training_id)
        with open(filename, mode="rb") as f:
            model.set_state(f.read())
