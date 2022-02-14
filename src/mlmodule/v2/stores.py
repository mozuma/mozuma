import abc
import dataclasses
import os
from io import BytesIO
from typing import NoReturn

import boto3

from mlmodule.v2.base.models import ModelWithState


class AbstractModelStore(abc.ABC):
    """Interface between model state store and the model architecture"""

    @abc.abstractmethod
    def save(self, model: ModelWithState) -> None:
        """Saves the model state to the store

        Attributes:
            model (ModelWithState): Model to save
        """

    @abc.abstractmethod
    def load(self, model: ModelWithState) -> None:
        """Loads the models weights from the store

        Attributes:
            model (ModelWithState): Model to update
        """


class MLModuleModelStore(AbstractModelStore):
    """Default MLModule store with pretrained model states"""

    def save(self, model: ModelWithState) -> NoReturn:
        raise ValueError("MLModuleStore states are read-only")

    def load(self, model: ModelWithState) -> None:
        session = boto3.session.Session(
            aws_access_key_id=os.environ.get("MLMODULE_AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("MLMODULE_AWS_SECRET_ACCESS_KEY"),
            profile_name=os.environ.get("MLMODULE_AWS_PROFILE_NAME"),
        )
        s3 = session.resource("s3", endpoint_url="https://sos-ch-gva-2.exo.io")
        # Select lsir-public-assets bucket
        b = s3.Bucket("lsir-public-assets")

        # Download state dict into BytesIO file
        f = BytesIO()
        b.Object(f"pretrained-models/{model.mlmodule_model_uri}").download_fileobj(f)

        # Set the model state
        f.seek(0)
        model.set_state(f.read())


@dataclasses.dataclass
class LocalFileModelStore(AbstractModelStore):
    """Local filebased store

    Attributes:
        filename (str): Path to the file to save model's state
    """
    filename: str

    def save(self, model: ModelWithState) -> None:
        """Saves the model state to the local file

        Attributes:
            model (ModelWithState): Model to save
        """
        with open(self.filename, mode="wb") as f:
            f.write(model.get_state())

    def load(self, model: ModelWithState) -> None:
        """Loads the models weights from the local file

        Attributes:
            model (ModelWithState): Model to update
        """
        with open(self.filename, mode="rb") as f:
            model.set_state(f.read())
