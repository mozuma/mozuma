import abc
import os
from io import BytesIO
from typing import NoReturn, cast

from typing_extensions import Protocol, runtime_checkable

import boto3


@runtime_checkable
class ModelWithState(Protocol):
    """Identifies a model that has state that can be set and gotten"""
    # Unique identifier for the model
    mlmodule_model_uri: str

    def set_state(self, state: bytes, **options) -> None:
        ...

    def get_state(self, **options) -> bytes:
        ...


class ModelWithStateFromProvider(Protocol):
    """Set the model state from data provided by the model author."""

    def set_state_from_provider(self, **options) -> None:
        ...


class AbstractModelStore(abc.ABC):
    """Interface between model state store and the model architecture"""

    @abc.abstractmethod
    def save(self, model: ModelWithState, **options) -> None:
        """Saves the model to the binary file handler"""

    @abc.abstractmethod
    def load(self, model: ModelWithState, **options) -> None:
        """Loads the models weights from the binary file"""


class MLModuleModelStore(AbstractModelStore):
    """Default MLModule store with model states stored in a S3 bucket"""

    def save(self, model: ModelWithState, **options) -> NoReturn:
        raise ValueError("MLModuleStore states are read-only")

    def load(self, model: ModelWithState, **options) -> None:
        """Reads the model weights from LSIR public assets S3"""
        session = boto3.session.Session(
            aws_access_key_id=os.environ.get("MLMODULE_AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("MLMODULE_AWS_SECRET_ACCESS_KEY"),
            profile_name=os.environ.get("MLMODULE_AWS_PROFILE_NAME")
        )
        s3 = session.resource(
            "s3",
            endpoint_url="https://sos-ch-gva-2.exo.io"
        )
        # Select lsir-public-assets bucket
        b = s3.Bucket("lsir-public-assets")

        # Download state dict into BytesIO file
        f = BytesIO()
        b.Object(f"pretrained-models/{model.mlmodule_model_uri}").download_fileobj(f)

        # Set the model state
        f.seek(0)
        model.set_state(f.read(), **options)


class ProviderModelStore(AbstractModelStore):

    def save(self, model: ModelWithState, **options) -> NoReturn:
        raise ValueError("ProviderModelStore states are read-only")

    def load(self, model: ModelWithState, **options) -> None:
        """Reads the model weights from LSIR public assets S3"""
        cast(ModelWithStateFromProvider, model).set_state_from_provider(**options)
