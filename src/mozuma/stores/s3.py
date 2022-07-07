import dataclasses
from io import BytesIO
from typing import List, NoReturn, Optional, TypeVar

import boto3

from mozuma.models.types import ModelWithState
from mozuma.states import StateKey, StateType
from mozuma.stores.abstract import AbstractStateStore

_ModelType = TypeVar("_ModelType", bound=ModelWithState)


@dataclasses.dataclass
class S3StateStore(AbstractStateStore[_ModelType]):
    """State store on top of S3 object storage

    Given the state keys, states are organised in in folders with the following structure:

    ```shell
    base_path/
    ├─ {backend}/
    │  ├─ {architecture}.{extra1}.{extra2}.{training_id}.pt
    ├─ pytorch/     # e.g. for torch models
    │  ├─ resnet18.cls1000.imagenet.pt
    │  ├─ clip-image-rn50.clip.pt
    ```

    Attributes:
        bucket (str): Bucket to use to store states
        session_kwargs (dict, optional): Arguments passed to initialise `boto3.session.Session`
        s3_endpoint_url (str, optional): To connect to S3 compatible storage
        base_path (str, optional): The base path to store states
    """

    bucket: str
    session_kwargs: dict = dataclasses.field(default_factory=dict)
    s3_endpoint_url: Optional[str] = None
    base_path: str = ""

    def _get_bucket(self):
        session = boto3.session.Session(**self.session_kwargs)
        s3 = session.resource("s3", endpoint_url=self.s3_endpoint_url)
        return s3.Bucket(self.bucket)

    def _state_type_prefix(self, state_type: StateType) -> str:
        return f"{state_type.backend}/{state_type.architecture}"

    def _state_type_prefix_with_extra(self, state_type: StateType) -> str:
        if not state_type.extra:
            return self._state_type_prefix(state_type)
        state_type_extra_str = ".".join(state_type.extra)
        return f"{self._state_type_prefix(state_type)}.{state_type_extra_str}"

    def _parse_state_key(self, bucket_key: str) -> StateKey:
        # Removing the base path and extension
        base_path_l = len(self.base_path)
        bucket_key = bucket_key[base_path_l:-3]
        # Extracting backend
        backend, file_key = bucket_key.split("/")
        # Extracting architecture, extra and training id
        file_key_parts = file_key.split(".")
        architecture = file_key_parts[0]
        extra = tuple(file_key_parts[1:-1])
        training_id = file_key_parts[-1]
        return StateKey(
            state_type=StateType(
                backend=backend, architecture=architecture, extra=extra
            ),
            training_id=training_id,
        )

    def _list_bucket_keys_by_prefix(self, prefix: str) -> List[str]:
        # Listing available key under the given prefix
        return [s.key for s in self._get_bucket().objects.filter(Prefix=prefix)]

    def get_state_keys(self, state_type: StateType) -> List[StateKey]:
        # Listing keys
        available_states = self._list_bucket_keys_by_prefix(
            f"{self.base_path}{self._state_type_prefix(state_type)}"
        )
        return [self._parse_state_key(s) for s in available_states]

    def save(self, model: _ModelType, training_id: str) -> NoReturn:
        """Not implemented"""
        raise NotImplementedError("Store states are read-only")

    def load(self, model: _ModelType, state_key: StateKey) -> None:
        """Loads the models weights from the store

        Attributes:
            model (_ModelType): Model to update
            state_key (StateKey): The state identifier to load.
        """
        # Making sure state_id is compatible with the model
        super().load(model, state_key=state_key)

        # S3 Bucket
        bucket = self._get_bucket()

        # S3 object key for the model state
        s3_state_key = (
            f"{self.base_path}{self._state_type_prefix_with_extra(state_key.state_type)}"
            f".{state_key.training_id}.pt"
        )

        # Download state dict into BytesIO file
        f = BytesIO()
        bucket.Object(s3_state_key).download_fileobj(f)

        # Set the model state
        f.seek(0)
        model.set_state(f.read())
