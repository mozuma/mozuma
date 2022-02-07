import abc
from io import BytesIO
from typing import Callable, Generic, List, TypeVar

import torch

from mlmodule.v2.base.predictions import BatchModelPrediction
from mlmodule.v2.torch.utils import save_state_dict_to_bytes

# Type of data of a batch passed to the forward function
_BatchType = TypeVar("_BatchType")


class TorchMlModule(torch.nn.Module, Generic[_BatchType]):
    """
    Module for Torch models.
    """

    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device

    @abc.abstractmethod
    def forward_predictions(
        self, batch: _BatchType
    ) -> BatchModelPrediction[torch.Tensor]:
        """Applies the module on a batch and returns all potentially interesting data point (features, labels...)"""

    def set_state(self, state: bytes) -> None:
        state_dict = torch.load(BytesIO(state), map_location=self.device)
        self.load_state_dict(state_dict)

    def get_state(self) -> bytes:
        return save_state_dict_to_bytes(self.state_dict())

    @abc.abstractmethod
    def get_dataset_transforms(self) -> List[Callable]:
        """Returns a callable that will by used to tranform input data into a Tensor passed to the forward function"""
