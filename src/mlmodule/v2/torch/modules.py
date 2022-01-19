import abc
from io import BytesIO
from typing import Callable, List

import torch

from mlmodule.v2.torch.utils import save_state_dict_to_bytes


class TorchMlModule(torch.nn.Module):
    """
    Module for Torch models.
    """

    def set_state(self, state: bytes, **options) -> None:
        map_location = options.get("device")
        state_dict = torch.load(BytesIO(state), map_location=map_location)
        self.load_state_dict(state_dict)

    def get_state(self, **options) -> bytes:
        return save_state_dict_to_bytes(self.state_dict())

    @abc.abstractmethod
    def get_dataset_transforms(self) -> List[Callable]:
        """Returns a callable that will by used to tranform input data into a Tensor passed to the forward function"""
