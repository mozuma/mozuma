import abc
from typing import Callable, List

from torch import nn


class TorchModel(nn.Module):

    @abc.abstractmethod
    def get_dataset_transforms(self) -> List[Callable]:
        """Returns a callable that will by used to tranform input data into a Tensor passed to the forward function"""
