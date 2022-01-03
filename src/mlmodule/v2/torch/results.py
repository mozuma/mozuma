import abc
import dataclasses
from typing import Generic, Optional, Tuple, TypeVar

import numpy as np
import torch

from mlmodule.torch.utils import tensor_to_python_list_safe

_ForwardOutput = TypeVar("_ForwardOutput")
_Results = TypeVar("_Results")


class AbstractResultsProcessor(abc.ABC, Generic[_ForwardOutput, _Results]):
    """Holds and processes results from the forward function of a Torch Module"""

    @abc.abstractmethod
    def process(self, indices: list, batch, forward_output: _ForwardOutput) -> None:
        """Results processing"""

    @abc.abstractmethod
    def get_results(self) -> Tuple[list, _Results]:
        """Returns the results values"""


@dataclasses.dataclass
class ArrayConcatResultsProcessor(AbstractResultsProcessor[torch.Tensor, np.ndarray]):
    indices: list = dataclasses.field(default_factory=list, init=False)
    array: Optional[np.ndarray] = dataclasses.field(default=None, init=False)

    def process(self, indices: list, batch, forward_output: torch.Tensor) -> None:
        """Generic result handler that stacks results and indices together

        :param acc_results: The results accumulated
        :param new_indices: The new indices of data for the current batch
        :param new_output: The new data for the current batch
        :return: new accumulated results
        """
        # Transforming the forward output to Numpy
        array_output: np.ndarray = forward_output.cpu().numpy()

        # Setting the array default shape
        if self.array is None:
            self.array = np.empty((0, array_output.shape[1]), dtype=array_output.dtype)

        # Adding results
        self.array = np.vstack((self.array, array_output))

        # Saving indices
        self.indices += tensor_to_python_list_safe(indices)

    def get_results(self) -> Tuple[list, np.ndarray]:
        return self.indices, self.array or np.array([])
