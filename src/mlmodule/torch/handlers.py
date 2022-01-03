from typing import List, Tuple, TypeVar, Union, cast

import numpy as np
import torch

from mlmodule.box import BBoxCollection, BBoxOutputArrayFormat
from mlmodule.torch.utils import tensor_to_python_list_safe

_IndicesType = TypeVar("_IndicesType")


def results_handler_numpy_array(
    acc_results: Tuple[List[_IndicesType], np.ndarray],
    new_indices: Union[torch.Tensor, List],
    new_output: torch.Tensor,
) -> Tuple[List[_IndicesType], np.ndarray]:
    """Generic result handler that stacks results and indices together

    :param acc_results: The results accumulated
    :param new_indices: The new indices of data for the current batch
    :param new_output: The new data for the current batch
    :return: new accumulated results
    """
    # Transforming new_output to Numpy
    new_output = new_output.cpu().numpy()

    # Collecting accumulated results or default value
    res_indices, res_output = acc_results or (
        [],
        np.empty((0, new_output.shape[1]), dtype=new_output.dtype),
    )

    # Adding indices
    res_indices += tensor_to_python_list_safe(new_indices)

    # Adding data
    res_output = np.vstack((cast(np.ndarray, res_output), new_output))
    return res_indices, res_output


def results_handler_bbox(
    acc_results: Tuple[List[_IndicesType], List[BBoxCollection]],
    new_indices: Union[torch.Tensor, List],
    new_output: BBoxOutputArrayFormat,
) -> Tuple[List[_IndicesType], List[BBoxCollection]]:
    """Runs after the forward pass at inference

    :param acc_results: Holds a tuple with indices, list of FacesFeatures namedtuple
    :param new_indices: New indices for the current batch
    :param new_output: New inference output for the current batch
    :return:
    """

    # Dealing for the first call where acc_results is None
    output: List[BBoxCollection]
    indices, output = acc_results or ([], [])

    # Converting to list
    new_indices = tensor_to_python_list_safe(new_indices)
    indices += new_indices

    # Adding the new output
    output += list(new_output)

    return indices, output
