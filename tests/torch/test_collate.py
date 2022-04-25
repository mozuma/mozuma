import numpy as np
import torch

from mlmodule.v2.torch.collate import (
    TorchMlModuleCollateFn,
    TorchMlModuleTrainingCollateFn,
)


def test_custom_collate():
    """Tests that the custom collect function

    This function does not converts the indices of a dataset to a torch.Tensor
    """
    collate_fn = TorchMlModuleCollateFn()

    data = [(1, np.array([1, 2])), (2, np.array([2, 3]))]

    collated_data = collate_fn(data)

    assert collated_data[0] == [1, 2]
    torch.testing.assert_equal(collated_data[1], torch.LongTensor([[1, 2], [2, 3]]))


def test_custom_trainin_collate():
    """Tests that the custom collect function for training

    This function does not converts the targets of a dataset to a torch.Tensor
    """
    collate_fn = TorchMlModuleTrainingCollateFn()

    data = [(np.array([1, 2]), 0), (np.array([2, 3]), 1)]

    collated_data = collate_fn(data)

    assert collated_data[1] == [0, 1]
    torch.testing.assert_equal(collated_data[0], torch.LongTensor([[1, 2], [2, 3]]))
