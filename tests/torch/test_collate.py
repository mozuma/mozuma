import functools

import numpy as np
import torch
import torch.testing

from mozuma.torch.collate import TorchModelCollateFn

assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)


def test_custom_collate():
    """Tests that the custom collect function

    This function does not converts the indices of a dataset to a torch.Tensor
    """
    collate_fn = TorchModelCollateFn()

    data = [(1, np.array([1, 2])), (2, np.array([2, 3]))]

    collated_data = collate_fn(data)

    assert collated_data[0] == [1, 2]
    assert_equal(collated_data[1], torch.LongTensor([[1, 2], [2, 3]]))
