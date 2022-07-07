from typing import Iterable, List, Tuple

import torch

from mozuma.models.vinvl.models.structures.image_list import to_image_list


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def transpose_batch(
        self, batch: Iterable[Tuple[torch.Tensor, Tuple[int, int]]]
    ) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
        tensors: List[torch.Tensor] = []
        sizes: List[Tuple[int, int]] = []
        for t, s in batch:
            tensors.append(t)
            sizes.append(s)
        return tensors, sizes

    def __call__(self, batch: List[Tuple[torch.Tensor, Tuple[int, int]]]):
        image_tensors, sizes = self.transpose_batch(batch)
        images = to_image_list(image_tensors, self.size_divisible)
        return (images, sizes)
