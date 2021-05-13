from typing import Tuple, List, Union, TypeVar, Any

import numpy as np
import torch
from PIL.Image import Image

from mlmodule.box import BBoxPoint, BBoxOutput, BBoxCollection
from mlmodule.contrib.rpn.transforms import RegionCrop, StandardTorchvisionRegionTransforms
from mlmodule.contrib.densenet.features import BaseDenseNetPretrainedFeatures
from mlmodule.torch.data.base import IndexedDataset


""" Creates encodings for regions extracted from an image """


InputDatasetType = TypeVar('InputDatasetType',
                           bound=IndexedDataset[Tuple[Any, int], Any, Tuple[Image, BBoxOutput]])


class RegionEncoder(BaseDenseNetPretrainedFeatures):
    """ Computes encodings for regions using a pretrained DenseNet """

    def __init__(self, densenet_arch, dataset="imagenet", device=None):
        super().__init__(densenet_arch, dataset=dataset, device=device)

    @classmethod
    def prep_encoding(cls, indices: List, images: List[Image], regions: List[BBoxCollection]) -> InputDatasetType:
        assert len(indices) == len(images) == len(regions)
        new_idx = []
        new_data = []
        for index, img, boxes in zip(indices, images, regions):
            for box_num, box in enumerate(boxes):
                new_idx.append((index, box_num))
                new_data.append((img, box))

        return IndexedDataset[Tuple[Any, int], Tuple[Image, BBoxOutput], Tuple[Image, BBoxOutput]](new_idx, new_data)

    def get_dataset_transforms(self):
        return [
            RegionCrop(),
            StandardTorchvisionRegionTransforms(),
        ]

    def bulk_inference(self, data: InputDatasetType, data_loader_options=None, **opts):
        # Don't shuffle data as it will make it more complicated to regroup the boxes for images into a collection!
        # Force shuffle off
        data_loader_options = data_loader_options or {}
        data_loader_options['shuffle'] = False

        return super().bulk_inference(
            data, data_loader_options=data_loader_options, result_handler_options={}, **opts
        )

    @classmethod
    def results_handler(
            cls, acc_results: Tuple[List[Tuple[Any, int]], List[np.ndarray]],
            new_indices: List,
            new_output: torch.Tensor
    ) -> Tuple[List, List[BBoxCollection]]:
        """ Runs after the forward pass at inference

        :param acc_results: Holds a tuple with indices of image regions, which is a tuple containing the image index and
            the region index, and a list of region encodings
        :param new_indices: New indices for the current batch
        :param new_output: New inference output for the current batch
        :return:
        """
        # Dealing for the first call where acc_results is None
        output: List[torch.Tensor]
        indices, output = acc_results or ([], [])

        # Converting to list
        new_image_indices = cls.tensor_to_python_list_safe(new_indices[0])
        new_box_indices = cls.tensor_to_python_list_safe(new_indices[1])
        assert len(new_image_indices) == len(new_box_indices)
        new_indices = list(zip(new_image_indices, new_box_indices))

        indices += new_indices

        # Transform new output tensor to list of numpy arrays, add them to the output
        for encoding in new_output:
            output.append(encoding.cpu().detach().numpy())

        return indices, output


class DenseNet161ImageNetEncoder(RegionEncoder):

    def __init__(self, device=None):
        super().__init__("densenet161", dataset="imagenet", device=device)
