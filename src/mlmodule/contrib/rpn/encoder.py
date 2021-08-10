from typing import Tuple, List, TypeVar, Any, Union

import numpy as np
import torch
from PIL.Image import Image
from torchvision.transforms.transforms import Compose

from mlmodule.box import BBoxOutput, BBoxCollection
from mlmodule.contrib.rpn.transforms import RegionCrop, StandardTorchvisionRegionTransforms
from mlmodule.contrib.densenet.features import BaseDenseNetPretrainedFeatures
from mlmodule.torch.data.base import IndexedDataset
from mlmodule.torch.data.box import ApplyFunctionToPosition


""" Creates encodings for regions extracted from an image """


InputDatasetType = TypeVar('InputDatasetType',
                           bound=IndexedDataset[Tuple[Any, int], Any, Tuple[Image, BBoxOutput]])

ImageDatasetType = TypeVar('ImageDatasetType', bound=IndexedDataset[Any, Any, Union[Image, np.ndarray]])


class RegionEncoder(BaseDenseNetPretrainedFeatures):
    """ Computes encodings for regions using a pretrained DenseNet """

    def __init__(self, densenet_arch, dataset="imagenet", device=None):
        super().__init__(densenet_arch, dataset=dataset, device=device)

    @classmethod
    def prep_encodings(cls, image_dataset: ImageDatasetType, regions: List[BBoxCollection]) -> InputDatasetType:
        assert len(image_dataset) == len(regions)
        new_idx = []
        new_data = []
        for (index, img), boxes in zip(zip(image_dataset.indices, image_dataset.items), regions):
            for box_num, box in enumerate(boxes):
                new_idx.append((index, box_num))
                new_data.append((img, box))

        dataset = IndexedDataset[
            Tuple[Any, int], Tuple[Image, BBoxOutput], Tuple[Image, BBoxOutput]
        ](new_idx, new_data)
        dataset.transforms = [
            ApplyFunctionToPosition(Compose(image_dataset.transforms), pos=0)
        ] + dataset.transforms
        return dataset

    @classmethod
    def parse_encodings(cls, indices: List[Tuple[Any, int]],
                        encodings: List[BBoxOutput]) -> Tuple[List, List[BBoxCollection]]:
        """
        Parses the encodings for regions back into a list of images, represented by their index, and a list of
        collections of regions (containing the region features) for each image.

        :param indices: List of (image_index, region_index) for the images. The regions must be in order, i.e. of the
            form[(img1, 0), (img1, 1), ..., (img1, N1), (img2, 0), ..., (img2, N2), ..., (imgM, NM)]
        :param encodings: List of box outputs for each region, containing the extracted features. Must be in the same
            order as the indices.
        :return:
        """
        new_indices = []
        regions = []

        # The regions must be in order: [(img1, 0), (img1, 1), ..., (img1, N), (img2, 0), ...]
        for (idx, region_idx), box in zip(indices, encodings):
            if region_idx == 0:
                new_indices.append(idx)
                regions.append([box])
            else:
                regions[-1].append(box)

        return new_indices, regions

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

        # Extract boxes to give to the result handler - data.items: List[Tuple[Image, BBoxOutput]]
        data_boxes = [data.apply_transforms(item)[1] for item in data.items]
        result_handler_options = {'box_outputs': data_boxes}

        return super().bulk_inference(
            data, data_loader_options=data_loader_options, result_handler_options=result_handler_options, **opts
        )

    @classmethod
    def results_handler(
            cls, acc_results: Tuple[List[Tuple[Any, int]], List[BBoxOutput]],
            new_indices: List,
            new_output: torch.Tensor,
            box_outputs=None,
    ) -> Tuple[List[Tuple[Any, int]], List[BBoxOutput]]:
        """ Runs after the forward pass at inference

        :param acc_results: Holds a tuple with indices of image regions,
            which is a tuple containing the image index and
            the region index, and a list of region encodings
        :param new_indices: New indices for the current batch
        :param new_output: New inference output for the current batch
        :param box_outputs: List of boxes for the dataset
        :return:
        """
        # Dealing for the first call where acc_results is None
        output: List[BBoxOutput]
        indices, output = acc_results or ([], [])
        regions_processed = len(indices)

        # Converting to list
        new_image_indices = cls.tensor_to_python_list_safe(new_indices[0])
        new_box_indices = cls.tensor_to_python_list_safe(new_indices[1])
        assert len(new_image_indices) == len(new_box_indices)
        new_indices = list(zip(new_image_indices, new_box_indices))

        # Appending new indices
        indices += new_indices

        # Transform new output tensor to numpy array, add to corresponding BBoxOutput
        for i, encoding in enumerate(new_output):
            region_box = box_outputs[regions_processed + i]
            features = encoding.cpu().detach().numpy()
            output.append(BBoxOutput(
                bounding_box=region_box.bounding_box,  # Extracting two points
                probability=region_box.probability,
                features=features
            ))

        return indices, output


class DenseNet161ImageNetEncoder(RegionEncoder):

    def __init__(self, device=None):
        super().__init__("densenet161", dataset="imagenet", device=device)
