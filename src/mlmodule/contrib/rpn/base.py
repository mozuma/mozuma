from typing import Any, List, Optional, Tuple, TypeVar, Union

import numpy as np
from PIL.Image import Image
import torch

from mlmodule.box import BBoxCollection
from mlmodule.contrib.resnet.features import ResNet18ImageNetFeatures
from mlmodule.contrib.rpn.encoder import RegionEncoder
from mlmodule.contrib.rpn.rpn import RPN
from mlmodule.contrib.rpn.selector import CosineSimilarityRegionSelector
from mlmodule.metrics import MetricsCollector
from mlmodule.torch.base import BaseTorchMLModule
from mlmodule.torch.data.base import IndexedDataset
from mlmodule.torch.mixins import DownloadPretrainedStateFromProvider
from mlmodule.types import StateDict


InputDatasetType = TypeVar('InputDatasetType', bound=IndexedDataset[Any, Any, Union[Image, np.ndarray]])
OutputDatasetType = TypeVar('OutputDatasetType', bound=IndexedDataset[Any, Any, BBoxCollection])


class RegionFeatures(BaseTorchMLModule, DownloadPretrainedStateFromProvider):

    state_dict_key = "pretrained-models/rpn/rf_rnim18_ga_rpn_x101_32x4d_fpn_1x_coco_20200220-c28d1b18.pth"

    def __init__(self, device: Optional[torch.device] = None):
        super().__init__(device=device)
        self.rpn = RPN(device=device)
        self.region_encoder = RegionEncoder(encoder=ResNet18ImageNetFeatures(device=device), device=device)
        self.region_selector = CosineSimilarityRegionSelector(device=device)

    @property
    def metrics(self) -> MetricsCollector:
        m = super().metrics
        m.add_submetrics('rpn', self.rpn.metrics)
        m.add_submetrics('region_encoder', self.region_encoder.metrics)
        m.add_submetrics('region_selector', self.region_selector.metrics)
        return m

    def get_default_pretrained_state_dict_from_provider(self) -> StateDict:
        state_dict = {}

        # Getting RPN
        state_dict.update({
            f"rpn.{key}": value for key, value in self.rpn.get_default_pretrained_state_dict_from_provider().items()
        })

        # Getting region encoder
        state_dict.update({
            f"region_encoder.{key}": value
            for key, value in self.region_encoder.get_default_pretrained_state_dict().items()
        })

        return state_dict

    def bulk_inference(
        self, data: InputDatasetType, regions_per_image=30, min_region_score=0.7, **_kwargs
    ) -> Tuple[list, List[BBoxCollection]]:
        """Performs inference for all the given data points

        :param data:
        :param regions_per_image:
        :param min_region_score:
        :return:
        """
        self.metrics.add('dataset_size', len(data))

        saved_transforms = data.transforms.copy()

        with self.metrics.measure('time_region_proposal'):
            indices, regions = self.rpn.bulk_inference(
                data, data_loader_options={'batch_size': 1},
                regions_per_image=regions_per_image, min_score=min_region_score
            ) or ([], [])

        data.transforms = saved_transforms

        with self.metrics.measure('time_region_encoder'):
            # Compute features for regions
            box_dataset = RegionEncoder.prep_encodings(data, regions)
            img_reg_indices, img_reg_encodings = self.region_encoder.bulk_inference(box_dataset) or ([], [])
            indices, box_collections = RegionEncoder.parse_encodings(img_reg_indices, img_reg_encodings)

        with self.metrics.measure('time_region_selector'):
            # Select regions based on cosine similarity
            box_dataset_w_features = IndexedDataset[str, BBoxCollection, BBoxCollection](indices, box_collections)
            image_indices, region_features = self.region_selector.bulk_inference(box_dataset_w_features)

        # Ordering results by indices in a dictionary
        indexed_results = dict(list(zip(image_indices, region_features)))
        # Reconstructing the full list of indices passed in the input
        return data.indices, [indexed_results.get(i, []) for i in data.indices]
