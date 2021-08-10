from typing import Any, Optional, TypeVar, Union

import numpy as np
from PIL.Image import Image
import torch

from mlmodule.box import BBoxCollection
from mlmodule.contrib.rpn.encoder import DenseNet161ImageNetEncoder, RegionEncoder
from mlmodule.contrib.rpn.rpn import RPN
from mlmodule.contrib.rpn.selector import CosineSimilarityRegionSelector
from mlmodule.torch.base import BaseTorchMLModule
from mlmodule.torch.data.base import IndexedDataset
from mlmodule.torch.mixins import DownloadPretrainedStateFromProvider, TorchPretrainedModuleMixin
from mlmodule.types import StateDict


InputDatasetType = TypeVar('InputDatasetType', bound=IndexedDataset[Any, Any, Union[Image, np.ndarray]])
OutputDatasetType = TypeVar('OutputDatasetType', bound=IndexedDataset[Any, Any, BBoxCollection])


class RegionFeatures(BaseTorchMLModule, TorchPretrainedModuleMixin, DownloadPretrainedStateFromProvider):

    state_dict_key = "pretrained-models/rpn/rf_dnim161_ga_rpn_x101_32x4d_fpn_1x_coco_20200220-c28d1b18.pth"

    def __init__(self, device: Optional[torch.device] = None):
        super().__init__(device=device)
        self.rpn = RPN(device=device)
        self.region_encoder = DenseNet161ImageNetEncoder(device=device)
        self.region_selector = CosineSimilarityRegionSelector(device=device)

    def get_default_pretrained_state_dict_from_provider(self) -> StateDict:
        state_dict = {}

        # Getting RPN
        state_dict.update({
            f"rpn.{key}": value for key, value in self.rpn.get_default_pretrained_state_dict_from_provider().items()
        })

        # Getting region encoder
        state_dict.update({
            f"region_encoder.{key}": value for key, value in self.region_encoder.get_default_pretrained_state_dict().items()
        })

        return state_dict

    def bulk_inference(self, data: InputDatasetType, regions_per_image=30, min_region_score=0.7, **_kwargs) -> OutputDatasetType:
        """Performs inference for all the given data points

        :param data:
        :param regions_per_image:
        :param min_region_score:
        :return:
        """
        saved_transforms = data.transforms.copy()
        indices, regions = self.rpn.bulk_inference(
            data, data_loader_options={'batch_size': 1},
            regions_per_image=regions_per_image, min_score=min_region_score
        )

        data.transforms = saved_transforms

        # Compute features for regions
        box_dataset = RegionEncoder.prep_encodings(data, regions)
        img_reg_indices, img_reg_encodings = self.region_encoder.bulk_inference(box_dataset)
        indices, box_collections = RegionEncoder.parse_encodings(img_reg_indices, img_reg_encodings)

        # Select regions based on cosine similarity
        box_dataset_w_features = IndexedDataset[str, BBoxCollection, BBoxCollection](indices, box_collections)
        box_dataset_w_features_selected = self.region_selector.bulk_inference(box_dataset_w_features)

        return box_dataset_w_features_selected
