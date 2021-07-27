from typing import Any, Optional, TypeVar, Union

import numpy as np
from PIL.Image import Image
import torch

from mlmodule.base import BaseMLModule
from mlmodule.box import BBoxCollection
from mlmodule.contrib.rpn.encoder import DenseNet161ImageNetEncoder, RegionEncoder
from mlmodule.contrib.rpn.rpn import RPN
from mlmodule.contrib.rpn.selector import CosineSimilarityRegionSelector
from mlmodule.torch.base import BaseTorchMLModule
from mlmodule.torch.data.base import IndexedDataset


InputDatasetType = TypeVar('InputDatasetType', bound=IndexedDataset[Any, Any, Union[Image, np.ndarray]])
OutputDatasetType = TypeVar('OutputDatasetType', bound=IndexedDataset[Any, Any, BBoxCollection])


class RegionFeatures(BaseMLModule):

    def __init__(self, 
            rpn: Optional[BaseTorchMLModule] = None, 
            region_encoder: Optional[BaseTorchMLModule] = None, 
            region_selector: Optional[BaseMLModule] = None,
            device: Optional[torch.device] = None
    ):
        self.rpn = rpn or RPN(device=device).load()
        self.region_encoder = region_encoder or DenseNet161ImageNetEncoder(device=device).load()
        self.region_selector = region_selector or CosineSimilarityRegionSelector(device=device)

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
