from typing import Tuple, List, Union, TypeVar, Any

import numpy as np
import torch
from PIL.Image import Image

from mlmodule.base import BaseMLModule
from mlmodule.box import BBoxPoint, BBoxOutput, BBoxCollection
from mlmodule.torch.data.base import IndexedDataset


""" Given regions extracted from images, filters some of them out to reduce redundancy """


InputDatasetType = TypeVar('InputDatasetType',
                           bound=IndexedDataset[Any, Any, Tuple[Union[Image, np.ndarray], BBoxCollection]])


class CosineSimilarityRegionSelector(BaseMLModule):
    """ Filters regions based on the cosine similarity between their encodings. """

    def __init__(self, device=None):
        super().__init__()
        self.device = device or self._resolve_device()

    @classmethod
    def _resolve_device(cls):
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    def bulk_inference(self, data: InputDatasetType):
        # TODO
        raise NotImplementedError()

    @classmethod
    def results_handler(cls, acc_results, new_indices, new_output):
        # TODO
        raise NotImplementedError()
