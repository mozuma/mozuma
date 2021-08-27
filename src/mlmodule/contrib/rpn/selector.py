import dataclasses
import itertools
import logging
from typing import Any, List, Tuple, TypeVar
import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from mlmodule.base import BaseMLModule
from mlmodule.box import BBoxCollection
from mlmodule.torch.data.base import IndexedDataset


logger = logging.getLogger(__name__)

""" Given regions extracted from images, filters some of them out to reduce redundancy """


InputDatasetType = TypeVar('InputDatasetType', bound=IndexedDataset[Any, Any, BBoxCollection])


class CosineSimilarityRegionSelector(BaseMLModule):
    """ Filters regions based on the cosine similarity between their encodings. """

    def __init__(self, device=None):
        super().__init__()
        self.device = device or self._resolve_device()

    @classmethod
    def _resolve_device(cls):
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    @classmethod
    def similarity(cls, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.dot(x/np.linalg.norm((x)), y/np.linalg.norm(y))

    def get_data_loader(self, data):
        """Configured data loader with applied transforms

        :param data:
        :param data_loader_options:
        :return:
        """
        # Data loader options options
        data_loader_options = {
            "shuffle": False,
            "drop_last": False,
            "batch_size": 1,
            "pin_memory": self.device != torch.device('cpu'),
            "collate_fn": lambda x: ([x[0][0]], x[0][1])
        }
        # Building data loader
        return DataLoader(data, **data_loader_options)

    def bulk_inference(
            self, data: InputDatasetType, max_similarity=0.75, tqdm_enabled=False
    ) -> Tuple[list, List[BBoxCollection]]:
        indices: List[Any] = []
        bbox_collections: List[BBoxCollection] = []
        data_loader = self.get_data_loader(data)

        with torch.no_grad():
            n_batches = len(data_loader)
            if tqdm_enabled:
                data_loader = tqdm(data_loader)

            img_boxes: BBoxCollection
            # img_index: List containing one index
            for batch_n, (img_index, img_boxes) in enumerate(data_loader):
                logger.debug(f"Sending batch number: {batch_n}/{n_batches}")

                # Making a copy of the Bounding Boxes
                img_boxes = [dataclasses.replace(box) for box in img_boxes]

                for box_l, box_r in itertools.combinations(img_boxes, 2):
                    if box_l.features is not None and box_r.features is not None:
                        similarity = self.similarity(box_l.features, box_r.features)
                        # if the 2 boxes are too similar, set the features for the worst box to None
                        if similarity > max_similarity:
                            box_r.features = None

                indices.append(img_index[0])
                bbox_collections.append(img_boxes)

                logger.debug(f"Collecting results: {batch_n}/{n_batches}")

            # Returning accumulated results
        return indices, bbox_collections
