"""
This file contains classes to help organise complex model outputs such as bounding boxes
"""
import dataclasses
from typing import Iterable, Iterator, Tuple, Optional, List, Union, cast

import numpy as np
import torch

from mlmodule.torch.utils import tensor_to_ndarray


@dataclasses.dataclass
class BBoxPoint:
    """Bounding box point coordinates"""
    x: float
    y: float

    def __getitem__(self, item: int) -> float:
        """Makes the BBoxPoint behave as a tuple with self[0] = x and self[1] = y

        Raises:
            IndexError: if `item` > 1
        """
        if item == 0:
            return self.x
        elif item == 1:
            return self.y
        else:
            raise IndexError(f'Index {item} is out of range for BBoxPoint')

    def __len__(self) -> int:
        """The length of a BBoxPoint is fixed"""
        return 2


@dataclasses.dataclass
class BBoxOutput:
    """Bounding box model output"""
    bounding_box: Tuple[BBoxPoint, BBoxPoint]   # Point pair
    probability: float  # Confidence of the bounding box
    features: Optional[np.ndarray] = None


BBoxCollection = List[BBoxOutput]


_TensorOrArray = Union[torch.Tensor, np.ndarray]


@dataclasses.dataclass
class BBoxOutputArrayFormat:
    """Bounding box container in array format

    This type should be used as the return type of the forward function
    """
    # Each row contains all the bounding boxes for one image
    bounding_boxes: Iterable[_TensorOrArray]
    # Each row contains the list of probabities for one image
    probabilities: Iterable[_TensorOrArray]
    # Each row contains the features associated with all bounding boxes for one image
    features: Optional[Iterable[_TensorOrArray]] = None

    _iter_boundin_boxes: Iterator[Optional[_TensorOrArray]] = dataclasses.field(init=False)
    _iter_probabilities: Iterator[Optional[_TensorOrArray]] = dataclasses.field(init=False)
    _iter_features: Optional[Iterator[Optional[_TensorOrArray]]] = dataclasses.field(init=False, default=None)

    def __iter__(self) -> 'BBoxOutputArrayFormat':
        self._iter_boundin_boxes = iter(self.bounding_boxes)
        self._iter_probabilities = iter(self.probabilities)
        if self.features:
            self._iter_features = iter(self.features)
        return self

    def __next__(self) -> BBoxCollection:
        """Iterates over images and returns their associated bounding box collection"""
        bounding_boxes_array = next(self._iter_boundin_boxes)
        if bounding_boxes_array is None:
            # No bounding box detected for this images
            return []

        ret_bbox_collection: BBoxCollection = []
        bounding_boxes = bounding_boxes_array.tolist()
        probabilities: List[float] = cast(_TensorOrArray, next(self._iter_probabilities)).tolist()
        features: Optional[np.ndarray] = None
        if self._iter_features:
            features = tensor_to_ndarray(cast(_TensorOrArray, next(self._iter_features)))
        for i, (bbox, prob) in enumerate(zip(bounding_boxes, probabilities)):
            # Iterating through each bounding box
            if bbox is not None:
                # We have detected a face
                ret_bbox_collection.append(BBoxOutput(
                    bounding_box=(BBoxPoint(*bbox[:2]), BBoxPoint(*bbox[2:])),  # Extracting two points
                    probability=prob,
                    features=features[i] if features is not None else None
                ))

        return ret_bbox_collection
