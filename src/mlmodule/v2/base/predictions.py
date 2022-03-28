import dataclasses
from logging import getLogger
from typing import Generic, Optional, Sequence, TypeVar, Union

import numpy as np
import torch

_ArrayType = TypeVar("_ArrayType", bound=Union[np.ndarray, torch.Tensor])

logger = getLogger()


@dataclasses.dataclass
class BatchVideoFramesPrediction(Generic[_ArrayType]):
    features: Optional[_ArrayType]
    frame_indices: Sequence[int]


@dataclasses.dataclass
class BatchBoundingBoxesPrediction(Generic[_ArrayType]):
    """Holds the result of bounding box extraction on a batch of images

    Attributes:
        bounding_boxes (_ArrayType): Array of bounding box coordinates `shape=(n_boxes, 4)`
        scores (Optional[_ArrayType]): Array of scores for each bounding boxes `shape=(n_boxes, 1)`
        features (Optional[_ArrayType]): Array of features for each bounding boxes `shape=(n_boxes, features_len)`
    """

    bounding_boxes: _ArrayType
    scores: Optional[_ArrayType] = None
    features: Optional[_ArrayType] = None

    def get_by_index(self, i: int) -> "BatchBoundingBoxesPrediction":
        return BatchBoundingBoxesPrediction(
            bounding_boxes=self.bounding_boxes[i : i + 1],
            scores=self.scores[i : i + 1] if self.scores is not None else None,
            features=self.features[i : i + 1] if self.features is not None else None,
        )


@dataclasses.dataclass
class BatchModelPrediction(Generic[_ArrayType]):
    features: Optional[_ArrayType] = None
    label_scores: Optional[_ArrayType] = None
    frames: Optional[Sequence[BatchVideoFramesPrediction[_ArrayType]]] = None
    bounding_boxes: Optional[Sequence[BatchBoundingBoxesPrediction]] = None

    def __post_init__(self):
        # Workaround for PyTorch' gather (from DataParallel):
        # the gather of all model's output, as a mapping, is stored as features.
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/parallel/scatter_gather.py#L55=
        if (
            isinstance(self.features, map)
            and not self.label_scores
            and not self.frames
            and not self.bounding_boxes
        ):
            # try to unpack the mapping and handle if it doesn't work
            try:
                _features, _label_scores, _frames, _bounding_boxes = self.features

            except ValueError:
                logger.warning(
                    "It wasn't possible to unpack BatchModelPrediction "
                    "fields after a gather"
                )
                _features, _label_scores, _frames, _bounding_boxes = (
                    None,
                    None,
                    None,
                    None,
                )

            self.features = _features
            self.label_scores = _label_scores
            self.frames = _frames
            self.bounding_boxes = _bounding_boxes

    def __iter__(self):
        yield from dataclasses.astuple(self)
