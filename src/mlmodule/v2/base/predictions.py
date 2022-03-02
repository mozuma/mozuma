import dataclasses
from typing import Generic, Optional, Sequence, TypeVar

_ArrayType = TypeVar("_ArrayType")


@dataclasses.dataclass
class BatchVideoFramesPrediction(Generic[_ArrayType]):
    features: Optional[_ArrayType]
    frame_indices: Sequence[int]


@dataclasses.dataclass
class BatchBoundingBoxesPrediction(Generic[_ArrayType]):
    """Holds the result of bounding box extraction on a batch of images

    Attributes:
        bounding_boxes (_ArrayType): Array of bounding box coordinates `shape=(n_boxes, 4)`
        scores (Optional[_ArrayType]): Array of scores for each bounding boxes
        features (Optional[_ArrayType]): Array of features for each bounding boxes
    """

    bounding_boxes: _ArrayType
    scores: Optional[_ArrayType] = None
    features: Optional[_ArrayType] = None


@dataclasses.dataclass
class BatchModelPrediction(Generic[_ArrayType]):
    features: Optional[_ArrayType] = None
    label_scores: Optional[_ArrayType] = None
    frames: Optional[Sequence[BatchVideoFramesPrediction[_ArrayType]]] = None
    bounding_boxes: Optional[Sequence[BatchBoundingBoxesPrediction]] = None
