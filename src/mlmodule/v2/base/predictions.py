import dataclasses
from typing import Generic, Optional, Sequence, TypeVar

_ArrayType = TypeVar("_ArrayType")


@dataclasses.dataclass
class BatchVideoFramesPrediction(Generic[_ArrayType]):
    features: Optional[_ArrayType]
    frame_indices: Sequence[int]


@dataclasses.dataclass
class BatchModelPrediction(Generic[_ArrayType]):
    features: Optional[_ArrayType] = None
    label_scores: Optional[_ArrayType] = None
    frames: Optional[Sequence[BatchVideoFramesPrediction[_ArrayType]]] = None
