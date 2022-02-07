import dataclasses
from typing import Generic, Optional, TypeVar

_ArrayType = TypeVar("_ArrayType")


@dataclasses.dataclass
class BatchModelPrediction(Generic[_ArrayType]):
    features: Optional[_ArrayType]
    label_scores: Optional[_ArrayType]
