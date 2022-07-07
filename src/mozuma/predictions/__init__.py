import dataclasses
from logging import getLogger
from typing import Generic, Optional, Sequence, TypeVar, Union

import numpy as np
import torch

_ArrayType = TypeVar("_ArrayType", bound=Union[np.ndarray, torch.Tensor])

logger = getLogger()


@dataclasses.dataclass
class BatchVideoFramesPrediction(Generic[_ArrayType]):
    """Result of video frames extraction on a video

    Attributes:
        frame_indices (Sequence[int]): The frame index in the source video
        features (ArrayLike | None): The features of the video frames.

            Dimensions=`(n_frames, features_dims...)`
    """

    frame_indices: Sequence[int]
    features: Optional[_ArrayType] = None


@dataclasses.dataclass
class BatchBoundingBoxesPrediction(Generic[_ArrayType]):
    """Results of bounding box extraction on an images

    Attributes:
        bounding_boxes (ArrayLike): Array of bounding box coordinates.

            Dimensions=`(n_boxes, 4)`.

        scores (ArrayLike | None): Array of scores for each bounding boxes.

            Dimensions=`(n_boxes, 1)`.

        features (ArrayLike | None): Array of features for each bounding boxes.

            Dimensions=`(n_boxes, features_dims...)`.
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
    """Class defining the accepted types for model's predictions

    Attributes:
        features (ArrayLike | None): Features or embeddings.

            Dimensions=`(dataset_size, feature_dims...)`

        label_scores (ArrayLike | None): Score for each predicted label.

            Dimensions=`(dataset_size, n_labels)`

        frames (Sequence[BatchVideoFramesPrediction[ArrayLike]] | None):
            Prediction for each frame and video.
            See [BatchVideoFramesPrediction][mozuma.predictions.BatchVideoFramesPrediction].

            Sequence length=`dataset_size`

        bounding_boxes (Sequence[BatchBoundingBoxesPrediction[ArrayLike]] | None):
            Prediction for each bounding_box and image.
            See [BatchBoundingBoxesPrediction][mozuma.predictions.BatchBoundingBoxesPrediction].

            Sequence length=`dataset_size`
    """

    features: Optional[_ArrayType] = None
    label_scores: Optional[_ArrayType] = None
    frames: Optional[Sequence[BatchVideoFramesPrediction[_ArrayType]]] = None
    bounding_boxes: Optional[Sequence[BatchBoundingBoxesPrediction[_ArrayType]]] = None
