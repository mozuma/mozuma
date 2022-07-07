import dataclasses
from typing import TYPE_CHECKING, Any, List, Sequence, Union

import numpy as np
from typing_extensions import TypeAlias

from mozuma.callbacks.base import (
    BaseSaveBoundingBoxCallback,
    BaseSaveFeaturesCallback,
    BaseSaveLabelsCallback,
    BaseSaveVideoFramesCallback,
)
from mozuma.helpers.numpy import (
    convert_batch_bounding_boxes_to_numpy,
    convert_batch_video_frames_to_numpy,
    convert_numeric_array_like_to_numpy,
)
from mozuma.models.types import ModelWithLabels
from mozuma.predictions import BatchBoundingBoxesPrediction, BatchVideoFramesPrediction

if TYPE_CHECKING:
    import torch


_NumericArrayTypes: TypeAlias = Union["torch.Tensor", np.ndarray]


@dataclasses.dataclass
class CollectFeaturesInMemory(BaseSaveFeaturesCallback[_NumericArrayTypes]):
    """Callback to collect features in memory

    Attributes:
        indices (list): List of dataset indices
        features (numpy.ndarray): Array of features.
            The first dimension correspond to `self.indices` values.

    Note:
        This callback works with any array-like features
    """

    indices: list = dataclasses.field(init=False, default_factory=list)
    features: np.ndarray = dataclasses.field(
        init=False, default_factory=lambda: np.empty(0)
    )

    def save_features(
        self, model: Any, indices: Sequence, features: _NumericArrayTypes
    ) -> None:
        # Convert features to numpy
        features_numpy = convert_numeric_array_like_to_numpy(features)

        # Adding indices
        self.indices += list(indices)

        if self.features.size > 0:
            # We stack new features
            self.features = np.vstack((self.features, features_numpy))
        else:
            # This is the first feature, we replace the default value with size=0
            self.features = features_numpy


@dataclasses.dataclass
class CollectLabelsInMemory(BaseSaveLabelsCallback[_NumericArrayTypes]):
    """Callback to collect labels in memory

    Attributes:
        indices (list): List of dataset indices
        label_scores (numpy.ndarray): Array of label scores.
            The first dimension correspond to `self.indices` values.
        labels (list[str]): List of matching labels (label with maximum score)

    Note:
        This callback works with any array-like features
    """

    indices: list = dataclasses.field(init=False, default_factory=list)
    label_scores: np.ndarray = dataclasses.field(
        init=False, default_factory=lambda: np.empty(0)
    )
    labels: List[str] = dataclasses.field(init=False, default_factory=list)

    def save_label_scores(
        self,
        model: ModelWithLabels,
        indices: Sequence,
        labels_scores: _NumericArrayTypes,
    ) -> None:
        label_set = model.get_labels()

        self.indices += list(indices)
        scores_numpy = convert_numeric_array_like_to_numpy(labels_scores)

        if self.label_scores.size > 0:
            # We stack new features
            self.label_scores = np.vstack((self.label_scores, scores_numpy))
        else:
            # This is the first feature, we replace the default value with size=0
            self.label_scores = scores_numpy

        # Getting the matching label
        label_idx = scores_numpy.argmax(axis=1)
        self.labels += [label_set[idx] for idx in label_idx]


@dataclasses.dataclass
class CollectBoundingBoxesInMemory(BaseSaveBoundingBoxCallback[_NumericArrayTypes]):
    """Callback to collect bounding boxes predictions in memory

    Attributes:
        indices (list): List of dataset indices
        frames (list[BatchVideoFramesPrediction[np.ndarray]]): Sequence of video frames.
            The first dimension correspond to `self.indices` values.

    Note:
        This callback works with any array-like features
    """

    indices: list = dataclasses.field(init=False, default_factory=list)
    bounding_boxes: List[BatchBoundingBoxesPrediction[np.ndarray]] = dataclasses.field(
        init=False, default_factory=list
    )

    def save_bounding_boxes(
        self,
        model: Any,
        indices: Sequence,
        bounding_boxes: Sequence[BatchBoundingBoxesPrediction[_NumericArrayTypes]],
    ) -> None:
        # Updating indices
        self.indices += list(indices)

        # Adding bounding boxes
        self.bounding_boxes += [
            convert_batch_bounding_boxes_to_numpy(b) for b in bounding_boxes
        ]


@dataclasses.dataclass
class CollectVideoFramesInMemory(BaseSaveVideoFramesCallback[_NumericArrayTypes]):
    """Callback to collect video frames in memory

    Attributes:
        indices (list): List of dataset indices
        frames (list[BatchVideoFramesPrediction[np.ndarray]]): Sequence of video frames.
            The first dimension correspond to `self.indices` values.

    Note:
        This callback works with any array-like features
    """

    indices: list = dataclasses.field(init=False, default_factory=list)
    frames: List[BatchVideoFramesPrediction[np.ndarray]] = dataclasses.field(
        init=False, default_factory=list
    )

    def save_frames(
        self,
        model: Any,
        indices: Sequence,
        frames: Sequence[BatchVideoFramesPrediction[_NumericArrayTypes]],
    ) -> None:
        # Updating indices
        self.indices += list(indices)

        # Adding frames
        self.frames += [convert_batch_video_frames_to_numpy(f) for f in frames]
