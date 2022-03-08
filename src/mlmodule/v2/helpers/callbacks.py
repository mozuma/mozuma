import dataclasses
from typing import Any, List, Sequence

import numpy as np

from mlmodule.v2.base.callbacks import (
    BaseSaveBoundingBoxCallback,
    BaseSaveFeaturesCallback,
    BaseSaveLabelsCallback,
    BaseSaveVideoFramesCallback,
)
from mlmodule.v2.base.models import ModelWithLabels
from mlmodule.v2.base.predictions import (
    BatchBoundingBoxesPrediction,
    BatchVideoFramesPrediction,
)
from mlmodule.v2.helpers.types import NumericArrayTypes
from mlmodule.v2.helpers.utils import (
    convert_batch_bounding_boxes_to_numpy,
    convert_batch_video_frames_to_numpy,
    convert_numeric_array_like_to_numpy,
)


@dataclasses.dataclass
class CollectFeaturesInMemory(BaseSaveFeaturesCallback[NumericArrayTypes]):
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
        self, model: Any, indices: Sequence, features: NumericArrayTypes
    ) -> None:
        # Convert features to numpy
        features_numpy = convert_numeric_array_like_to_numpy(features)

        # Indices to skip (features with only NaNs on dimensions != 0)
        keep_indices = ~np.isnan(features_numpy).all(
            axis=tuple(range(1, len(features_numpy.shape)))
        )

        # Removing elements
        features_numpy = features_numpy[keep_indices]
        self.indices += list(i for i, k in zip(indices, keep_indices) if k)

        if self.features.size > 0:
            # We stack new features
            self.features = np.vstack((self.features, features_numpy))
        else:
            # This is the first feature, we replace the default value with size=0
            self.features = features_numpy


@dataclasses.dataclass
class CollectLabelsInMemory(BaseSaveLabelsCallback[NumericArrayTypes]):
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
        labels_scores: NumericArrayTypes,
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
class CollectBoundingBoxesInMemory(BaseSaveBoundingBoxCallback[NumericArrayTypes]):
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
        bounding_boxes: Sequence[BatchBoundingBoxesPrediction[NumericArrayTypes]],
    ) -> None:
        # Updating indices
        self.indices += list(indices)

        # Adding bounding boxes
        self.bounding_boxes += [
            convert_batch_bounding_boxes_to_numpy(b) for b in bounding_boxes
        ]


@dataclasses.dataclass
class CollectVideoFramesInMemory(BaseSaveVideoFramesCallback[NumericArrayTypes]):
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
        frames: Sequence[BatchVideoFramesPrediction[NumericArrayTypes]],
    ) -> None:
        # Updating indices
        self.indices += list(indices)

        # Adding frames
        self.frames += [convert_batch_video_frames_to_numpy(f) for f in frames]
