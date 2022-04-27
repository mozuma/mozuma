import dataclasses
from logging import getLogger
from typing import Any, List, Sequence

import ignite.distributed as idist
import numpy as np
from ignite.engine import Engine

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
from mlmodule.v2.stores.abstract import AbstractStateStore

logger = getLogger()


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

        # Adding indices
        self.indices += list(indices)

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


@dataclasses.dataclass
class SaveModelState:
    """Simple callback to save model state during training

    Attributes:
        store (AbstractStateStore): Object to handle model state saving
        training_id (str): Identifier for the training activity

    Note:
        During training, the current epoch number is appended to the
        provied `training_id`, making the filename looking like this:
        `...<training_id>-<epoch>...`.
        When the training is complete, just the `training_id` is used.

    Warning:
        This callback only saves the model state, thus does not create a whole
        training checkpoint (optimizer state, loss, etc..).
    """

    store: AbstractStateStore = dataclasses.field()
    training_id: str = dataclasses.field()

    @idist.one_rank_only()
    def save_model_state(self, engine: Engine, model: Any) -> None:
        """Save model state by calling the state store

        Arguments:
            model (Any): The MLModule model to save
        """
        epoch = engine.state.epoch
        logger.debug(f"Calling store.save for epoch {epoch}")

        # Append current epoch number to training_id
        new_training_id = "{}-{}".format(self.training_id, epoch)

        # If the training is done instead, use the pure training_id,
        # without epoch information
        is_done_epochs = (
            engine.state.max_epochs is not None
            and engine.state.epoch >= engine.state.max_epochs
        )
        if is_done_epochs:
            new_training_id = self.training_id

        # LocalStateStore raise an error when the file already exists
        # Catch it and log a warning instead, not to distrupt the runner execution
        try:
            self.store.save(model, new_training_id)
        except ValueError as err:
            logger.warning(err)
