import abc
from typing import Any, Callable, Generic, Optional, Sequence, TypeVar, Union

import numpy as np
import torch

from mozuma.models.types import ModelWithLabels
from mozuma.predictions import BatchBoundingBoxesPrediction, BatchVideoFramesPrediction

_ContraArrayType = TypeVar(
    "_ContraArrayType", bound=Union[torch.Tensor, np.ndarray], contravariant=True
)


class BaseSaveFeaturesCallback(abc.ABC, Generic[_ContraArrayType]):
    @abc.abstractmethod
    def save_features(
        self, model: Any, indices: Sequence, features: _ContraArrayType
    ) -> None:
        """Save features output returned by a module

        Arguments:
            model (Any): The MoZuMa model that produced the features
            indices (Sequence): The list of indices as defined by the dataset
            features (ArrayLike): The feature object as returned by the model
        """
        pass


class BaseSaveLabelsCallback(abc.ABC, Generic[_ContraArrayType]):
    @abc.abstractmethod
    def save_label_scores(
        self, model: ModelWithLabels, indices: Sequence, labels_scores: _ContraArrayType
    ) -> None:
        """Save labels scores returned by a module

        Arguments:
            model (ModelWithLabels): The MoZuMa model that produced the label scores
            indices (Sequence): The list of indices as defined by the dataset
            labels_scores (ArrayLike): Contains the output score/probability for each label
        """
        pass


class BaseSaveVideoFramesCallback(abc.ABC, Generic[_ContraArrayType]):
    @abc.abstractmethod
    def save_frames(
        self,
        model: Any,
        indices: Sequence,
        frames: Sequence[BatchVideoFramesPrediction[_ContraArrayType]],
    ) -> None:
        """Save frames extracted from a video

        Arguments:
            model (Any): The model that produces the video frames encoding
            indices (Sequence): The list of indices as defined by the dataset
            frames (Sequence[BatchVideoFramesPrediction[_ArrayType]]):
                The sequence of frame features and indices
        """


class BaseSaveBoundingBoxCallback(abc.ABC, Generic[_ContraArrayType]):
    @abc.abstractmethod
    def save_bounding_boxes(
        self,
        model: Any,
        indices: Sequence,
        bounding_boxes: Sequence[BatchBoundingBoxesPrediction[_ContraArrayType]],
    ) -> None:
        """Save bounding boxes output of a module

        Arguments:
            model (Any): The model that produces the bounding boxes
            indices (Sequence): The list of indices as defined by the dataset
            bounding_boxes (Sequence[BatchBoundingBoxesPrediction[_ContraArrayType]]):
                The sequence bounding predictions
        """


class BaseRunnerEndCallback(abc.ABC):
    @abc.abstractmethod
    def on_runner_end(self, model: Any) -> None:
        """Called when the runner finishes

        This can be used to do clean up.
        For instance if the data is being processed by a thread,
        this function can wait for the thread to finish.$

        Arguments:
            model (Any): The currently run model
        """


def callbacks_caller(
    callbacks: list, callback_name: str, *args: Any, **kwds: Any
) -> None:
    """Calls all function named `callback_name` of classes in `callbacks` list with the provided arguments"""
    for c in callbacks:
        # Getting the callback function if exists
        fun: Optional[Callable] = getattr(c, callback_name, None)
        if fun:
            # Call the function
            fun(*args, **kwds)
