from typing import TYPE_CHECKING, Union, cast, overload

import numpy as np
from typing_extensions import TypeAlias

from mozuma.predictions import BatchBoundingBoxesPrediction, BatchVideoFramesPrediction

if TYPE_CHECKING:
    import torch


_NumericArrayTypes: TypeAlias = Union["torch.Tensor", np.ndarray]


@overload
def convert_numeric_array_like_to_numpy(num_array: None) -> None:
    ...


@overload
def convert_numeric_array_like_to_numpy(num_array: _NumericArrayTypes) -> np.ndarray:
    ...


def convert_numeric_array_like_to_numpy(num_array):
    if num_array is None:
        return None
    if hasattr(num_array, "cpu"):
        # We expect a tensor
        return cast("torch.Tensor", num_array).cpu().numpy()
    else:
        # We expect a numpy array
        return cast(np.ndarray, num_array)


def convert_batch_video_frames_to_numpy(
    video_frames: BatchVideoFramesPrediction[_NumericArrayTypes],
) -> BatchVideoFramesPrediction[np.ndarray]:
    """Convert features type of extracted frames to numpy"""
    return BatchVideoFramesPrediction(
        features=convert_numeric_array_like_to_numpy(video_frames.features),
        frame_indices=video_frames.frame_indices,
    )


def convert_batch_bounding_boxes_to_numpy(
    bounding_boxes: BatchBoundingBoxesPrediction[_NumericArrayTypes],
) -> BatchBoundingBoxesPrediction[np.ndarray]:
    """Convert array types to numpy"""
    return BatchBoundingBoxesPrediction(
        bounding_boxes=convert_numeric_array_like_to_numpy(
            bounding_boxes.bounding_boxes
        ),
        scores=convert_numeric_array_like_to_numpy(bounding_boxes.scores),
        features=convert_numeric_array_like_to_numpy(bounding_boxes.features),
    )
