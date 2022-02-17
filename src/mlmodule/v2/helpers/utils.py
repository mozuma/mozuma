from typing import cast

import numpy as np
import torch

from mlmodule.v2.base.predictions import BatchVideoFramesPrediction
from mlmodule.v2.helpers.types import NumericArrayTypes


def convert_numeric_array_like_to_numpy(num_array: NumericArrayTypes) -> np.ndarray:
    if hasattr(num_array, "cpu"):
        # We expect a tensor
        return cast(torch.Tensor, num_array).cpu().numpy()
    else:
        # We expect a numpy array
        return cast(np.ndarray, num_array)


def convert_batch_video_frames_to_numpy(
    video_frames: BatchVideoFramesPrediction[NumericArrayTypes],
) -> BatchVideoFramesPrediction[np.ndarray]:
    """Convert features type of extracted frames to numpy"""
    return BatchVideoFramesPrediction(
        features=convert_numeric_array_like_to_numpy(video_frames.features)
        if video_frames.features is not None
        else None,
        frame_indices=video_frames.frame_indices,
    )
