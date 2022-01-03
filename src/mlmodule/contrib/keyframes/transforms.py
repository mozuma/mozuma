import dataclasses
from typing import Callable, Generic, List, Tuple, TypeVar

import torch

from mlmodule.types import FrameIdxType, FrameSequenceType, ImageDatasetType

_OutputDatasetType = TypeVar("_OutputDatasetType")


@dataclasses.dataclass
class ApplyImageTransformToVideoFrames(Generic[_OutputDatasetType]):
    image_transform_func: Callable[[ImageDatasetType], _OutputDatasetType]

    def __call__(
        self, frame_seq: FrameSequenceType
    ) -> Tuple[List[FrameIdxType], List[_OutputDatasetType]]:
        ret: List[_OutputDatasetType] = []
        frame_indices, frame_images = frame_seq
        for frame_image in frame_images:
            ret.append(self.image_transform_func(frame_image))
        return frame_indices, ret


def stack_and_squeeze_video_frames(
    video_frames: Tuple[List[int], List[torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Take video frames and stack them into one Tensor"""
    frame_indices, frame_tensors = video_frames
    return torch.tensor(frame_indices), torch.stack(
        frame_tensors
    ).squeeze() if frame_tensors else torch.Tensor(0)
