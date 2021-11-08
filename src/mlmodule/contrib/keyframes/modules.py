from typing import Callable, List, Tuple

import torch
from torchvision.transforms import Compose
from mlmodule.contrib.keyframes.transforms import ApplyImageTransformToVideoFrames, stack_and_squeeze_video_frames
from mlmodule.v2.torch.models import TorchModel


class VideoFramesEncoder(TorchModel):

    def __init__(self, image_encoder: TorchModel):
        super().__init__()
        self.image_encoder = image_encoder

    def forward(self, video_frames: Tuple[torch.Tensor, torch.Tensor]):
        """Encodes the video frames passed as input

        Arguments:
            video_frames: tuple
        """
        frame_indices, frame_images = video_frames
        if len(frame_images) > 1:
            raise ValueError(
                f"Unexpected len(frame_images)={len(frame_images)}, should be 1."
            )
        frames: torch.Tensor
        if len(frame_images[0]) > 0:
            frames = self.image_encoder(frame_images[0])
        else:
            frames = torch.Tensor(0)
        return frame_indices[0], frames

    def get_dataset_transforms(self) -> List[Callable]:
        return [
            ApplyImageTransformToVideoFrames(
                image_transform_func=Compose(self.image_encoder.get_dataset_transforms())
            ),
            stack_and_squeeze_video_frames
        ]
