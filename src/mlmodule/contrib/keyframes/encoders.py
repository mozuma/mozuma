from typing import Callable, List, Optional, Tuple

import torch
from torchvision.transforms import Compose

from mlmodule.contrib.keyframes.datasets import FPSVideoFrameExtractorTransform
from mlmodule.contrib.keyframes.transforms import (
    ApplyImageTransformToVideoFrames,
    stack_and_squeeze_video_frames,
)
from mlmodule.v2.base.predictions import (
    BatchModelPrediction,
    BatchVideoFramesPrediction,
)
from mlmodule.v2.stores import MLModuleModelStore
from mlmodule.v2.torch.modules import TorchMlModule


class GenericVideoFramesEncoder(
    TorchMlModule[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
):
    """Video frames encoder

    This module will extract and encode frames of a video using an `image_encoder`.

    Attributes:
        image_encoder (TorchMlModule[torch.Tensor]): The PyTorch module to encode frames
        fps (int, optional): The number of frames per seconds to extract from the video.
            Defaults to 1.
        device (torch.device, optional): The PyTorch device to initialise the model weights.
            Defaults to `torch.device("cpu")`.
    """

    def __init__(
        self,
        image_encoder: TorchMlModule[torch.Tensor, torch.Tensor],
        fps: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(device=device)
        self.fps = fps
        self.image_encoder = image_encoder

    def forward_predictions(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> BatchModelPrediction[torch.Tensor]:
        """Encodes video frames

        Arguments:
            batch (Tuple[torch.Tensor, torch.Tensor]): A tuple of video (frame index, frame content)

        Returns:
            BatchModelPrediction[torch.Tensor]: Fills the `frames` attribute only
        """
        frame_indices, frame_images = batch
        if len(frame_images) > 1:
            raise ValueError(
                f"Unexpected len(frame_images)={len(frame_images)}, should be 1. "
                "Make sure that batch size is set to 1"
            )
        frames: Optional[torch.Tensor]
        if len(frame_images.shape) == 4:
            frames = self.image_encoder.forward_predictions(frame_images).features
        elif len(frame_images[0]) > 0:
            frames = self.image_encoder.forward_predictions(frame_images[0]).features
        else:
            frames = torch.Tensor(0)

        if frames is None:
            raise ValueError(
                "Cannot encode video frames, the image encoder does not return features."
            )

        return BatchModelPrediction(
            frames=[
                BatchVideoFramesPrediction(
                    features=frames, frame_indices=frame_indices[0].tolist()
                )
            ]
        )

    def get_dataset_transforms(self) -> List[Callable]:
        """Video pre-processing transforms

        Returns:
            List[Callable]: List of transforms:

            - extract frames from a video
            - Apply the `image_encoder` transforms
            - Stack frames into a `torch.Tensor`
        """
        return [
            FPSVideoFrameExtractorTransform(fps=self.fps),
            ApplyImageTransformToVideoFrames(
                image_transform_func=Compose(
                    self.image_encoder.get_dataset_transforms()
                )
            ),
            stack_and_squeeze_video_frames,
        ]

    def set_state_from_provider(self) -> None:
        MLModuleModelStore().load(self.image_encoder)
