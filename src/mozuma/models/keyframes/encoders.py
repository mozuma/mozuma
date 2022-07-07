from typing import Any, Callable, List, Sequence, Tuple

import torch
from torchvision.transforms import Compose

from mozuma.models.keyframes.datasets import FPSVideoFrameExtractorTransform
from mozuma.models.keyframes.transforms import (
    ApplyImageTransformToVideoFrames,
    stack_video_frames,
)
from mozuma.predictions import BatchModelPrediction, BatchVideoFramesPrediction
from mozuma.states import StateType
from mozuma.torch.modules import TorchModel


class VideoFramesEncoder(
    TorchModel[
        Tuple[Sequence[torch.LongTensor], Sequence[torch.Tensor]],
        Tuple[List[torch.LongTensor], List[torch.Tensor]],
    ]
):
    """Video frames encoder

    This module will extract and encode frames of a video using an `image_encoder`.

    Attributes:
        image_encoder (TorchModel[torch.Tensor]): The PyTorch module to encode frames
        fps (float, optional): The number of frames per seconds to extract from the video.
            Defaults to 1.
        device (torch.device, optional): The PyTorch device to initialise the model weights.
            Defaults to `torch.device("cpu")`.
    """

    def __init__(
        self,
        image_encoder: TorchModel[torch.Tensor, Any],
        fps: float = 1,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(device=device)
        self.fps = fps
        self.image_encoder = image_encoder

    @property
    def state_type(self) -> StateType:
        return self.image_encoder.state_type

    def get_state(self) -> bytes:
        return self.image_encoder.get_state()

    def set_state(self, state: bytes) -> None:
        return self.image_encoder.set_state(state)

    def forward_single(
        self, batch: Tuple[torch.LongTensor, torch.Tensor]
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        """Applies image encoder to a batch of frames

        Arguments:
            batch (Tuple[torch.LongTensor, torch.Tensor]): A tuple of

                - frame index array `torch.LongTensor, shape=(n_frames,)`
                - stacked frame images `torch.Tensor, shape=(n_frames, channel, width, height,)`

        Returns:
            Tuple[torch.LongTensor, torch.Tensor]: A tuple of

                - frame index array `torch.LongTensor, shape=(n_frames,)`
                - stacked frame images `torch.Tensor, shape=(n_frames, feature_length,)`
        """
        # Getting indices and images
        frame_indices, frame_images = batch

        # Checking input dimensions
        if len(frame_indices.shape) != 1:
            raise ValueError(
                f"Expecting 1 dimension for the frame index array, got {len(frame_indices.shape)}"
            )
        if len(frame_indices) == 0:
            # In this scenario, the list of frames is empty, so we return an empty result
            return torch.LongTensor(0), torch.Tensor(0)
        if len(frame_images.shape) != 4:
            raise ValueError(
                f"Expecting 4 dimensions for the frame images, got {len(frame_images.shape)}"
            )

        # Encoding the frames images in a batch
        frame_features = self.image_encoder.to_predictions(
            self.image_encoder(frame_images)
        ).features

        # In case the provided image encoder does not return the features attribute
        if frame_features is None:
            raise ValueError(
                "Cannot encode video frames, the image encoder does not return features."
            )

        return frame_indices, frame_features

    def forward(
        self, batch: Tuple[Sequence[torch.LongTensor], Sequence[torch.Tensor]]
    ) -> Tuple[List[torch.LongTensor], List[torch.Tensor]]:
        """Applies image encoder to a batch of frames

        Arguments:
            batch (Tuple[Sequence[torch.LongTensor], Sequence[torch.Tensor]]): A tuple of

                - Sequence of frame index array `torch.LongTensor, shape=(n_frames,)`
                - Sequence of stacked frame images `torch.Tensor, shape=(n_frames, channel, width, height,)`

                Both sequences should have the same number of elements

        Returns:
            Tuple[List[torch.LongTensor], List[torch.Tensor]]: A tuple of

                - List of frame index array `torch.LongTensor, shape=(n_frames,)`
                - List of stacked frame images `torch.Tensor, shape=(n_frames, feature_length,)`
        """
        # Produce features for all videos
        collected_features = [self.forward_single(element) for element in zip(*batch)]

        # From a list of tuple[LongTensor, Tensor] to a tuple of lists of LongTensor and Tensor
        return_value: Tuple[List[torch.LongTensor], List[torch.Tensor]] = ([], [])
        for frame_indices, frame_features in collected_features:
            return_value[0].append(frame_indices)
            return_value[1].append(frame_features)
        return return_value

    def to_predictions(
        self, forward_output: Tuple[List[torch.LongTensor], List[torch.Tensor]]
    ) -> BatchModelPrediction[torch.Tensor]:
        return BatchModelPrediction(
            frames=[
                BatchVideoFramesPrediction(
                    features=frame_features, frame_indices=frame_indices.tolist()
                )
                for frame_indices, frame_features in zip(*forward_output)
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
            stack_video_frames,
        ]
