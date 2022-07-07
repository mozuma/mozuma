from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from mozuma.helpers.torchvision import ResNetArch
from mozuma.labels.base import LabelSet
from mozuma.models.keyframes.encoders import VideoFramesEncoder
from mozuma.models.keyframes.keyframes import KeyFramesExtractor
from mozuma.models.resnet.modules import TorchResNetModule
from mozuma.predictions import BatchModelPrediction, BatchVideoFramesPrediction
from mozuma.states import StateType
from mozuma.torch.modules import TorchModel


class KeyFrameSelector(
    TorchModel[
        Tuple[Sequence[torch.LongTensor], Sequence[torch.Tensor]],
        Tuple[List[torch.LongTensor], List[torch.Tensor]],
    ]
):
    """Video key-frames selector

    Attributes:
        image_encoder (TorchModel[torch.Tensor, torch.Tensor]): The PyTorch module to encode frames.
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
        self.frames_encoder = VideoFramesEncoder(
            image_encoder=image_encoder, fps=fps, device=device
        )

    @property
    def state_type(self) -> StateType:
        return self.frames_encoder.state_type

    def get_state(self) -> bytes:
        return self.frames_encoder.get_state()

    def set_state(self, state: bytes) -> None:
        return self.frames_encoder.set_state(state)

    def _filter_keyframes(
        self,
        frame_indices: Sequence[int],
        frame_features: np.ndarray,
    ) -> Tuple[List[int], np.ndarray]:
        if len(frame_features) > 0:
            extractor = KeyFramesExtractor()
            keyframe_indices = extractor.extract_keyframes(frame_features)
            return [frame_indices[i] for i in keyframe_indices], frame_features[
                keyframe_indices
            ]
        else:
            return list(frame_indices), frame_features

    def forward(
        self, batch: Tuple[Sequence[torch.LongTensor], Sequence[torch.Tensor]]
    ) -> Tuple[List[torch.LongTensor], List[torch.Tensor]]:
        """Selects the key-frames from a sequence of frames

        Arguments:
            batch (Tuple[Sequence[torch.LongTensor], Sequence[torch.Tensor]]): A tuple of

                - Sequence of frame index array `torch.LongTensor, shape=(n_frames,)`
                - Sequence of stacked frame images `torch.Tensor, shape=(n_frames, channel, width, height,)`

                Both sequences should have the same number of elements

        Returns:
            Tuple[List[torch.LongTensor], List[torch.Tensor]]: A tuple for the selected frames with:

                - List of frame index array `torch.LongTensor, shape=(n_frames,)`
                - List of stacked frame images `torch.Tensor, shape=(n_frames, features_length,)`
        """
        # Encoding frames
        ret_encoder = self.frames_encoder.forward(batch)

        ret_indices: List[torch.LongTensor] = []
        ret_frames: List[torch.Tensor] = []
        for frame_indices, frame_features in zip(*ret_encoder):
            # Selecting the key frames
            selected_indices, selected_features = self._filter_keyframes(
                frame_indices.tolist(), frame_features.cpu().numpy()
            )
            ret_indices.append(torch.LongTensor(selected_indices))
            ret_frames.append(torch.tensor(selected_features))

        return ret_indices, ret_frames

    def to_predictions(
        self, forward_output: Tuple[List[torch.LongTensor], List[torch.Tensor]]
    ) -> BatchModelPrediction[torch.Tensor]:
        """Selects the key-frames from a sequence of frames

        Arguments:
            batch (Tuple[Sequence[torch.LongTensor], Sequence[torch.Tensor]]): A tuple of

                - Sequence of frame index array `torch.LongTensor, shape=(n_frames,)`
                - Sequence of stacked frame images `torch.Tensor, shape=(n_frames, channel, width, height,)`

                Both sequences should have the same number of elements

        Returns:
            BatchModelPrediction[np.ndarray]: Fills the `frames` attribute with features of ndarray type
        """
        return BatchModelPrediction(
            frames=[
                BatchVideoFramesPrediction(
                    features=selected_features, frame_indices=selected_indices.tolist()
                )
                for selected_indices, selected_features in zip(*forward_output)
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
        return self.frames_encoder.get_dataset_transforms()
