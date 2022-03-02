from typing import Callable, List, Sequence, Tuple

import numpy as np
import torch

from mlmodule.contrib.keyframes.encoders import VideoFramesEncoder
from mlmodule.contrib.keyframes.keyframes import KeyFramesExtractor
from mlmodule.v2.base.predictions import (
    BatchModelPrediction,
    BatchVideoFramesPrediction,
)
from mlmodule.v2.states import StateType
from mlmodule.v2.torch.modules import TorchMlModule


class KeyFrameSelector(
    TorchMlModule[Tuple[Sequence[torch.LongTensor], Sequence[torch.Tensor]], np.ndarray]
):
    """Video key-frames selector

    Attributes:
        image_encoder (TorchMlModule[torch.Tensor, torch.Tensor]): The PyTorch module to encode frames.
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
        self.frames_encoder = VideoFramesEncoder(
            image_encoder=image_encoder, fps=fps, device=device
        )

    @property
    def state_type(self) -> StateType:
        return self.frames_encoder.state_type

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
    ) -> Tuple[List[List[int]], List[np.ndarray]]:
        """Selects the key-frames from a sequence of frames

        Arguments:
            batch (Tuple[Sequence[torch.LongTensor], Sequence[torch.Tensor]]): A tuple of

                - Sequence of frame index array `torch.LongTensor, shape=(n_frames,)`
                - Sequence of stacked frame images `torch.Tensor, shape=(n_frames, channel, width, height,)`

                Both sequences should have the same number of elements

        Returns:
            Tuple[List[List[int]], List[np.ndarray]]: A tuple for the selected frames with:

                - List of frame index array `List[int], length=n_frames`
                - List of stacked frame images `np.ndarray, shape=(n_frames, features_length,)`
        """
        # Encoding frames
        ret_encoder = self.frames_encoder.forward(batch)

        return_value: Tuple[List[List[int]], List[np.ndarray]] = ([], [])
        for frame_indices, frame_features in zip(*ret_encoder):
            # Selecting the key frames
            selected_indices, selected_features = self._filter_keyframes(
                frame_indices.tolist(), frame_features.cpu().numpy()
            )
            return_value[0].append(selected_indices)
            return_value[1].append(selected_features)

        return return_value

    def forward_predictions(
        self, batch: Tuple[Sequence[torch.LongTensor], Sequence[torch.Tensor]]
    ) -> BatchModelPrediction[np.ndarray]:
        """Selects the key-frames from a sequence of frames

        Arguments:
            batch (Tuple[Sequence[torch.LongTensor], Sequence[torch.Tensor]]): A tuple of

                - Sequence of frame index array `torch.LongTensor, shape=(n_frames,)`
                - Sequence of stacked frame images `torch.Tensor, shape=(n_frames, channel, width, height,)`

                Both sequences should have the same number of elements

        Returns:
            BatchModelPrediction[np.ndarray]: Fills the `frames` attribute with features of ndarray type
        """
        selected_frames = self.forward(batch)

        return BatchModelPrediction(
            frames=[
                BatchVideoFramesPrediction(
                    features=selected_features, frame_indices=selected_indices
                )
                for selected_indices, selected_features in zip(*selected_frames)
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
