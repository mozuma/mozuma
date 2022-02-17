from typing import Callable, List, Sequence, Tuple

import numpy as np
import torch

from mlmodule.contrib.keyframes.encoders import GenericVideoFramesEncoder
from mlmodule.contrib.keyframes.keyframes import KeyFramesExtractor
from mlmodule.contrib.resnet.modules import TorchResNetModule
from mlmodule.v2.base.predictions import (
    BatchModelPrediction,
    BatchVideoFramesPrediction,
)
from mlmodule.v2.stores import MLModuleModelStore
from mlmodule.v2.torch.modules import TorchMlModule


class GenericKeyFrameSelector(
    TorchMlModule[Tuple[torch.Tensor, torch.Tensor], np.ndarray]
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
        self.frames_encoder = GenericVideoFramesEncoder(
            image_encoder=image_encoder, fps=fps, device=device
        )

    def _filter_keyframes(
        self,
        frame_indices: Sequence[int],
        frame_features: np.ndarray,
    ) -> Tuple[Sequence[int], np.ndarray]:
        if len(frame_features) > 0:
            extractor = KeyFramesExtractor()
            keyframe_indices = extractor.extract_keyframes(frame_features)
            return [frame_indices[i] for i in keyframe_indices], frame_features[
                keyframe_indices
            ]
        else:
            return frame_indices, frame_features

    def forward_predictions(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> BatchModelPrediction[np.ndarray]:
        """Selects the key-frames from a sequence of frames

        Arguments:
            batch (Tuple[torch.Tensor, torch.Tensor]): A tuple of video (frame index, frame content)

        Returns:
            BatchModelPrediction[np.ndarray]: Fills the `frames` attribute with features of ndarray type
        """
        encoded_frames = self.frames_encoder.forward_predictions(batch)

        if encoded_frames.frames is None:
            raise ValueError(
                "The frame_encoder should return frames predictions with frames != None"
            )

        ret_predictions: List[BatchVideoFramesPrediction[np.ndarray]] = []
        for video_frames in encoded_frames.frames:
            if video_frames.features is None:
                raise ValueError(
                    "The frame_encoder should return frames predictions with frames[].features != None."
                )
            frame_indices, frame_features = self._filter_keyframes(
                video_frames.frame_indices,
                video_frames.features.cpu().numpy(),
            )
            ret_predictions.append(
                BatchVideoFramesPrediction(
                    features=frame_features, frame_indices=frame_indices
                )
            )

        return BatchModelPrediction(frames=ret_predictions)

    def get_dataset_transforms(self) -> List[Callable]:
        """Video pre-processing transforms

        Returns:
            List[Callable]: List of transforms:

            - extract frames from a video
            - Apply the `image_encoder` transforms
            - Stack frames into a `torch.Tensor`
        """
        return self.frames_encoder.get_dataset_transforms()

    def set_state_from_provider(self) -> None:
        MLModuleModelStore().load(self.frames_encoder)


class ResNet18KeyFramesSelector(GenericKeyFrameSelector):
    """Video key-frames selector with ResNet-18 image encoder

    See [GenericKeyFrameSelector][mlmodule.contrib.keyframes.selectors.GenericKeyFrameSelector]
    for more details.

    Attributes:
        fps (int, optional): The number of frames per seconds to extract from the video.
            Defaults to 1.
        device (torch.device, optional): The PyTorch device to initialise the model weights.
            Defaults to `torch.device("cpu")`."""

    mlmodule_model_uri = "keyframes/rn18-imagenet.pth"

    def __init__(self, fps: int = 1, device: torch.device = torch.device("cpu")):
        super().__init__(
            image_encoder=TorchResNetModule("resnet18", device=device),
            fps=fps,
            device=device,
        )
