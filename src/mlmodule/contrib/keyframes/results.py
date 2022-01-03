import dataclasses
from typing import List, Tuple

import numpy as np
import torch

from mlmodule.contrib.keyframes.keyframes import KeyFramesExtractor
from mlmodule.frames import FrameOutput, FrameOutputCollection
from mlmodule.torch.utils import tensor_to_python_list_safe
from mlmodule.v2.torch.results import AbstractResultsProcessor


@dataclasses.dataclass
class KeyFramesSelector(
    AbstractResultsProcessor[
        Tuple[torch.Tensor, torch.Tensor], List[FrameOutputCollection]
    ]
):
    """Selects the key frames from a video using encoded frames"""

    indices: list = dataclasses.field(default_factory=list, init=False)
    frames: List[FrameOutputCollection] = dataclasses.field(
        default_factory=list, init=False
    )

    @staticmethod
    def filter_keyframes(
        forward_output: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Gathering forward output into numpy arrays
        frame_indices_tensor, frame_features_tensor = forward_output
        frame_indices: np.ndarray = frame_indices_tensor.cpu().numpy()
        frame_features: np.ndarray = frame_features_tensor.cpu().numpy()
        if len(frame_features) > 0:
            extractor = KeyFramesExtractor()
            keyframe_indices = extractor.extract_keyframes(frame_features)
            return frame_indices[keyframe_indices], frame_features[keyframe_indices]
        else:
            return frame_indices, frame_features

    def process(
        self, indices: list, batch, forward_output: Tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        # Proceesing indices
        self.indices += tensor_to_python_list_safe(indices)
        # Processing keyframes
        frame_indices, frame_features = self.filter_keyframes(forward_output)
        self.frames.append(
            [
                FrameOutput(frame_pos=int(pos), probability=1, features=features)
                for pos, features in zip(frame_indices, frame_features)
            ]
        )

    def get_results(self) -> Tuple[list, List[FrameOutputCollection]]:
        return self.indices, self.frames
