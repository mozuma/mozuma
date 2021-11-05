import dataclasses
from typing import List, Tuple

import torch
from mlmodule.frames import FrameOutput, FrameOutputCollection
from mlmodule.torch.utils import tensor_to_python_list_safe
from mlmodule.v2.torch.results import AbstractResultsProcessor


@dataclasses.dataclass
class KeyFramesSelector(AbstractResultsProcessor[
    Tuple[torch.Tensor, torch.Tensor], List[FrameOutputCollection]
]):
    """Selects the key frames from a video using encoded frames"""
    indices: list = dataclasses.field(default_factory=list, init=False)
    frames: List[FrameOutputCollection] = dataclasses.field(default_factory=list, init=False)

    def process(
        self, indices: list, batch, forward_output: Tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        self.indices += tensor_to_python_list_safe(indices)
        # TODO: Extract key frames
        frame_indices, frame_features = forward_output
        self.frames.append([
            FrameOutput(frame_pos=pos, probability=1, features=features)
            for pos, features in zip(frame_indices, frame_features)
        ])

    def get_results(self) -> Tuple[list, List[FrameOutputCollection]]:
        return self.indices, self.frames
