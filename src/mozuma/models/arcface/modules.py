import functools
import os
from typing import Callable, List

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Normalize, ToTensor

from mozuma.models.arcface.transforms import ArcFaceAlignment
from mozuma.predictions import BatchModelPrediction
from mozuma.states import StateType
from mozuma.torch.layers import Bottleneck_IR_SE, get_block
from mozuma.torch.modules import TorchModel
from mozuma.torch.utils import l2_norm

# See https://quip.com/blC4A0YmfhbQ/Approach-to-remove-face-embeddings-leading-to-false-positive
ARCFACE_DEFAULT_BAD_FACE_DISTANCE = 0.87


# We do not expect to have to load it for more than 10 different GPUs
@functools.lru_cache(10)
def load_normalized_arcface_faces(device: torch.device) -> torch.Tensor:
    """Loads the normalized embeddings of bad quality faces

    Returns:
        torch.Tensor
    """
    embeddings: np.ndarray = np.load(
        os.path.join(os.path.dirname(__file__), "normalized_faces.npy")
    )

    # Multiplying with a matrix that takes to mean value for each feature
    # This greatly reduces the size of the dot product with features
    norm_faces = embeddings @ np.ones((embeddings.shape[1], 1)) / embeddings.shape[1]
    return torch.Tensor(norm_faces)


def bad_quality_faces_index(
    arcface_embeddings: torch.Tensor,
    bad_quality_threshold: float = ARCFACE_DEFAULT_BAD_FACE_DISTANCE,
) -> torch.Tensor:
    """Indices of bad quality faces from embeddings

    Args:
        arcface_embeddings (torch.Tensor): Shape=(batch_size, embeddings_size).

    Returns:
        torch.Tensor: A boolean tensor to identify which position in the given
            `arcface_embeddings` are bad quality faces. Shape=(batch_size,).
    """
    # Getting the reference faces
    norm_faces = load_normalized_arcface_faces(arcface_embeddings.device)

    # Filter for faces with good quality
    return (
        1 - torch.matmul(arcface_embeddings, norm_faces)[:, 0]
    ) > bad_quality_threshold


class TorchArcFaceModule(
    TorchModel[torch.Tensor, torch.Tensor],
):
    """Creates face embeddings from MTCNN output

    Attributes:
        device (torch.device): Torch device to initialise the model weights
        remove_bad_faces (bool): Whether to remove the faces with bad quality from the output.
            This will replace features of bad faces with `float("nan")`. Defaults to `False`.
        bad_faces_threshold (float): The cosine similarity distance to reference faces for which we
            consider the face is of bad quality.
    """

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        remove_bad_faces: bool = False,
        drop_ratio: float = 0.6,
        bad_quality_threshold: float = ARCFACE_DEFAULT_BAD_FACE_DISTANCE,
    ):
        super().__init__(device=device)
        self.remove_bad_faces = remove_bad_faces
        self.bad_quality_threshold = bad_quality_threshold
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False), nn.BatchNorm2d(64), nn.PReLU(64)
        )
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(drop_ratio),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512),
        )
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    Bottleneck_IR_SE(
                        bottleneck.in_channel, bottleneck.depth, bottleneck.stride
                    )
                )
        self.body = nn.Sequential(*modules)

    @property
    def state_type(self) -> StateType:
        return StateType(
            backend="pytorch",
            architecture="arcface",
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Creates the faces features"""
        x = self.input_layer(batch)
        x = self.body(x)
        x = self.output_layer(x)

        if self.remove_bad_faces:
            # Marking faces of bad quality to NaN
            bad_faces = bad_quality_faces_index(
                x, bad_quality_threshold=self.bad_quality_threshold
            )
            x[bad_faces] = float("nan")
        return l2_norm(x)

    def to_predictions(
        self, forward_output: torch.Tensor
    ) -> BatchModelPrediction[torch.Tensor]:
        return BatchModelPrediction(features=forward_output)

    def get_dataset_transforms(self) -> List[Callable]:
        """Returns transforms to be applied on bulk_inference input data"""
        return [
            ArcFaceAlignment(),
            ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
