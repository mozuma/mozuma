import functools
from typing import Callable, List, Tuple

import torch
from torch.functional import F

from mozuma.helpers.torchvision import (
    DenseNetArch,
    get_torchvision_model,
    sanitize_torchvision_arch,
)
from mozuma.labels.base import LabelSet
from mozuma.labels.imagenet import IMAGENET_LABELS
from mozuma.labels.places import PLACES_LABELS
from mozuma.predictions import BatchModelPrediction
from mozuma.states import StateType
from mozuma.torch.modules import TorchModel
from mozuma.torch.transforms import TORCHVISION_STANDARD_IMAGE_TRANSFORMS


class TorchDenseNetModule(TorchModel[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch implementation of DenseNet

    See [TorchVision source code](https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py).

    Attributes:
        densenet_arch (DenseNetArch): Identifier for the DenseNet architecture.
            Must be one of:

                - densenet121
                - densenet161
                - densenet169
                - densenet201
        label_set (LabelSet): The output labels. Defaults to ImageNet 1000 labels.
        device (torch.device): Torch device to initialise the model weights
    """

    def __init__(
        self,
        densenet_arch: DenseNetArch,
        label_set: LabelSet = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(device=device)
        self.densenet_arch = densenet_arch
        self.label_set = label_set or IMAGENET_LABELS

        # Getting the DenseNet architecture from torchvision
        densenet_model = get_torchvision_model(
            self.densenet_arch, num_classes=len(self.label_set)
        )
        self.features = densenet_model.features
        self.classifier = densenet_model.classifier

    @property
    def densenet_arch_safe(self):
        return sanitize_torchvision_arch(self.densenet_arch)

    @property
    def state_type(self) -> StateType:
        return StateType(
            backend="pytorch",
            architecture=self.densenet_arch_safe,
            extra=(f"cls{len(self.label_set)}",),
        )

    def forward_features(self, batch: torch.Tensor) -> torch.Tensor:
        features = self.features(batch)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        return torch.flatten(out, 1)

    def forward_classifier(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the DenseNet module"""
        features = self.forward_features(batch)
        classes = self.forward_classifier(features)
        return features, classes

    def to_predictions(
        self, forward_output: Tuple[torch.Tensor, torch.Tensor]
    ) -> BatchModelPrediction[torch.Tensor]:
        """Forward pass of the DenseNet model

        Returns:
            BatchModelPrediction: Features and labels_scores
        """
        features, labels_scores = forward_output
        return BatchModelPrediction(features=features, label_scores=labels_scores)

    def get_dataset_transforms(self) -> List[Callable]:
        """Standard TorchVision image transforms"""
        return TORCHVISION_STANDARD_IMAGE_TRANSFORMS

    def get_labels(self) -> LabelSet:
        return self.label_set


torch_densenet_places365 = functools.partial(
    TorchDenseNetModule, label_set=PLACES_LABELS
)
