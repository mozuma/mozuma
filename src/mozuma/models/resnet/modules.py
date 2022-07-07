from collections import OrderedDict
from enum import Enum
from typing import Callable, List, Optional, Tuple, Union, cast

import torch

from mozuma.helpers.torchvision import (
    ResNetArch,
    get_torchvision_model,
    sanitize_torchvision_arch,
)
from mozuma.labels.base import LabelSet
from mozuma.labels.imagenet import IMAGENET_LABELS
from mozuma.predictions import BatchModelPrediction
from mozuma.states import StateType
from mozuma.torch.modules import TorchModel
from mozuma.torch.transforms import TORCHVISION_STANDARD_IMAGE_TRANSFORMS

TorchResNetForwardOutputType = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


class TorchResNetTrainingMode(Enum):
    """Enables to train a `TorchResNetModule` either on features or on labels"""

    features: str = "features"
    labels: str = "labels"


class TorchResNetModule(TorchModel[torch.Tensor, TorchResNetForwardOutputType]):
    """PyTorch ResNet architecture.

    See [PyTorch's documentation](https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html).

    Attributes:
        resnet_arch (ResNetArchs): Identifier for the ResNet architecture to load.
            Must be one of:

            - `resnet18`
            - `resnet34`
            - `resnet50`
            - `resnet101`
            - `resnet152`
            - `resnext50_32x4d`
            - `resnext101_32x8d`
            - `wide_resnet50_2`
            - `wide_resnet101_2`

        label_set (LabelSet): The output labels. Defaults to ImageNet 1000 labels.
        device (torch.device): Torch device to initialise the model weights
        training_mode (TorchResNetTrainingMode | None): Whether to return features or labels in the forward function.
            Used for training when computing the loss.
    """

    def __init__(
        self,
        resnet_arch: ResNetArch,
        label_set: LabelSet = None,
        device: torch.device = torch.device("cpu"),
        training_mode: Optional[TorchResNetTrainingMode] = None,
    ):
        super().__init__(device)
        self.resnet_arch = resnet_arch
        self.label_set = label_set or IMAGENET_LABELS
        self.training_mode = training_mode

        # Getting the resnet architecture from torchvision
        base_resnet = get_torchvision_model(
            resnet_arch, num_classes=len(self.label_set)
        )
        self.features_module = torch.nn.Sequential(
            OrderedDict(
                [
                    ("conv1", base_resnet.conv1),
                    ("bn1", base_resnet.bn1),
                    ("relu", base_resnet.relu),
                    ("maxpool", base_resnet.maxpool),
                    ("layer1", base_resnet.layer1),
                    ("layer2", base_resnet.layer2),
                    ("layer3", base_resnet.layer3),
                    ("layer4", base_resnet.layer4),
                    ("avgpool", base_resnet.avgpool),
                ]
            )
        )
        self.classifier_module = base_resnet.fc

    @property
    def resnet_arch_safe(self) -> str:
        return sanitize_torchvision_arch(self.resnet_arch)

    @property
    def state_type(self) -> StateType:
        """ResNet for ImageNet architecture

        Returns:
            StateType: The ResNet identifier for imagenet classification:
                `StateType(backend="pytorch", architecture={resnet_arch}, extra=("cls1000",))`
        """
        return StateType(
            backend="pytorch",
            architecture=self.resnet_arch_safe,
            extra=(f"cls{len(self.label_set)}",),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.features_module(x), 1)

    def forward_classifier(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier_module(x)

    def forward(self, batch: torch.Tensor) -> TorchResNetForwardOutputType:
        """Forward pass of the ResNet model"""
        features = self.forward_features(batch)

        # If the model is set on training, return either features or labels_scores
        if self.training_mode == TorchResNetTrainingMode.features:
            return features
        elif self.training_mode == TorchResNetTrainingMode.labels:
            labels_scores = self.forward_classifier(features)
            return labels_scores

        # During inference use both instead
        labels_scores = self.forward_classifier(features)
        return features, labels_scores

    def to_predictions(
        self, forward_output: TorchResNetForwardOutputType
    ) -> BatchModelPrediction[torch.Tensor]:
        """Forward pass of the ResNet model

        Returns:
            BatchModelPrediction: Features and labels_scores
        """
        features = None
        labels_scores = None

        if self.training_mode == TorchResNetTrainingMode.features:
            features = cast(torch.Tensor, forward_output)
        elif self.training_mode == TorchResNetTrainingMode.labels:
            labels_scores = cast(torch.Tensor, forward_output)
        else:
            features, labels_scores = forward_output

        return BatchModelPrediction(features=features, label_scores=labels_scores)

    def get_dataset_transforms(self) -> List[Callable]:
        """Standard TorchVision image transforms"""
        return TORCHVISION_STANDARD_IMAGE_TRANSFORMS

    def get_labels(self) -> LabelSet:
        return self.label_set
