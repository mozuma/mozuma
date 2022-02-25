from collections import OrderedDict
from typing import Callable, List

import torch
import torchvision.models
from typing_extensions import Literal

from mlmodule.contrib.resnet.utils import sanitize_resnet_arch
from mlmodule.labels.base import LabelSet
from mlmodule.labels.imagenet import IMAGENET_LABELS
from mlmodule.v2.base.predictions import BatchModelPrediction
from mlmodule.v2.states import StateType
from mlmodule.v2.torch.modules import TorchMlModule
from mlmodule.v2.torch.transforms import TORCHVISION_STANDARD_IMAGE_TRANSFORMS

ResNetArchs = Literal[
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


class TorchResNetImageNetModule(TorchMlModule[torch.Tensor, torch.Tensor]):
    """PyTorch ResNet architecture for ImageNet classification.

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

        device (torch.device): Torch device to initialise the model weights
    """

    def __init__(
        self, resnet_arch: ResNetArchs, device: torch.device = torch.device("cpu")
    ):
        super().__init__(device)
        self.resnet_arch = resnet_arch

        # Getting the resnet architecture from torchvision
        base_resnet = self.get_resnet_module(resnet_arch)
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
        return sanitize_resnet_arch(self.resnet_arch)

    @property
    def state_type(self) -> StateType:
        """ResNet for ImageNet architecture

        Returns:
            StateType: The ResNet identifier for imagenet classification:
                `StateType(backend="pytorch", architecture={resnet_arch}, extra=("imagenet",))`
        """
        return StateType(
            backend="pytorch", architecture=self.resnet_arch_safe, extra=("cls1000",)
        )

    @classmethod
    def get_resnet_module(cls, resnet_arch: ResNetArchs) -> torchvision.models.ResNet:
        # Getting the ResNet architecture https://pytorch.org/vision/stable/models.html
        return getattr(torchvision.models, resnet_arch)()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.features_module(x), 1)

    def forward_classifier(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier_module(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ResNet model"""
        return self.forward_classifier(self.forward_features(x))

    def forward_predictions(
        self, batch: torch.Tensor
    ) -> BatchModelPrediction[torch.Tensor]:
        """Forward pass of the ResNet model

        Returns:
            BatchModelPrediction: Features and labels_scores (ImageNet)
        """
        features = self.forward_features(batch)
        labels_scores = self.forward_classifier(features)
        return BatchModelPrediction(features=features, label_scores=labels_scores)

    def get_dataset_transforms(self) -> List[Callable]:
        """Standard TorchVision image transforms"""
        return TORCHVISION_STANDARD_IMAGE_TRANSFORMS

    def get_labels(self) -> LabelSet:
        return IMAGENET_LABELS
