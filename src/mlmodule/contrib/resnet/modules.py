from collections import OrderedDict
from typing import Callable, List, Optional, Set

import torch
import torchvision.models
from torch.hub import load_state_dict_from_url
from typing_extensions import Literal

from mlmodule.labels.base import LabelSet
from mlmodule.labels.imagenet import IMAGENET_LABELS
from mlmodule.v2.base.predictions import BatchModelPrediction
from mlmodule.v2.helpers.utils import take_unique_element
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

    def state_architecture(self) -> str:
        """ResNet for ImageNet architecture identifier

        Returns:
            str: The ResNet identifier for imagenet classification: `pytorch-{resnet_arch}-imagenet`
        """
        return f"pytorch-{self.resnet_arch}-imagenet"

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

    def provider_state_architectures(self) -> Set[str]:
        """Identifiers for torchvision supported pretrained states

        Returns:
            Set[str]: A single supported architecture for ImageNet: `pytorch-{resnet_arch}-imagenet`
        """
        return {self.state_architecture()}

    def set_state_from_provider(self, state_arch: Optional[str] = None) -> None:
        """Downloads weights from torchvision's repositories

        Arguments:
            state_arch (Optional[str]): One of `provider_state_architectures`
        """
        # Validate provided state_arch
        provider_architectures = self.provider_state_architectures()
        state_arch = state_arch or take_unique_element(provider_architectures)
        if state_arch is None or state_arch not in provider_architectures:
            raise ValueError(f"Invalid state_arch={state_arch}")

        # Getting URL to download model
        url = torchvision.models.resnet.model_urls[self.resnet_arch]
        # Downloading state dictionary
        pretrained_state_dict: OrderedDict[
            str, torch.Tensor
        ] = load_state_dict_from_url(url, map_location=self.device)
        # Splitting features and classifier layers
        features_state_dict = OrderedDict(
            [
                (key, value)
                for key, value in pretrained_state_dict.items()
                if not key.startswith("fc")
            ]
        )
        classifier_state_dict = OrderedDict(
            [
                (key[3:], value)
                for key, value in pretrained_state_dict.items()
                if key.startswith("fc")
            ]
        )
        self.features_module.load_state_dict(features_state_dict)
        self.classifier_module.load_state_dict(classifier_state_dict)

    def get_dataset_transforms(self) -> List[Callable]:
        """Standard TorchVision image transforms"""
        return TORCHVISION_STANDARD_IMAGE_TRANSFORMS

    def get_labels(self) -> LabelSet:
        return IMAGENET_LABELS
