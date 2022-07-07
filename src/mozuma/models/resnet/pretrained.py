from typing import Optional

import torch

from mozuma.helpers.torchvision import ResNetArch
from mozuma.labels.imagenet import IMAGENET_LABELS
from mozuma.models.resnet.modules import TorchResNetModule, TorchResNetTrainingMode
from mozuma.stores import load_pretrained_model


def torch_resnet_imagenet(
    resnet_arch: ResNetArch,
    device: torch.device = torch.device("cpu"),
    training_mode: Optional[TorchResNetTrainingMode] = None,
) -> TorchResNetModule:
    """[TorchResNetModule][mozuma.models.resnet.TorchResNetModule] model pre-trained on ImageNet

    Args:
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
        training_mode (TorchResNetTrainingMode | None): Whether to return features or labels in the forward function.
            Used for training when computing the loss.

    Returns:
        TorchResNetModule: A PyTorch ResNet module pre-trained on ImageNet
    """
    resnet = TorchResNetModule(
        resnet_arch,
        label_set=IMAGENET_LABELS,
        device=device,
        training_mode=training_mode,
    )

    return load_pretrained_model(resnet, training_id="imagenet")
