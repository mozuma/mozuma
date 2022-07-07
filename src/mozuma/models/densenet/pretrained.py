import torch

from mozuma.helpers.torchvision import DenseNetArch
from mozuma.labels.imagenet import IMAGENET_LABELS
from mozuma.labels.places import PLACES_LABELS
from mozuma.models.densenet.modules import TorchDenseNetModule
from mozuma.stores import load_pretrained_model


def torch_densenet_imagenet(
    densenet_arch: DenseNetArch,
    device: torch.device = torch.device("cpu"),
) -> TorchDenseNetModule:
    """PyTorch DenseNet model pretrained on ImageNet

    Args:
        densenet_arch (DenseNetArch): Identifier for the DenseNet architecture.
            Must be one of:

                - densenet121
                - densenet161
                - densenet169
                - densenet201
        device (torch.device): Torch device to initialise the model weights

    Returns:
        TorchDenseNetModule: PyTorch DenseNet model pretrained on ImageNet
    """
    model = TorchDenseNetModule(
        densenet_arch=densenet_arch, label_set=IMAGENET_LABELS, device=device
    )

    return load_pretrained_model(model, training_id="imagenet")


def torch_densenet_places365(
    device: torch.device = torch.device("cpu"),
) -> TorchDenseNetModule:
    """PyTorch DenseNet model pretrained on Places365.

    See [places365 documentation](https://github.com/CSAILVision/places365) for more info.

    Args:
        device (torch.device): Torch device to initialise the model weights

    Returns:
        TorchDenseNetModule: PyTorch DenseNet model pretrained on Places365
    """
    model = TorchDenseNetModule(
        densenet_arch="densenet161", label_set=PLACES_LABELS, device=device
    )

    return load_pretrained_model(model, training_id="places365")
