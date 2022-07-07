import torch

from mozuma.helpers.torchvision import DenseNetArch, ResNetArch
from mozuma.models.densenet.pretrained import (
    torch_densenet_imagenet,
    torch_densenet_places365,
)
from mozuma.models.keyframes.selectors import KeyFrameSelector
from mozuma.models.resnet.pretrained import torch_resnet_imagenet


def torch_keyframes_resnet_imagenet(
    resnet_arch: ResNetArch,
    fps: float = 1,
    device: torch.device = torch.device("cpu"),
    **kwargs
) -> KeyFrameSelector:
    """KeyFrames selector with PyTorch's ResNet pre-trained on ImageNet


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
        fps (float, optional): The number of frames per seconds to extract from the video.
            Defaults to 1.
        device (torch.device): Torch device to initialise the model weights

    Returns:
        KeyFrameSelector: Keyframes model with ResNet pre-trained on ImageNet encoder
    """
    return KeyFrameSelector(
        torch_resnet_imagenet(resnet_arch, device=device, **kwargs),
        fps=fps,
        device=device,
    )


def torch_keyframes_densenet_imagenet(
    densenet_arch: DenseNetArch,
    fps: float = 1,
    device: torch.device = torch.device("cpu"),
    **kwargs
) -> KeyFrameSelector:
    """KeyFrames selector with PyTorch DenseNet model pretrained on ImageNet

    Args:
        densenet_arch (DenseNetArch): Identifier for the DenseNet architecture.
            Must be one of:

                - densenet121
                - densenet161
                - densenet169
                - densenet201
        fps (float, optional): The number of frames per seconds to extract from the video.
            Defaults to 1.
        device (torch.device): Torch device to initialise the model weights

    Returns:
        KeyFrameSelector: Keyframes model with DenseNet pre-trained on ImageNet encoder
    """
    return KeyFrameSelector(
        torch_densenet_imagenet(densenet_arch, device=device, **kwargs),
        fps=fps,
        device=device,
    )


def torch_keyframes_densenet_places365(
    fps: float = 1, device: torch.device = torch.device("cpu"), **kwargs
) -> KeyFrameSelector:
    """KeyFrames selector with PyTorch DenseNet model pretrained on Places365.

    See [places365 documentation](https://github.com/CSAILVision/places365) for more info.

    Args:
        device (torch.device): Torch device to initialise the model weights

    Returns:
        KeyFrameSelector: Keyframes model with DenseNet model pretrained on Places365
    """
    return KeyFrameSelector(
        torch_densenet_places365(device=device, **kwargs), fps=fps, device=device
    )
