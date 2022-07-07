from typing import Container, Optional, Union, overload

import torch
import torchvision.models
from typing_extensions import Literal

from mozuma.states import StateType

ResNetArch = Literal[
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

DenseNetArch = Literal[
    "densenet121",
    "densenet161",
    "densenet169",
    "densenet201",
]

TorchVisionArch = Union[ResNetArch, DenseNetArch]


def sanitize_torchvision_arch(torchvision_arch: TorchVisionArch) -> str:
    """Puts to lower case and replace underscore with dash"""
    return torchvision_arch.lower().replace("_", "-")


def validate_torchvision_state_type(
    state_type: StateType,
    valid_architectures: Optional[Container[str]] = None,
) -> None:
    """Makes sure that the given state type is valid.

    Test that the backend is PyTorch and that `state.architecture`
    is in `valid_architectures`

    Arguments:
        state_type (StateType): The state type to test
        valid_architectures (Container[TorchVIsionArch] | None): Optionally restricting the architecture names.
    """
    if state_type.backend != "pytorch":
        raise ValueError("TorchVision state type should have backend='pytorch'")
    if (
        valid_architectures is not None
        and state_type.architecture not in valid_architectures
    ):
        raise ValueError(
            f"State type architecture {state_type.architecture} is not valid"
        )


@overload
def get_torchvision_model(
    model_arch: ResNetArch, **model_kwargs
) -> torchvision.models.ResNet:
    ...


@overload
def get_torchvision_model(
    model_arch: DenseNetArch, **model_kwargs
) -> torchvision.models.DenseNet:
    ...


def get_torchvision_model(
    model_arch: TorchVisionArch, **model_kwargs
) -> torch.nn.Module:
    """Get a torchvision models

    This model will be retrieved from
    [`torchvision.models`](https://github.com/pytorch/vision/tree/main/torchvision/models)

    Arguments:
        model_arch (TorchVisionArch): The name of the model architecture to build.
    """
    return getattr(torchvision.models, model_arch)(**model_kwargs)
