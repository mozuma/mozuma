import abc
import re
from collections import OrderedDict
from typing import Any, Dict, List, NoReturn, Tuple

import torch
from torch.hub import load_state_dict_from_url

from mozuma.helpers.torch import state_dict_get
from mozuma.helpers.torchvision import (
    DenseNetArch,
    sanitize_torchvision_arch,
    validate_torchvision_state_type,
)
from mozuma.models.densenet.modules import TorchDenseNetModule
from mozuma.states import StateKey, StateType
from mozuma.stores.abstract import AbstractStateStore

_DENSENET_TORCHVISION_WEIGHTS_MAP: List[Tuple[DenseNetArch, str]] = [
    ("densenet121", "https://download.pytorch.org/models/densenet121-a639ec97.pth"),
    ("densenet161", "https://download.pytorch.org/models/densenet161-8d451a50.pth"),
    ("densenet169", "https://download.pytorch.org/models/densenet169-b2777c0a.pth"),
    ("densenet201", "https://download.pytorch.org/models/densenet201-c1103571.pth"),
]

DENSENET_TORCHVISION_WEIGHTS: Dict[str, str] = {
    sanitize_torchvision_arch(k): v for k, v in _DENSENET_TORCHVISION_WEIGHTS_MAP
}


_DENSENET_PLACES365_WEIGHTS_MAP: List[Tuple[DenseNetArch, str]] = [
    (
        "densenet161",
        "http://places2.csail.mit.edu/models_places365/densenet161_places365.pth.tar",
    ),
]


DENSENET_PLACES365_WEIGHTS: Dict[str, str] = {
    sanitize_torchvision_arch(k): v for k, v in _DENSENET_PLACES365_WEIGHTS_MAP
}


def _fix_densenet_state_dict_torchvision(
    state_dict_from_url: "OrderedDict[str, torch.Tensor]",
) -> "OrderedDict[str, torch.Tensor]":
    # See https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )

    new_state_dict: List[Tuple[str, torch.Tensor]] = []
    for key, value in state_dict_from_url.items():
        # Fixing the dot problem
        res = pattern.match(key)
        if res:
            key = res.group(1) + res.group(2)
        new_state_dict.append((key, value))

    return OrderedDict(new_state_dict)


class _BaseDenseNetTorchVisionStore(AbstractStateStore[TorchDenseNetModule]):
    """Model store to load DenseNet weights pretrained on ImageNet from TorchVision"""

    _training_id: str
    """The training_id of the weights for this store"""

    _arch_to_state_url: Dict[str, str]
    """A dictionary with the URL to download weights for each architecture"""

    @abc.abstractmethod
    def _fix_densenet_state_dict(
        self, state_dict: Any
    ) -> "OrderedDict[str, torch.Tensor]":
        """Returns a cleaned version of the state dict"""

    def _validate_state_type(self, state_type: StateType) -> None:
        validate_torchvision_state_type(
            state_type, valid_architectures=self._arch_to_state_url
        )

    def save(self, model: TorchDenseNetModule, training_id: str) -> NoReturn:
        """Not implemented for this store"""
        raise NotImplementedError("Saving a model to this store is not possible")

    def load(self, model: TorchDenseNetModule, state_key: StateKey) -> None:
        """Downloads weights from torchvision's repositories

        Arguments:
            model (TorchDenseNetModule): The Torch DenseNet module to load with weights
            state_key (StateKey): Identifier of the weights to load
        """

        # Making sure the requested state has been trained on the right training_id
        if state_key.training_id != self._training_id:
            raise ValueError(
                f"TorchVision state keys should have training_id='{self._training_id}'"
            )
        self._validate_state_type(state_key.state_type)
        # Warnings if the state types are not compatible with the model
        super().load(model, state_key)

        # Loading state dict from TorchVision
        weights_url = self._arch_to_state_url[model.densenet_arch_safe]
        downloaded_state_dict: "OrderedDict[str, torch.Tensor]" = (
            load_state_dict_from_url(weights_url, map_location=model.device)
        )
        # Fixing key mismatches
        downloaded_state_dict = self._fix_densenet_state_dict(downloaded_state_dict)

        # Loading state dict
        model.load_state_dict(downloaded_state_dict)

    def get_state_keys(self, state_type: StateType) -> List[StateKey]:
        """Current store available state_keys

        Attributes:
            state_type (StateType): Filter state keys by type,
                valid state types are given by the
                [`TorchDenseNetModule.state_type`][mozuma.models.densenet.TorchDenseNetModule.state_type]
        """
        try:
            self._validate_state_type(state_type)
        except ValueError:
            return []
        else:
            return [StateKey(state_type=state_type, training_id=self._training_id)]


class DenseNetTorchVisionStore(_BaseDenseNetTorchVisionStore):
    """Model store to load DenseNet weights pretrained on ImageNet from TorchVision"""

    _training_id = "imagenet"
    _arch_to_state_url = DENSENET_TORCHVISION_WEIGHTS

    def _fix_densenet_state_dict(
        self, state_dict: "OrderedDict[str, torch.Tensor]"
    ) -> "OrderedDict[str, torch.Tensor]":
        return _fix_densenet_state_dict_torchvision(state_dict)


def _fix_densenet_state_dict_places365(
    state_dict: Dict[str, Any]
) -> "OrderedDict[str, torch.Tensor]":
    """Formats the state dict coming from places

    1. Extracts the state_dict key and applies the torchvision transforms
    2. Removes the module prefix
    """
    densenet_state_dict: "OrderedDict[str, torch.Tensor]" = state_dict["state_dict"]
    # Getting the key module
    densenet_state_dict = state_dict_get(densenet_state_dict, "module")
    return _fix_densenet_state_dict_torchvision(densenet_state_dict)


class DenseNetPlaces365Store(_BaseDenseNetTorchVisionStore):
    """Model store to load DenseNet weights pretrained on Places365

    See [places365 documentation](https://github.com/CSAILVision/places365) for more info.
    """

    _training_id = "places365"
    _arch_to_state_url = DENSENET_PLACES365_WEIGHTS

    def _fix_densenet_state_dict(
        self, state_dict: Dict[str, Any]
    ) -> "OrderedDict[str, torch.Tensor]":
        return _fix_densenet_state_dict_places365(state_dict)
