import functools
from collections import OrderedDict
from typing import List, NoReturn

import torch
from torch.hub import load_state_dict_from_url
from torchvision.models import resnet

from mozuma.helpers.torch import state_dict_combine, state_dict_get
from mozuma.helpers.torchvision import (
    sanitize_torchvision_arch,
    validate_torchvision_state_type,
)
from mozuma.models.resnet.modules import TorchResNetModule
from mozuma.states import StateKey, StateType
from mozuma.stores.abstract import AbstractStateStore

RESNET_ARCHITECTURES_MAP = {sanitize_torchvision_arch(a): a for a in resnet.model_urls}


_validate_resnet_state_type = functools.partial(
    validate_torchvision_state_type, valid_architectures=RESNET_ARCHITECTURES_MAP
)


class ResNetTorchVisionStore(AbstractStateStore[TorchResNetModule]):
    """Model store to load ResNet weights pretrained on ImageNet from TorchVision"""

    def save(self, model: TorchResNetModule, training_id: str) -> NoReturn:
        """Not implemented for this store"""
        raise NotImplementedError("Saving a model to TorchVision is not possible")

    def load(self, model: TorchResNetModule, state_key: StateKey) -> None:
        """Downloads weights from torchvision's repositories

        Arguments:
            model (TorchResNetModule): The Torch ResNet module to load weights
            state_key (StateKey): Identifier of the weights to load
        """
        # Making sure the requested state has been trained on imagenet
        if state_key.training_id != "imagenet":
            raise ValueError(
                "TorchVision state keys should have training_id='imagenet'"
            )
        _validate_resnet_state_type(state_key.state_type)
        # Warnings if the state types are not compatible with the model
        super().load(model, state_key)

        # Getting URL to download model
        url = resnet.model_urls[
            RESNET_ARCHITECTURES_MAP[state_key.state_type.architecture]
        ]
        # Downloading state dictionary
        pretrained_state_dict: OrderedDict[
            str, torch.Tensor
        ] = load_state_dict_from_url(url, map_location=model.device)
        # Splitting features and classifier layers
        features_state_dict = OrderedDict(
            [
                (key, value)
                for key, value in pretrained_state_dict.items()
                if not key.startswith("fc")
            ]
        )
        classifier_state_dict = state_dict_get(pretrained_state_dict, "fc")

        model.load_state_dict(
            state_dict_combine(
                features_module=features_state_dict,
                classifier_module=classifier_state_dict,
            )
        )

    def get_state_keys(self, state_type: StateType) -> List[StateKey]:
        """ImageNet pre-training state_keys

        Attributes:
            state_type (StateType): Filter state keys by type,
                valid state types are given by the
                [`TorchResNetModule.state_type`][mozuma.models.resnet.TorchResNetModule.state_type]
        """
        try:
            _validate_resnet_state_type(state_type)
        except ValueError:
            return []
        else:
            return [StateKey(state_type=state_type, training_id="imagenet")]
