from collections import OrderedDict
from typing import List, NoReturn

import torch
from torch.hub import load_state_dict_from_url
from torchvision.models import resnet

from mlmodule.contrib.resnet.modules import TorchResNetImageNetModule
from mlmodule.contrib.resnet.utils import sanitize_resnet_arch
from mlmodule.v2.states import StateKey, StateType
from mlmodule.v2.stores.abstract import AbstractStateStore

RESNET_ARCHITECTURES_MAP = {sanitize_resnet_arch(a): a for a in resnet.model_urls}


class ResNetTorchVisionStore(AbstractStateStore[TorchResNetImageNetModule]):
    """Model store to load ResNet weights pretrained on ImageNet from TorchVision"""

    def _valid_resnet_state_type(self, state_type: StateType) -> None:
        if state_type.backend != "pytorch":
            raise ValueError("TorchVision state type should have backend='pytorch'")
        if state_type.architecture not in RESNET_ARCHITECTURES_MAP:
            raise ValueError(
                f"ResNet state type architecture {state_type.architecture} not found in TorchVision"
            )

    def save(self, model: TorchResNetImageNetModule, training_id: str) -> NoReturn:
        """Not implemented for this store"""
        raise NotImplementedError("Saving a model to TorchVision is not possible")

    def load(self, model: TorchResNetImageNetModule, state_key: StateKey) -> None:
        """Downloads weights from torchvision's repositories

        Arguments:
            model (TorchResNetImageNetModule): The Torch ResNet module to load weights
            state_key (StateKey): Identifier of the weights to load
        """
        # Making sure the requested state has been trained on imagenet
        if state_key.training_id != "imagenet":
            raise ValueError(
                "TorchVision state keys should have training_id='imagenet'"
            )
        self._valid_resnet_state_type(state_key.state_type)
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
        classifier_state_dict = OrderedDict(
            [
                (key[3:], value)
                for key, value in pretrained_state_dict.items()
                if key.startswith("fc")
            ]
        )
        model.features_module.load_state_dict(features_state_dict)
        model.classifier_module.load_state_dict(classifier_state_dict)

    def get_state_keys(self, state_type: StateType) -> List[StateKey]:
        """ImageNet pre-training state_keys

        Attributes:
            state_type (StateType): Filter state keys by type,
                valid state types are given by the
                [`TorchResNetImageNetModule.state_type`][mlmodule.contrib.resnet.TorchResNetImageNetModule.state_type]
        """
        self._valid_resnet_state_type(state_type)
        return [StateKey(state_type=state_type, training_id="imagenet")]
