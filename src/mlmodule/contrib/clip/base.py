from typing import Dict

import clip
import torch
from clip.model import CLIP

from mlmodule.contrib.clip.parameters import PARAMETERS
from mlmodule.torch import BaseTorchMLModule
from mlmodule.torch.mixins import TorchPretrainedModuleMixin
from mlmodule.torch.utils import torch_apply_state_to_partial_model


class BaseCLIPModule(BaseTorchMLModule, TorchPretrainedModuleMixin):
    clip_model_name = None

    @classmethod
    def _get_clip_module(cls) -> CLIP:
        """Returns the CLIP architecture

        :param model_name:
        :return:
        """
        return CLIP(*PARAMETERS[cls.clip_model_name].values())

    def get_default_pretrained_state_dict_from_provider(self) -> Dict[str, torch.Tensor]:
        """Get the pretrained state dictionary directly from CLIP repository

        :return:
        """
        clip_pretrained, _ = clip.load(self.clip_model_name, jit=False)
        partial_state_dict = torch_apply_state_to_partial_model(self, clip_pretrained.state_dict())

        return partial_state_dict
