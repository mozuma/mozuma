from typing import Dict

import clip
import torch
from clip.model import CLIP
from torch import nn

from mlmodule.contrib.clip.parameters import PARAMETERS
from mlmodule.torch import BaseTorchMLModule
from mlmodule.torch.mixins import TorchPretrainedModuleMixin
from mlmodule.torch.utils import torch_apply_state_to_partial_model


class BaseCLIPModule(BaseTorchMLModule, TorchPretrainedModuleMixin):
    clip_model_name = None

    def __init__(self, device=None):
        super().__init__(device=device)
        if self.device == torch.device("cpu"):
            self._dtype = torch.float32
        else:
            self._dtype = torch.float16

    def convert_weights(self: nn.Module):
        """
        Convert applicable model parameters to fp16

        This needs to be called at the end of the init function once the layers have been defined.

        This makes the layer work as float16 as this is the default when GPU is enabled.

        See https://github.com/openai/CLIP/issues/30
        """

        if self.device != torch.device('cpu'):
            def _convert_weights_to_fp16(l):
                if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                    l.weight.data = l.weight.data.half()
                    if l.bias is not None:
                        l.bias.data = l.bias.data.half()

                if isinstance(l, nn.MultiheadAttention):
                    for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                        tensor = getattr(l, attr)
                        if tensor is not None:
                            tensor.data = tensor.data.half()

                for name in ["text_projection", "proj"]:
                    if hasattr(l, name):
                        attr = getattr(l, name)
                        if attr is not None:
                            attr.data = attr.data.half()

            self.apply(_convert_weights_to_fp16)

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
