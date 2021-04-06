"""
Base classes for CLIP implementation
"""
from typing import Dict

import clip
import torch
from clip.model import CLIP
from torch import nn

from mlmodule.contrib.clip.parameters import PARAMETERS
from mlmodule.torch import BaseTorchMLModule
from mlmodule.torch.mixins import TorchPretrainedModuleMixin, DownloadPretrainedStateFromProvider
from mlmodule.torch.utils import torch_apply_state_to_partial_model


class BaseCLIPModule(BaseTorchMLModule, TorchPretrainedModuleMixin, DownloadPretrainedStateFromProvider):
    """
    Base class for CLIP modules
    """
    clip_model_name = None
    model_type = None   # image or text

    def __init__(self, device: torch.device = None):
        super().__init__(device=device)
        if self.device == torch.device("cpu"):
            self._dtype = torch.float32
        else:
            self._dtype = torch.float16

    @property
    def url_safe_clip_model_name(self) -> str:
        """Clip model name used in lsir public assets"""
        return self.clip_model_name.lower().replace("/", "")

    @property
    def state_dict_key(self) -> str:
        """Key in LSIR public asset bucket to download model"""
        return f"pretrained-models/" \
               f"{self.model_type}-encoder/" \
               f"clip-{self.url_safe_clip_model_name}-{self.model_type}.pt"

    def convert_weights(self: nn.Module):
        """
        Convert applicable model parameters to fp16

        This needs to be called at the end of the init function once the layers have been defined.

        This makes the layer work as float16 as this is the default when GPU is enabled.

        See https://github.com/openai/CLIP/issues/30
        """

        if self.device != torch.device('cpu'):
            def _convert_weights_to_fp16(layer: nn.Module):
                if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                    layer.weight.data = layer.weight.data.half()
                    if layer.bias is not None:
                        layer.bias.data = layer.bias.data.half()

                if isinstance(layer, nn.MultiheadAttention):
                    for attr in [
                        *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"
                    ]:
                        tensor = getattr(layer, attr)
                        if tensor is not None:
                            tensor.data = tensor.data.half()

                for name in ["text_projection", "proj"]:
                    if hasattr(layer, name):
                        attr = getattr(layer, name)
                        if attr is not None:
                            attr.data = attr.data.half()

            self.apply(_convert_weights_to_fp16)

    @classmethod
    def _get_clip_module(cls) -> CLIP:
        """Returns the CLIP architecture

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
