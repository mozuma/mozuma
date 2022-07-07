import abc
from typing import OrderedDict

import torch
from torch import nn
from typing_extensions import Literal

from mozuma.models.clip.utils import sanitize_clip_model_name
from mozuma.predictions import BatchModelPrediction
from mozuma.states import StateType
from mozuma.torch.modules import TorchModel


class BaseCLIPModule(TorchModel[torch.Tensor, torch.Tensor]):
    """Base class for CLIP modules

    Attributes:
        clip_model_name (str): Name of the model to load
            (see [CLIP doc](https://github.com/openai/CLIP#clipavailable_models))
        model_type (Literal["image", "text"]): Load text or image encoder
        device (torch.device, optional): The PyTorch device to initialise the model weights.
            Defaults to `torch.device("cpu")`.
    """

    def __init__(
        self,
        clip_model_name: str,
        model_type: Literal["image", "text"],
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(device=device, is_trainable=False)
        self.clip_model_name = clip_model_name
        self.model_type = model_type

        # Setting dtype depending on device
        if self.device == torch.device("cpu"):
            self._dtype = torch.float32
        else:
            self._dtype = torch.float16

    @property
    def clip_model_name_safe(self) -> str:
        return sanitize_clip_model_name(self.clip_model_name)

    @property
    def state_type(self) -> StateType:
        return StateType(
            backend="pytorch",
            architecture=f"clip-{self.model_type}-{self.clip_model_name_safe}",
        )

    def convert_weights(self):
        """
        Convert applicable model parameters to fp16

        This needs to be called at the end of the init function once the layers have been defined.

        This makes the layer work as float16 as this is the default when GPU is enabled.

        See [OpenAI/CLIP#30](https://github.com/openai/CLIP/issues/30)
        """

        if self.device != torch.device("cpu"):

            def _convert_weights_to_fp16(layer: nn.Module):
                if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                    layer.weight.data = layer.weight.data.half()
                    if layer.bias is not None:
                        layer.bias.data = layer.bias.data.half()

                if isinstance(layer, nn.MultiheadAttention):
                    for attr_name in [
                        *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                        "in_proj_bias",
                        "bias_k",
                        "bias_v",
                    ]:
                        tensor: torch.Tensor = getattr(layer, attr_name)
                        if tensor is not None:
                            tensor.data = tensor.data.half()

                for name in ["text_projection", "proj"]:
                    if hasattr(layer, name):
                        attr: torch.Tensor = getattr(layer, name)
                        if attr is not None:
                            attr.data = attr.data.half()

            self.apply(_convert_weights_to_fp16)

    def to_predictions(
        self, forward_output: torch.Tensor
    ) -> BatchModelPrediction[torch.Tensor]:
        """CLIP forward returns features for the image or text module"""
        return BatchModelPrediction(features=forward_output)

    @abc.abstractmethod
    def load_full_clip_state_dict(self, state_dict: OrderedDict[str, torch.Tensor]):
        """Loads the model weights from the original CLIP model weights"""
