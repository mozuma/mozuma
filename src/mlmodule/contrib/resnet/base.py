from typing import Any, Union

import numpy as np
from PIL.Image import Image
from torch.hub import load_state_dict_from_url
import torchvision.models as m

from mlmodule.torch import BaseTorchMLModule
from mlmodule.torch.data.base import IndexedDataset
from mlmodule.torch.mixins import TorchPretrainedModuleMixin
from mlmodule.torch.utils import torch_apply_state_to_partial_model


class BaseResNetImageNetModule(BaseTorchMLModule[IndexedDataset[Any, Any, Union[np.ndarray, Image]]],
                               TorchPretrainedModuleMixin):

    def __init__(self, resnet_arch, device=None):
        super().__init__(device=device)
        self.resnet_arch = resnet_arch

    @classmethod
    def get_resnet_module(cls, resnet_arch):
        # Getting the ResNet architecture https://pytorch.org/docs/stable/torchvision/models.html
        return getattr(m, resnet_arch)()

    def get_default_pretrained_state_dict(self, **_opts):
        """Returns the state dict for a pretrained resnet model
        :return:
        """
        # Getting URL to download model
        url = m.resnet.model_urls[self.resnet_arch]
        # Downloading state dictionary
        pretrained_state_dict = load_state_dict_from_url(url)
        # Removing deleted layers from state dict and updating the other with pretrained data
        return torch_apply_state_to_partial_model(self, pretrained_state_dict)
