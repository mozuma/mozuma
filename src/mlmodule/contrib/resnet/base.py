from torch.hub import load_state_dict_from_url
import torchvision.models as m

from mlmodule.torch import BaseTorchMLModule
from mlmodule.torch.mixins import TorchPretrainedModuleMixin
from mlmodule.torch.utils import torch_apply_state_to_partial_model


class BaseResNetModule(BaseTorchMLModule, TorchPretrainedModuleMixin):

    def __init__(self, resnet_arch):
        super().__init__()
        self.resnet_arch = resnet_arch

    @classmethod
    def get_resnet_module(cls, resnet_arch):
        # Getting the ResNet architecture https://pytorch.org/docs/stable/torchvision/models.html
        return getattr(m, resnet_arch)()

    def get_default_pretrained_state_dict(self, map_location=None, cache_dir=None, **options):
        """Returns the state dict for a pretrained resnet model

        :param map_location:
        :param cache_dir:
        :param options:
        :return:
        """
        # Getting URL to download model
        url = m.resnet.model_urls[self.resnet_arch]
        # Downloading state dictionary
        pretrained_state_dict = load_state_dict_from_url(
            url, model_dir=cache_dir, map_location=map_location, **options
        )
        # Removing deleted layers from state dict and updating the other with pretrained data
        return torch_apply_state_to_partial_model(self, pretrained_state_dict)

