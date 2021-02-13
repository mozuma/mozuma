import torch
from torch.hub import load_state_dict_from_url
import torchvision.models as m

from mlmodule.torch import BaseTorchMLModule
from mlmodule.torch.mixins import TorchPretrainedModuleMixin
from mlmodule.torch.utils import torch_apply_state_to_partial_model


class BaseDenseNetPretrainedModule(BaseTorchMLModule, TorchPretrainedModuleMixin):

    def __init__(self, densenet_arch, dataset="imagenet", device=None):
        """
        
        :param densenet_arch: One of {"densenet121", "densenet161", "densenet169", "densenet201"}
        :param dataset: One of {"imageNet", "places"}. If "places", then densenet_arch must be
            "densenet161".
        """
        super().__init__(device=device)
        self.densenet_arch = densenet_arch
        self.dataset = dataset
        self.device = device

    @classmethod
    def get_densenet_module(cls, densenet_arch, num_classes=1000):
        # Getting the DenseNet architecture https://pytorch.org/docs/stable/torchvision/models.html
        return getattr(m, densenet_arch)(num_classes=num_classes)

    def get_default_pretrained_state_dict(self):
        """Returns the state dict for a pretrained densenet model
        :return:
        """
        # Downloading state dictionary
        if self.dataset == "places":
            # The following works in torch 1.7.1
            #pretrained_state_dict = load_state_dict_from_url(url, map_location=lambda storage, loc: storage, check_hash=False)
            
            # TODO: How to determine if weights for the features or classifier need to be loaded?
             
            model_path = 'places_weights/densenet161_places365.pth.tar'
            pretrained_state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

            # Correct naming differences in the state dictionary
            pretrained_state_dict = pretrained_state_dict['state_dict']
            pretrained_state_dict = {str.replace(k, 'module.', ''): v for k, v in pretrained_state_dict.items()}
        else:
            url = m.densenet.model_urls[self.densenet_arch]
            pretrained_state_dict = load_state_dict_from_url(url)

        # Removing deleted layers from state dict and updating the other with pretrained data
        return torch_apply_state_to_partial_model(self, pretrained_state_dict)
