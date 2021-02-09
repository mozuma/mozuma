import pickle
import torch
from functools import partial
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
            #url = "http://places2.csail.mit.edu/models_places365/densenet161_places365.pth.tar"
            #pretrained_state_dict = load_state_dict_from_url(url, check_hash=False)
            pickle.load = partial(pickle.load, encoding="latin1")
            pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

            model_path = 'places_weights/densenet161_places365.pth.tar'
            pretrained_state_dict = torch.load(model_path, map_location=lambda storage, loc: storage, pickle_module=pickle)

            # Correct naming differences in the state dictionary
            pretrained_state_dict = pretrained_state_dict['state_dict']
            pretrained_state_dict = {str.replace(k, 'module.', ''): v for k, v in pretrained_state_dict.items()}
        else:
            url = m.densenet.model_urls[self.densenet_arch]
            pretrained_state_dict = load_state_dict_from_url(url)

        # Weird: there are some naming differences in between the m.densenet() layers and the
        # state dict loaded from the url
        def replace_malformed_string(s):
            s = str.replace(s, 'norm.1', 'norm1')
            s = str.replace(s, 'norm.2', 'norm2')
            s = str.replace(s, 'conv.1', 'conv1')
            s = str.replace(s, 'conv.2', 'conv2')
            return s
        
        pretrained_state_dict = {replace_malformed_string(k): v for k, v in pretrained_state_dict.items()}

        # Removing deleted layers from state dict and updating the other with pretrained data
        return torch_apply_state_to_partial_model(self, pretrained_state_dict)
