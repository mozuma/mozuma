import torch
import torchvision.models as m

from mlmodule.torch import BaseTorchMLModule
from mlmodule.torch.mixins import TorchPretrainedModuleMixin


class BaseDenseNetPretrainedModule(BaseTorchMLModule, TorchPretrainedModuleMixin):

    def __init__(self, densenet_arch, dataset="imagenet", device=None):
        """
        
        :param densenet_arch: One of {"densenet121", "densenet161", "densenet169", "densenet201"}
        :param dataset: One of {"imagenet", "places365"}.
        """
        super().__init__(device=device)

        # Check input parameters
        assert densenet_arch in ["densenet121", "densenet161", "densenet169", "densenet201"]
        assert dataset in ["imagenet", "places365"]

        self.densenet_arch = densenet_arch
        self.dataset = dataset
        self.device = device

    @classmethod
    def get_densenet_module(cls, densenet_arch, num_classes=1000):
        # Getting the DenseNet architecture https://pytorch.org/docs/stable/torchvision/models.html
        return getattr(m, densenet_arch)(num_classes=num_classes)
