import torch
from torchvision import models

# Import the base NN Module for AI Data platform
from mlmodule.torch import BaseTorchMLModule
# Import the mixin for pretrained models
from mlmodule.torch.mixins import TorchPretrainedModuleMixin
# Utility to manipulate torch state dictionary
from mlmodule.torch.utils import torch_apply_state_to_partial_model
# Dataset and transforms for image processing
from mlmodule.torch.data.images import ImageDataset, TORCHVISION_STANDARD_IMAGE_TRANSFORMS


# Class definition for an AI Data Platform module that can be pretrained
class ExampleResNet18Module(BaseTorchMLModule, TorchPretrainedModuleMixin):

    def __init__(self, device=None):
        # Need to pass arguments to the parent class
        super().__init__(device=device)

        # Getting the resnet18 model
        base_resnet = models.resnet18()

        # Copying only the necessary layers
        self.conv1 = base_resnet.conv1
        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool
        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = base_resnet.layer4
        self.avgpool = base_resnet.avgpool

    def forward(self, x):
        # Defining the forward method
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    # This is how we handle the pretrain state
    # We need to provide a state dict for the current module
    # The idea is to download the pretrained model for resnet
    # and filter out the parameters of the layer we have discarded
    def get_default_pretrained_state_dict(self, **options):
        # Getting URL to download model
        # See `mlmodule.contrib.resnet.base` for a better implementation
        pretrained_state_dict = models.resnet18(pretrained=True).state_dict()

        # This function allows to remove unnecessary parameters
        # from the pretrained state dict
        return torch_apply_state_to_partial_model(self, pretrained_state_dict)

    # It is a good practice to redefine the bulk inference to annotate
    # the type of dataset you are expecting as argument.
    # Here we use the ImageDataset which returns PIL images.
    def bulk_inference(self, data: ImageDataset, **data_loader_options):
        return super().bulk_inference(data, **data_loader_options)

    # Defining data transforms to be applied in the data loader
    # Here we are using the standard transforms for torchvision
    def get_dataset_transforms(self):
        return TORCHVISION_STANDARD_IMAGE_TRANSFORMS
