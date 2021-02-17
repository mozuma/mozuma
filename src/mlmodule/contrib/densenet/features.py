import torch
import torch.nn.functional as F

from mlmodule.torch.data.images import TORCHVISION_STANDARD_IMAGE_TRANSFORMS
from mlmodule.contrib.densenet.base import BaseDenseNetPretrainedModule


class BaseDenseNetPretrainedFeatures(BaseDenseNetPretrainedModule):
    """
    DenseNet feature extraction for similarity search
    """

    def __init__(self, densenet_arch, dataset="imagenet", device=None):
        super().__init__(densenet_arch, dataset=dataset, device=device)
        base_densenet = self.get_densenet_module(densenet_arch)

        # Set the state_dict_key
        if dataset == "places":
            self.state_dict_key = "pretrained-models/image-classification/places365/densenet161_features.pth.tar"
        else:
            self.state_dict_key = "pretrained-models/image-classification/imagenet/densenet161_features.pth.tar"

        # TODO: Layer output selection
        """
        # Getting only the necessary steps
        self.conv0 = base_densenet.features.conv0
        self.norm0 = base_densenet.features.norm0
        self.relu0 = base_densenet.features.relu0
        self.pool0 = base_densenet.features.pool0

        self.block1 = base_densenet.features.denseblock1
        self.transition1 = base_densenet.features.transition1
        self.block2 = base_densenet.features.denseblock2
        self.transition2 = base_densenet.features.transition2
        self.block3 = base_densenet.features.denseblock3
        """

        self.features = base_densenet.features
        self.relu = F.relu
        self.avgpool = F.adaptive_avg_pool2d

    def forward(self, x):
        # Forward without the last step to get features
        fs = self.features(x)
        out = self.relu(fs, inplace=True)
        out = self.avgpool(out, (1, 1))
        return torch.flatten(out, 1)

    def get_dataset_transforms(self):
        return TORCHVISION_STANDARD_IMAGE_TRANSFORMS

    """
    TODO: Correct that I don't need to reimplement bulk_inference?
    """


class DenseNet161ImageNetFeatures(BaseDenseNetPretrainedFeatures):

    def __init__(self, device=None):
        super().__init__("densenet161", dataset="imagenet", device=device)


class DenseNet161PlacesFeatures(BaseDenseNetPretrainedFeatures):

    def __init__(self, device=None):
        super().__init__("densenet161", dataset="places", device=device)
