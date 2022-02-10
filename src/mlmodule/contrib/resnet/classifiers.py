import torch

from mlmodule.contrib.resnet.base import BaseResNetImageNetModule
from mlmodule.labels.imagenet import IMAGENET_LABELS


class BaseResNetImageNetClassifier(BaseResNetImageNetModule):
    """
    Default fully connected layer for classification before retraining
    """

    def __init__(self, resnet_arch, device=None):
        super().__init__(resnet_arch, device=device)
        base_resnet = self.get_resnet_module(resnet_arch)

        # Getting only the fully connected layer
        self.fc = base_resnet.fc

    def forward(self, x):
        """Forward pass

        :param x: Should be the output of ResNetFeatures.forward
        :return:
        """
        return self.fc(x)

    def get_labels(self):
        return IMAGENET_LABELS

    def get_dataset_transforms(self):
        return [torch.from_numpy]


class ResNet18ImageNetClassifier(BaseResNetImageNetClassifier):
    def __init__(self, device=None):
        super().__init__("resnet18", device=device)
