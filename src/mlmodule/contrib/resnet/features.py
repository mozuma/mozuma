import torch

from mlmodule.torch.data.images import ImageDataset, TORCHVISION_STANDARD_IMAGE_TRANSFORMS
from mlmodule.contrib.resnet.base import BaseResNetImageNetModule


class BaseResNetImageNetFeatures(BaseResNetImageNetModule):
    """
    ResNet feature extraction for similarity search
    """

    def __init__(self, resnet_arch, device=None):
        super().__init__(resnet_arch, device=device)
        base_resnet = self.get_resnet_module(resnet_arch)

        # Getting only the necessary steps
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
        # Forward without the last step to get features
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

    def bulk_inference(self, data: ImageDataset, batch_size=256, **data_loader_options):
        """Runs inference on all images in a ImageFilesDatasets

        :param data: A dataset returning tuples of item_index, PIL.Image
        :param batch_size:
        :param data_loader_options:
        :return:
        """
        return super().bulk_inference(data, batch_size=256, **data_loader_options)

    def get_dataset_transforms(self):
        return TORCHVISION_STANDARD_IMAGE_TRANSFORMS


class ResNet18ImageNetFeatures(BaseResNetImageNetFeatures):

    def __init__(self, device=None):
        super().__init__("resnet18", device=device)
