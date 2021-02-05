import torch
import torch.nn.functional as F

from mlmodule.torch.data.images import ImageDataset, TORCHVISION_STANDARD_IMAGE_TRANSFORMS
from mlmodule.contrib.densenet.base import BaseDenseNetPretrainedModule


class BaseDenseNetPretrainedFeatures(BaseDenseNetPretrainedModule):
    """
    DenseNet feature extraction for similarity search
    """

    def __init__(self, densenet_arch, dataset="imagenet", device=None):
        super().__init__(densenet_arch, dataset=dataset, device=device)
        base_densenet = self.get_densenet_module(densenet_arch)

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
        self.avgpool = F.avg_pool2d
        self.avgpool_kernel_size = 7

    def forward(self, x):
        # Forward without the last step to get features
        x = self.features(x)
        x = self.avgpool(x, self.avgpool_kernel_size)

        return torch.flatten(x, 1)

    def get_dataset_transforms(self):
        return TORCHVISION_STANDARD_IMAGE_TRANSFORMS

    """
    TODO: Correct that I don't need to reimplement bulk_inference?
    TODO: I believe the dataset transforms should be the same for the resnet and densenet, is that correct?
    The Pre-trained places models also seem to use the same image Transforms:
        Documentation:
            https://github.com/CSAILVision/places365

        Basic code for scene prediction:
            https://github.com/CSAILVision/places365/blob/master/run_placesCNN_basic.py

            # load the image transformer
            centre_crop = trn.Compose([
                trn.Resize((256,256)),
                trn.CenterCrop(224),
                trn.ToTensor(),
                trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    """


class DenseNet161ImageNetFeatures(BaseDenseNetPretrainedFeatures):

    def __init__(self, device=None):
        super().__init__("densenet161", dataset="imagenet", device=device)


class DenseNet161PlacesFeatures(BaseDenseNetPretrainedFeatures):

    def __init__(self, device=None):
        super().__init__("densenet161", dataset="places", device=device)
