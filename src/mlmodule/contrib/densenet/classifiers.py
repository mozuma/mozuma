import torch
import torchvision.models as m
from torch.hub import load_state_dict_from_url

from mlmodule.contrib.densenet.base import BaseDenseNetPretrainedModule
from mlmodule.labels import LabelsMixin, ImageNetLabels, PlacesLabels
from mlmodule.torch.utils import torch_apply_state_to_partial_model

class BaseDenseNetPretrainedClassifier(BaseDenseNetPretrainedModule, LabelsMixin):
    """
    Default fully connected layer for classification before retraining
    """

    def __init__(self, densenet_arch, dataset="imagenet", device=None):
        super().__init__(densenet_arch, dataset=dataset, device=device)
        if dataset == "places":
            base_densenet = self.get_densenet_module(densenet_arch, num_classes=365)
        else:
            base_densenet = self.get_densenet_module(densenet_arch, num_classes=1000)
        
        # Getting only the fully connected layer
        self.classifier = base_densenet.classifier

    def forward(self, x):
        """Forward pass

        :param x: Should be the output of DenseNetFeatures.forward
        :return:
        """
        return self.classifier(x)

    def get_labels(self):
        return PlacesLabels() if self.dataset == "places" else ImageNetLabels()

    def get_default_pretrained_state_dict(self):
        """Returns the state dict for a pretrained densenet model
        :return:
        """
        # Downloading state dictionary
        if self.dataset == "places":
            # URL: "http://places2.csail.mit.edu/models_places365/densenet161_places365.pth.tar"
            model_path = 'densenet161_places_classifier.pth.tar'
        else:
            # URL: https://download.pytorch.org/models/densenet161-8d451a50.pth
            model_path = 'densenet161_imagenet_classifier.pth.tar'
        
        pretrained_state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

        # Removing deleted layers from state dict and updating the other with pretrained data
        return torch_apply_state_to_partial_model(self, pretrained_state_dict)


class DenseNet161ImageNetClassifier(BaseDenseNetPretrainedClassifier):

    def __init__(self, device=None):
        super().__init__("densenet161", dataset="imagenet", device=device)


class DenseNet161PlacesClassifier(BaseDenseNetPretrainedClassifier):

    def __init__(self, device=None):
        super().__init__("densenet161", dataset="places", device=device)
