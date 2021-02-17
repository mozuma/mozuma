from mlmodule.contrib.densenet.base import BaseDenseNetPretrainedModule
from mlmodule.labels import LabelsMixin, ImageNetLabels, PlacesLabels

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

        # Set the state_dict_key
        if dataset == "places":
            self.state_dict_key = "pretrained-models/image-classification/places365/densenet161_classifier.pth.tar"
        else:
            self.state_dict_key = "pretrained-models/image-classification/imagenet/densenet161_classifier.pth.tar"

    def forward(self, x):
        """Forward pass

        :param x: Should be the output of DenseNetFeatures.forward
        :return:
        """
        return self.classifier(x)

    def get_labels(self):
        return PlacesLabels() if self.dataset == "places" else ImageNetLabels()


class DenseNet161ImageNetClassifier(BaseDenseNetPretrainedClassifier):

    def __init__(self, device=None):
        super().__init__("densenet161", dataset="imagenet", device=device)


class DenseNet161PlacesClassifier(BaseDenseNetPretrainedClassifier):

    def __init__(self, device=None):
        super().__init__("densenet161", dataset="places", device=device)
