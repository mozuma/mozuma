from mlmodule.contrib.resnet.base import BaseResNetImageNetModule
from mlmodule.labels import LabelsMixin, ImageNetLabels, PlacesLabels


class BaseDenseNetPretrainedClassifier(BaseDenseNetPretrainedModule, LabelsMixin):
    """
    Default fully connected layer for classification before retraining
    """

    def __init__(self, densenet_arch, dataset="imagenet", device=None):
        super().__init__(densenet_arch, dataset=dataset, device=device)
        base_densenet = self.get_densenet_module(densenet_arch)

        # Getting only the fully connected layer
        self.fc = base_densenet.classifier

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
