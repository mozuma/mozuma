from torch import nn

from mlmodule.contrib.resnet.base import BaseResNetModule
from mlmodule.torch import BaseTorchMLModule


class ResNetDefaultClassifier(BaseResNetModule):
    """
    Default fully connected layer for classification before retraining
    """

    def __init__(self, resnet_arch):
        super().__init__(resnet_arch)
        base_resnet = self.get_resnet_module(resnet_arch)

        # Getting only the fully connected layer
        self.fc = base_resnet.fc

    def forward(self, x):
        """Forward pass

        :param x: Should be the output of ResNetFeatures.forward
        :return:
        """
        return self.fc(x)


class ResNetReTrainClassifier(BaseTorchMLModule):
    """
    Custom classifier for retraining
    """

    def __init__(self, num_class):
        super().__init__()
        self.classifier = nn.Sequential(
            # Tried to reproduce fastai's classifier head, But didn't see meaningful improvements from
            # AdapativeConcatPool2d, so kept avgpool in previous layer to keep backwards compat with our code
            # AdaptiveConcatPool2d(),
            # nn.AdaptiveAvgPool2d(output_size=1),
            # nn.AdaptiveMaxPool2d(output_size=1),
            nn.Flatten(),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=512, out_features=512, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=num_class, bias=True),
        )

    def forward(self, x):
        """Forward pass

        :param x: Should be the output of ResNetFeatures.forward
        :return:
        """
        return self.classifier(x)
