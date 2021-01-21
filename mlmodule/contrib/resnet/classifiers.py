from fastai.layers import Flatten
from torch import nn

from mlmodule.contrib.resnet.base import BaseResNetModule


class ResNetDefaultClassifier(BaseResNetModule):
    """
    Default fully connected layer for classification before retraining
    """

    def __init__(self, resnet_arch):
        super().__init__()
        # Disabling all but the fully connected layer
        self.model.conv1 = None
        self.model.bn1 = None
        self.model.relu = None
        self.model.maxpool = None
        self.model.layer1 = None
        self.model.layer2 = None
        self.model.layer3 = None
        self.model.layer4 = None
        self.model.avgpool = None

    def forward(self, x):
        return self.model.fc(x)


class ResNetReTrainClassifier(nn.Module):
    """
    Custom classifier for retraining
    """

    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            # Tried to reproduce fastai's classifier head, But didn't see meaningful improvements from
            # AdapativeConcatPool2d, so kept avgpool in previous layer to keep backwards compat with our code
            # AdaptiveConcatPool2d(),
            # nn.AdaptiveAvgPool2d(output_size=1),
            # nn.AdaptiveMaxPool2d(output_size=1),
            Flatten(),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=512, out_features=512, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=num_class, bias=True),
        )

    def forward(self, x):
        return self.classifier(x)
