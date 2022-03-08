from typing import Callable, List

import torch
from torch import nn
from torchvision import transforms

from mlmodule.contrib.arcface.transforms import ArcFaceAlignment
from mlmodule.torch.modules import IBasicBlock, conv1x1
from mlmodule.v2.base.predictions import BatchModelPrediction
from mlmodule.v2.states import StateType
from mlmodule.v2.torch.modules import TorchMlModule


class TorchMagFaceModule(TorchMlModule[torch.Tensor, torch.Tensor]):
    """MagFace face embeddings from MTCNN detected faces

    The input dataset should return a tuple of image data and bounding box information

    Attributes:
        device (torch.device): Torch device to initialise the model weights
        remove_bad_faces (bool): Whether to remove the faces with bad quality from the output
        magnitude_threshold (float): Threshold to remove bad quality faces.
            The higher the stricter. Defaults to `22.5`.
    """

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        zero_init_residual: bool = False,
        remove_bad_faces: bool = True,
        magnitude_threshold: float = 22.5,
    ):
        super().__init__(device=device)
        self.fc_scale = 7 * 7
        self.magnitude_threshold = magnitude_threshold
        self.remove_bad_faces = remove_bad_faces

        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=2e-05, momentum=0.9)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(IBasicBlock, 64, 3, stride=2)
        self.layer2 = self._make_layer(IBasicBlock, 128, 13, stride=2, dilate=False)
        self.layer3 = self._make_layer(IBasicBlock, 256, 30, stride=2, dilate=False)
        self.layer4 = self._make_layer(IBasicBlock, 512, 3, stride=2, dilate=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.bn2 = nn.BatchNorm2d(512 * IBasicBlock.expansion, eps=2e-05, momentum=0.9)
        self.dropout = nn.Dropout2d(p=0.4, inplace=True)
        self.fc = nn.Linear(512 * IBasicBlock.expansion * self.fc_scale, 512)
        self.features = nn.BatchNorm1d(512, eps=2e-05, momentum=0.9)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=2e-05, momentum=0.9),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                )
            )

        return nn.Sequential(*layers)

    @property
    def state_type(self) -> StateType:
        return StateType(backend="pytorch", architecture="magface")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MagFace model

        Arguments:
            x (torch.Tensor): Batch of cropped and aligned faces

        Returns:
            torch.Tensor: The features for each input crop.
                A feature can be set to `nan` if the face is of bad quality and `self.remove_bad_faces == True`.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.features(x)

        if self.remove_bad_faces:
            # Remove bad quality faces, setting them to NaN
            x[torch.norm(x, dim=1) < self.magnitude_threshold] = float("nan")

        return x

    def forward_predictions(
        self, batch: torch.Tensor
    ) -> BatchModelPrediction[torch.Tensor]:
        """Forward pass

        Arguments:
            batch (torch.Tensor): The face cropped and aligned with `ArcFaceAlignment`

        Returns:
            torch.Tensor: A `BatchModelPrediction` with `features` attribute as face embeddings.
                A feature can be set to `nan` if the face is of bad quality and `self.remove_bad_faces == True`.
        """
        return BatchModelPrediction(features=self.forward(batch))

    def get_dataset_transforms(self) -> List[Callable]:
        """Returns transforms to be applied on bulk_inference input data"""
        return [
            ArcFaceAlignment(),
            transforms.ToTensor(),
            transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        ]
