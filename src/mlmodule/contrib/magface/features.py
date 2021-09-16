from typing import List, Callable, Any, Tuple, Union
import pickle
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from mlmodule.contrib.arcface.transforms import ArcFaceAlignment
from mlmodule.torch import BaseTorchMLModule
from mlmodule.torch.data.box import BoundingBoxDataset
from mlmodule.torch.data.images import transforms
from mlmodule.torch.mixins import DownloadPretrainedStateFromProvider
from mlmodule.torch.modules import IBasicBlock, conv1x1


# See https://arxiv.org/pdf/2103.06627.pdf (Figure 5)
MAGFACE_MAGNITUDE_THRESHOLD = 22.5


class MagFaceFeatures(BaseTorchMLModule[BoundingBoxDataset],
                      DownloadPretrainedStateFromProvider):
    """Creates face embeddings from MTCNN output"""

    state_dict_key = 'pretrained-models/face-detection/magface_epoch_00025.pth'
    fc_scale = 7 * 7

    def __init__(self, device: torch.device = None, zero_init_residual: bool = False):
        super().__init__(device=device)

        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=2e-05, momentum=0.9)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(IBasicBlock, 64, 3, stride=2)
        self.layer2 = self._make_layer(IBasicBlock, 128, 13, stride=2,
                                       dilate=False)
        self.layer3 = self._make_layer(IBasicBlock, 256, 30, stride=2,
                                       dilate=False)
        self.layer4 = self._make_layer(IBasicBlock, 512, 3, stride=2,
                                       dilate=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.bn2 = nn.BatchNorm2d(
            512 * IBasicBlock.expansion, eps=2e-05, momentum=0.9)
        self.dropout = nn.Dropout2d(p=0.4, inplace=True)
        self.fc = nn.Linear(512 * IBasicBlock.expansion * self.fc_scale, 512)
        self.features = nn.BatchNorm1d(512, eps=2e-05, momentum=0.9)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
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
                nn.BatchNorm2d(planes * block.expansion,
                               eps=2e-05, momentum=0.9),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
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

        return x

    def _torch_load(self, f, map_location=None, pickle_module=pickle, **pickle_load_args) -> Any:
        """Safe method to load the state dict directly on the right device"""
        map_location = map_location or self.device
        state_dict = torch.load(f, map_location=map_location, pickle_module=pickle_module, **pickle_load_args)
        cleaned_state_dict = OrderedDict()
        for k, v in state_dict['state_dict'].items():
            if k[0:16] == 'features.module.':
                new_k = '.'.join(k.split('.')[2:])
                cleaned_state_dict[new_k] = v
        return cleaned_state_dict

    def bulk_inference(
            self, data: BoundingBoxDataset,
            remove_bad_quality_faces=True,
            **opts
    ) -> Tuple[List, Union[List, np.ndarray]]:
        indices, features = super().bulk_inference(
            data, **opts
        )
        if remove_bad_quality_faces:
            # Filter for faces with good quality
            good_faces = np.linalg.norm(features, axis=1) > MAGFACE_MAGNITUDE_THRESHOLD
            return np.array(indices)[good_faces].tolist(), features[good_faces]
        else:
            return indices, features

    def get_dataset_transforms(self) -> List[Callable]:
        """Returns transforms to be applied on bulk_inference input data"""
        return [
            ArcFaceAlignment(),
            transforms.ToTensor(),
            transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        ]
