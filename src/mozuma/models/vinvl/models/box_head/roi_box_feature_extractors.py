# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
import torch
from torch import nn

from mozuma.models.vinvl.models.poolers import Pooler
from mozuma.models.vinvl.models.resnet import ResNetHead, StageSpec


class ResNet50Conv5ROIFeatureExtractor(nn.Module):
    def __init__(self, config, in_channels):
        super(ResNet50Conv5ROIFeatureExtractor, self).__init__()

        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        # ResNet50Conv5 input size is 1024, which is the default for backbone
        # without FPN. For FPN structure, the in_channels will be 256. So we
        # need to add a transit conv layer to make it compatible.
        # The corresponding predictor should be FastRCNNPredictor.
        if in_channels != 1024:
            self.trans_conv = nn.Conv2d(in_channels, 1024, kernel_size=3, padding=1)
            torch.nn.init.normal_(self.trans_conv.weight, std=0.01)
            torch.nn.init.constant_(self.trans_conv.bias, 0)
        else:
            self.trans_conv = None

        stage = StageSpec(index=4, block_count=3, return_features=False)
        head = ResNetHead(
            stages=(stage,),
            num_groups=config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=config.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=config.MODEL.RESNETS.RES5_DILATION,
        )

        self.pooler = pooler
        self.head = head
        self.out_channels = head.out_channels

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        if self.trans_conv:
            x = self.trans_conv(x)
        x = self.head(x)
        return x
