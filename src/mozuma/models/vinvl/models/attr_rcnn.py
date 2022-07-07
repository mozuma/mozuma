# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
"""
Implements the FRCNN with Attribute Head
"""
from collections import OrderedDict

import torch.nn as nn

from mozuma.models.vinvl.models.attribute_head.attribute_head import ROIAttributeHead
from mozuma.models.vinvl.models.resnet import ResNet
from mozuma.models.vinvl.models.roi_heads import CombinedROIHeads, ROIBoxHead
from mozuma.models.vinvl.models.rpn import RPNModule
from mozuma.models.vinvl.models.structures.bounding_box import BoxList
from mozuma.models.vinvl.models.structures.image_list import to_image_list


class AttrRCNN(nn.Module):
    """
    Main class for Generalized Relation R-CNN.
    It consists of three main parts:
    - backbone
    - rpn
    - object detection (roi_heads)
    - Scene graph parser model: IMP, MSDN, MOTIF, graph-rcnn, ect
    """

    def __init__(self, cfg, device="cpu"):
        # GeneralizedRCNN.__init__(self, cfg)
        super(AttrRCNN, self).__init__()
        body = ResNet(cfg)
        model = nn.Sequential(OrderedDict([("body", body)]))
        model.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
        self.backbone = model
        self.rpn = RPNModule(cfg, self.backbone.out_channels)
        self.roi_heads = CombinedROIHeads(
            cfg, [("box", ROIBoxHead(cfg, self.backbone.out_channels))]
        )
        self.force_boxes = cfg.MODEL.RPN.FORCE_BOXES

        self.cfg = cfg
        self.device = device
        feature_dim = self.backbone.out_channels

        if cfg.MODEL.ATTRIBUTE_ON:
            self.attribute = ROIAttributeHead(cfg, feature_dim)
            if cfg.MODEL.ROI_ATTRIBUTE_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                self.attribute.feature_extractor = self.roi_heads.box.feature_extractor

    def to(self, device, **kwargs):
        super(AttrRCNN, self).to(device, **kwargs)
        if self.cfg.MODEL.ATTRIBUTE_ON:
            self.attribute.to(device, **kwargs)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)
            We can assume that gt_boxlist contains two other fields:
                "relation_labels": list of [subj_id, obj_id, predicate_category]
                "pred_labels": n*n matrix with predicate_category (including BG) as values.

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.force_boxes and targets is None:
            # note targets cannot be None but could have 0 box.
            raise ValueError("In force_boxes setting, targets should be passed")

        images = to_image_list(images)
        images = images.to(self.device)
        features = self.backbone(images.tensors)

        if targets:
            targets = [
                target.to(self.device) for target in targets if target is not None
            ]

        if self.force_boxes:
            proposals = [
                BoxList(target.bbox, target.size, target.mode) for target in targets
            ]
            if self.training:
                # note we still need to compute a loss using all rpn
                # named parameters, otherwise it will
                # give unused_parameters error in distributed training.
                null_loss = 0
                for key, param in self.rpn.named_parameters():
                    null_loss += 0.0 * param.sum()
                proposal_losses = {"rpn_null_loss", null_loss}
        else:
            proposals, proposal_losses = self.rpn(images, features, targets)

        x, predictions, detector_losses = self.roi_heads(features, proposals, targets)

        if self.cfg.MODEL.ATTRIBUTE_ON:
            attribute_features = features
            # the attribute head reuse the features from the box head
            if (
                self.training
                and self.cfg.MODEL.ROI_ATTRIBUTE_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                attribute_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x_attr, predictions, loss_attribute = self.attribute(
                attribute_features, predictions, targets
            )
            detector_losses.update(loss_attribute)

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return predictions
