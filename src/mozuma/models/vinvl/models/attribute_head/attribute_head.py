# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
import torch

from mozuma.models.vinvl.models.attribute_head.inference import AttributePostProcessor
from mozuma.models.vinvl.models.attribute_head.roi_attribute_predictors import (
    AttributeRCNNPredictor,
)
from mozuma.models.vinvl.models.box_head.roi_box_feature_extractors import (
    ResNet50Conv5ROIFeatureExtractor,
)


class ROIAttributeHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIAttributeHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = ResNet50Conv5ROIFeatureExtractor(cfg, in_channels)
        self.predictor = AttributeRCNNPredictor(
            cfg, self.feature_extractor.out_channels
        )
        self.post_processor = AttributePostProcessor(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from box_head
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the attribute feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `attribute` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        # deal with the case when len(proposals)==0 in the inference time;
        # we assume that this will not happen in training
        num_dets = [len(box) for box in proposals]
        if sum(num_dets) == 0:
            return features, proposals, {}

        x = self.feature_extractor(features, proposals)

        labels = torch.cat(
            [
                boxes_per_image.get_field("labels").view(-1)
                for boxes_per_image in proposals
            ],
            dim=0,
        )
        attribute_logits, attribute_features = self.predictor(x, labels)

        result = self.post_processor(attribute_logits, proposals, attribute_features)
        return x, result, {}
