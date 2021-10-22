from typing import Dict, Optional, Tuple, List, TypeVar, Callable

import torch

from mlmodule.contrib.vinvl.config import sg_cfg
from mlmodule.contrib.vinvl.models.config import cfg
from mlmodule.contrib.vinvl.transforms import Resize, ToTensor, Normalize
from mlmodule.contrib.vinvl.collator import BatchCollator

from mlmodule.contrib.vinvl.models import AttrRCNN
from mlmodule.box import BBoxCollection, BBoxOutputArrayFormat
from mlmodule.torch.base import MLModuleDatasetProtocol, TorchMLModuleBBox
from mlmodule.torch.mixins import DownloadPretrainedStateFromProvider, \
    ResizableImageInputMixin
from mlmodule.torch.utils import torch_apply_state_to_partial_model
from mlmodule.types import ImageDatasetType
from mlmodule.labels import LabelsMixin, VinVLLabels, VinVLAttributeLabels


_IndexType = TypeVar('_IndexType', contravariant=True)

STATE_DICT_URL = 'https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/vinvl_vg_x152c4.pth'


class VinVLDetector(TorchMLModuleBBox[
    _IndexType,
    ImageDatasetType
],
        DownloadPretrainedStateFromProvider,
        ResizableImageInputMixin, LabelsMixin):
    """Face detection module"""

    state_dict_key = "pretrained-models/object-detection/vinvl_vg_x152c4.pt"

    def __init__(self, score_threshold=0.5, attr_score_threshold=0.5, device=None):
        super().__init__(device=device)
        cfg.set_new_allowed(True)
        cfg.merge_from_other_cfg(sg_cfg)
        cfg.set_new_allowed(False)
        vinvl_x152c4_config = [
            'MODEL.META_ARCHITECTURE', "AttrRCNN",
            'MODEL.BACKBONE.CONV_BODY', "R-152-C4",
            'MODEL.RESNETS.BACKBONE_OUT_CHANNELS', 1024,
            'MODEL.RESNETS.STRIDE_IN_1X1', False,
            'MODEL.RESNETS.NUM_GROUPS', 32,
            'MODEL.RESNETS.WIDTH_PER_GROUP', 8,
            'MODEL.RPN.PRE_NMS_TOP_N_TEST', 6000,
            'MODEL.RPN.POST_NMS_TOP_N_TEST', 300,
            'MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE', 384,
            'MODEL.ROI_HEADS.POSITIVE_FRACTION', 0.5,
            'MODEL.ROI_HEADS.SCORE_THRESH', 0.2,
            'MODEL.ROI_HEADS.DETECTIONS_PER_IMG', 100,
            'MODEL.ROI_HEADS.MIN_DETECTIONS_PER_IMG', 10,
            'MODEL.ROI_HEADS.NMS_FILTER', 1,
            'MODEL.ROI_HEADS.SCORE_THRESH', score_threshold,
            'MODEL.ROI_BOX_HEAD.NUM_CLASSES', 1595,
            'MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES', 525,
            'MODEL.ROI_ATTRIBUTE_HEAD.POSTPROCESS_ATTRIBUTES_THRESHOLD', 0.05,
            'MODEL.ATTRIBUTE_ON', True,
            'INPUT.MIN_SIZE_TEST', 600,
            'INPUT.MAX_SIZE_TEST', 1000,
            'INPUT.PIXEL_MEAN', [103.530, 116.280, 123.675],
            'TEST.IGNORE_BOX_REGRESSION', False,
            'TEST.OUTPUT_FEATURE', True,
        ]
        cfg.merge_from_list(vinvl_x152c4_config)
        cfg.freeze()
        self.cfg = cfg
        self.vinvl = AttrRCNN(cfg, device=device)
        self.attr_score_threshold = attr_score_threshold

    def get_default_pretrained_state_dict_from_provider(self) -> Dict[str, torch.Tensor]:
        """
        Get pretrained model from https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/vinvl_vg_x152c4.pth
        """
        pretrained_dict = torch.hub.load_state_dict_from_url(STATE_DICT_URL)
        cleaned_pretrained_dict = {k.replace('module.', 'vinvl.'): v for k, v in pretrained_dict.items()}
        return torch_apply_state_to_partial_model(self, cleaned_pretrained_dict)

    def forward(self, x, sizes) -> BBoxOutputArrayFormat:
        predictions = self.vinvl(x)
        boxes, classes, scores, features = [], [], [], []
        attr_labels, attr_scores = [], []
        for i, prediction in enumerate(predictions):
            prediction = prediction.to(torch.device("cpu"))
            prediction = prediction.resize(sizes[i])
            boxes.append(prediction.bbox)
            classes.append(prediction.get_field("labels"))
            scores.append(prediction.get_field("scores"))
            features.append(prediction.get_field("box_features"))
            attr_scores.append(prediction.get_field("attr_scores"))
            attr_labels.append(prediction.get_field("attr_labels"))

        return BBoxOutputArrayFormat(
            bounding_boxes=boxes,
            probabilities=scores,
            features=features,
            labels=classes,
            attributes=attr_labels,
            attr_scores=attr_scores,
        )

    def bulk_inference(
            self,
            data: MLModuleDatasetProtocol[_IndexType, ImageDatasetType],
            **options
    ) -> Optional[Tuple[List[_IndexType], List[BBoxCollection]]]:
        """Runs inference on all images in a ImageFilesDatasets

        :param data_loader_options:
        :param data: A dataset returning tuples of item_index, PIL.Image
        :return:
        """
        # Default batch size
        data_loader_options = options.pop("data_loader_options", {})
        data_loader_options.setdefault('batch_size', 256)
        data_loader_options.setdefault('collate_fn', BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY))
        return super().bulk_inference(
            data,
            data_loader_options=data_loader_options,
            **options
        )

    def get_dataset_transforms(self) -> List[Callable]:
        return [
            Resize(self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MAX_SIZE_TEST),
            ToTensor(),
            Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD, self.cfg.INPUT.TO_BGR255)
        ]

    def get_labels(self):
        return VinVLLabels()

    def get_attribute_labels(self):
        return VinVLAttributeLabels()
