from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch

from mozuma.labels.vinvl import VINVL_LABELS
from mozuma.labels.vinvl_attributes import VINVL_ATTRIBUTE_LABELS
from mozuma.models.vinvl.collator import BatchCollator
from mozuma.models.vinvl.config import sg_cfg
from mozuma.models.vinvl.models import AttrRCNN
from mozuma.models.vinvl.models.config import cfg
from mozuma.models.vinvl.models.structures.bounding_box import BoxList
from mozuma.models.vinvl.transforms import Normalize, Resize, ToTensor
from mozuma.predictions import BatchBoundingBoxesPrediction, BatchModelPrediction
from mozuma.states import StateType
from mozuma.torch.modules import TorchModel

_ForwardOutput = Tuple[
    Sequence[torch.Tensor],  # boxes
    Sequence[torch.Tensor],  # scores
    Sequence[torch.Tensor],  # features
    Sequence[torch.Tensor],  # classes
    Sequence[torch.Tensor],  # attr_labels
    Sequence[torch.Tensor],  # attr_scores
]

STATE_DICT_URL = "https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/vinvl_vg_x152c4.pth"


class TorchVinVLDetectorModule(
    TorchModel[Tuple[torch.Tensor, Sequence[Tuple[int, int]]], _ForwardOutput]
):
    """[VinVL](https://github.com/pzzhang/VinVL) object detection model

    Attributes:
        score_threshold (float):
        attr_score_threshold (float):
        device (torch.device): PyTorch device attribute to initialise model.
    """

    def __init__(
        self,
        score_threshold: float = 0.5,
        attr_score_threshold: float = 0.5,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(device=device, is_trainable=False)
        cfg.set_new_allowed(True)
        cfg.merge_from_other_cfg(sg_cfg)
        cfg.set_new_allowed(False)
        vinvl_x152c4_config = [
            "MODEL.META_ARCHITECTURE",
            "AttrRCNN",
            "MODEL.BACKBONE.CONV_BODY",
            "R-152-C4",
            "MODEL.RESNETS.BACKBONE_OUT_CHANNELS",
            1024,
            "MODEL.RESNETS.STRIDE_IN_1X1",
            False,
            "MODEL.RESNETS.NUM_GROUPS",
            32,
            "MODEL.RESNETS.WIDTH_PER_GROUP",
            8,
            "MODEL.RPN.PRE_NMS_TOP_N_TEST",
            6000,
            "MODEL.RPN.POST_NMS_TOP_N_TEST",
            300,
            "MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE",
            384,
            "MODEL.ROI_HEADS.POSITIVE_FRACTION",
            0.5,
            "MODEL.ROI_HEADS.SCORE_THRESH",
            0.2,
            "MODEL.ROI_HEADS.DETECTIONS_PER_IMG",
            100,
            "MODEL.ROI_HEADS.MIN_DETECTIONS_PER_IMG",
            10,
            "MODEL.ROI_HEADS.NMS_FILTER",
            1,
            "MODEL.ROI_HEADS.SCORE_THRESH",
            score_threshold,
            "MODEL.ROI_BOX_HEAD.NUM_CLASSES",
            1595,
            "MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES",
            525,
            "MODEL.ROI_ATTRIBUTE_HEAD.POSTPROCESS_ATTRIBUTES_THRESHOLD",
            0.05,
            "MODEL.ATTRIBUTE_ON",
            True,
            "INPUT.MIN_SIZE_TEST",
            600,
            "INPUT.MAX_SIZE_TEST",
            1000,
            "INPUT.PIXEL_MEAN",
            [103.530, 116.280, 123.675],
            "TEST.IGNORE_BOX_REGRESSION",
            False,
            "TEST.OUTPUT_FEATURE",
            True,
        ]
        cfg.merge_from_list(vinvl_x152c4_config)
        cfg.freeze()
        self.cfg = cfg
        self.vinvl = AttrRCNN(cfg, device=device)
        self.attr_score_threshold = attr_score_threshold

    @property
    def state_type(self) -> StateType:
        return StateType(backend="pytorch", architecture="vinvl-vg-x152c4")

    def forward(
        self, batch: Tuple[torch.Tensor, Sequence[Tuple[int, int]]]
    ) -> _ForwardOutput:
        """VinVL forward pass

        Arguments:
            batch (tuple[torch.Tensor, Sequence[tuple[int, int]]]):
                Tuple of image data and image resize tuple.

        Returns:
            tuple[Sequence[torch.Tensor] * 6]: A tuple of 6 sequence of torch.Tensor. Containing:

                - boxes
                - scores
                - features
                - classes
                - attr_labels
                - attr_scores
        """
        x, sizes = batch

        predictions: List[BoxList] = self.vinvl(x)
        boxes, classes, scores, features = [], [], [], []
        attr_labels, attr_scores = [], []
        for i, prediction in enumerate(predictions):
            prediction = prediction.to(torch.device("cpu"))
            prediction = prediction.resize(sizes[i])
            boxes.append(prediction.bbox)
            classes.append(prediction.get_field("labels").unsqueeze(1))
            scores.append(prediction.get_field("scores"))
            features.append(prediction.get_field("box_features"))
            attr_scores.append(prediction.get_field("attr_scores"))
            attr_labels.append(prediction.get_field("attr_labels"))

        return (
            boxes,
            scores,
            features,
            classes,
            attr_labels,
            attr_scores,
        )

    def to_predictions(
        self, forward_output: _ForwardOutput
    ) -> BatchModelPrediction[torch.Tensor]:
        # For now we ignore the classes and attributes
        boxes, scores, features, _, _, _ = forward_output
        return BatchModelPrediction(
            bounding_boxes=[
                BatchBoundingBoxesPrediction(bounding_boxes=b, scores=s, features=f)
                for b, s, f in zip(boxes, scores, features)
            ]
        )

    def get_dataset_transforms(self) -> List[Callable]:
        return [
            Resize(self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MAX_SIZE_TEST),
            ToTensor(),
            Normalize(
                cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD, self.cfg.INPUT.TO_BGR255
            ),
        ]

    def get_dataloader_collate_fn(self) -> Optional[Callable[[Any], Any]]:
        return BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)

    def get_labels(self):
        return VINVL_LABELS

    def get_attribute_labels(self):
        return VINVL_ATTRIBUTE_LABELS
