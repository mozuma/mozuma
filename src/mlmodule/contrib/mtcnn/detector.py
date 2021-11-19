import dataclasses
from typing import Dict, Iterable, Optional, Tuple, List, TypeVar, Union, cast

import torch
import numpy as np

from mlmodule.contrib.mtcnn.mtcnn import MLModuleMTCNN
from mlmodule.box import BBoxCollection, BBoxOutputArrayFormat
from mlmodule.torch.base import TorchMLModuleBBox
from mlmodule.torch.data.base import MLModuleDatasetProtocol
from mlmodule.torch.mixins import (
    DownloadPretrainedStateFromProvider,
    ResizableImageInputMixin,
)
from mlmodule.torch.utils import torch_apply_state_to_partial_model
from mlmodule.torch.data.images import transforms
from mlmodule.types import ImageDatasetType


_IndexType = TypeVar("_IndexType", contravariant=True)


@dataclasses.dataclass
class ResizeWithAspectRatios:
    """
    :param img_size: Desired output size (height, width).
    """

    img_size: Tuple[int, int]

    def __call__(self, img) -> Tuple[np.ndarray, np.ndarray]:
        height, width = self.img_size
        return (
            np.uint8(transforms.Resize(self.img_size)(img)),
            np.array([x / target for x, target in zip(img.size, (width, height))]),
        )


class MTCNNDetector(
    TorchMLModuleBBox[_IndexType, ImageDatasetType],
    DownloadPretrainedStateFromProvider,
    ResizableImageInputMixin,
):
    """Face detection module"""

    state_dict_key = "pretrained-models/face-detection/mtcnn.pt"

    def __init__(
        self,
        thresholds=None,
        image_size: Tuple[int, int] = (720, 720),
        min_face_size=20,
        device=None,
    ):
        super().__init__(device=device)
        thresholds = thresholds or [0.6, 0.7, 0.7]
        self.image_size = image_size
        self.mtcnn = MLModuleMTCNN(
            thresholds=thresholds,
            device=self.device,
            min_face_size=min_face_size,
            pretrained=False,
        )

    def shrink_input_image_size(self) -> Tuple[int, int]:
        return self.image_size

    def get_default_pretrained_state_dict_from_provider(
        self,
    ) -> Dict[str, torch.Tensor]:
        pretrained_mtcnn = MLModuleMTCNN(pretrained=True)
        pretrained_dict = {
            f"mtcnn.{key}": value
            for key, value in pretrained_mtcnn.state_dict().items()
            if key.startswith("onet")
            or key.startswith("pnet")
            or key.startswith("rnet")
        }
        return torch_apply_state_to_partial_model(self, pretrained_dict)

    def forward(self, x, aspect_ratios) -> BBoxOutputArrayFormat:
        prob: Iterable[Union[np.ndarray, List[None]]]
        boxes: List[Union[np.ndarray, None]]
        landmarks: List[Union[np.ndarray, None]]
        boxes, prob, landmarks = self.mtcnn.detect(x, landmarks=True)
        # Applying aspect ratios
        aspect_ratios = aspect_ratios.numpy()
        boxes_clean: List[np.ndarray] = [
            b * np.hstack((a, a)) if b is not None else np.array([])
            for b, a in zip(boxes, aspect_ratios)
        ]
        landmarks_clean: List[np.ndarray] = [
            land * a if land is not None else np.array([])
            for land, a in zip(landmarks, aspect_ratios)
        ]
        prob_clean: List[np.ndarray] = [cast(np.ndarray, p) if None not in p else np.array([]) for p in prob]
        return BBoxOutputArrayFormat(
            bounding_boxes=boxes_clean, probabilities=prob_clean, features=landmarks_clean
        )

    def bulk_inference(
        self, data: MLModuleDatasetProtocol[_IndexType, ImageDatasetType], **options
    ) -> Optional[Tuple[List[_IndexType], List[BBoxCollection]]]:
        """Runs inference on all images in a ImageFilesDatasets

        :param data_loader_options:
        :param data: A dataset returning tuples of item_index, PIL.Image
        :return:
        """
        # Default batch size
        data_loader_options = options.pop("data_loader_options", {})
        data_loader_options.setdefault("batch_size", 256)
        return super().bulk_inference(
            data, data_loader_options=data_loader_options, **options
        )

    def get_dataset_transforms(self):
        return [ResizeWithAspectRatios(self.image_size)]
