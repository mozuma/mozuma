from typing import Dict, Iterable, List, Optional, Tuple, TypeVar, Union, cast

import numpy as np
import torch

from mlmodule.box import BBoxCollection, BBoxOutputArrayFormat
from mlmodule.contrib.mtcnn.mtcnn import MLModuleMTCNN
from mlmodule.torch.base import TorchMLModuleBBox
from mlmodule.torch.data.base import MLModuleDatasetProtocol
from mlmodule.torch.mixins import DownloadPretrainedStateFromProvider
from mlmodule.torch.utils import torch_apply_state_to_partial_model
from mlmodule.types import ImageDatasetType

_IndexType = TypeVar("_IndexType", contravariant=True)


def pil_to_array(img):
    return np.array(img, dtype=np.uint8)


class MTCNNDetectorOriginal(
    TorchMLModuleBBox[_IndexType, ImageDatasetType],
    DownloadPretrainedStateFromProvider,
):
    """Face detection module without support for batch"""

    state_dict_key = "pretrained-models/face-detection/mtcnn.pt"

    def __init__(
        self,
        thresholds=None,
        min_face_size=20,
        device=None,
    ):
        super().__init__(device=device)
        thresholds = thresholds or [0.6, 0.7, 0.7]
        self.mtcnn = MLModuleMTCNN(
            thresholds=thresholds,
            device=self.device,
            min_face_size=min_face_size,
            pretrained=False,
        )

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

    def forward(self, x: torch.Tensor) -> BBoxOutputArrayFormat:
        prob: Iterable[Union[np.ndarray, List[None]]]
        boxes: List[Union[np.ndarray, None]]
        landmarks: List[Union[np.ndarray, None]]

        # Getting height and width
        h, w = x.shape[1:3]
        if h < self.mtcnn.min_face_size or w < self.mtcnn.min_face_size:
            # Image is too small to detect images
            return BBoxOutputArrayFormat(
                bounding_boxes=[np.array([])],
                probabilities=[np.array([])],
                features=[np.array([])],
            )

        boxes, prob, landmarks = self.mtcnn.detect(x, landmarks=True)
        # Applying aspect ratios
        boxes_clean: List[np.ndarray] = [
            b if b is not None else np.array([]) for b in boxes
        ]
        landmarks_clean: List[np.ndarray] = [
            land if land is not None else np.array([]) for land in landmarks
        ]
        prob_clean: List[np.ndarray] = [
            cast(np.ndarray, p) if None not in p else np.array([]) for p in prob
        ]
        return BBoxOutputArrayFormat(
            bounding_boxes=boxes_clean,
            probabilities=prob_clean,
            features=landmarks_clean,
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
        data_loader_options["batch_size"] = 1
        return super().bulk_inference(
            data, data_loader_options=data_loader_options, **options
        )

    def get_dataset_transforms(self):
        return [pil_to_array]
