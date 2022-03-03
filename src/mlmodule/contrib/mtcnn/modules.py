from typing import List, Sequence, Tuple, Union, cast

import numpy as np
import torch

from mlmodule.contrib.mtcnn._mtcnn import MLModuleMTCNN
from mlmodule.v2.base.predictions import (
    BatchBoundingBoxesPrediction,
    BatchModelPrediction,
)
from mlmodule.v2.states import StateType
from mlmodule.v2.torch.modules import TorchMlModule


def pil_to_array(img):
    return np.array(img, dtype=np.uint8)


class TorchMTCNNModule(TorchMlModule[Sequence[torch.Tensor], np.ndarray]):
    """MTCNN face detection module

    Attributes:
        thresholds (Tuple[float, float, float]): MTCNN threshold hyperparameters
        image_size (Tuple[int, int]): Image size after pre-preprocessing
        min_face_size (int): Minimum face size in pixels
        device (torch.device): Torch device to initialise the model weights
    """

    def __init__(
        self,
        thresholds: Tuple[float, float, float] = (0.6, 0.7, 0.7),
        image_size: Tuple[int, int] = (720, 720),
        min_face_size: int = 20,
        device=None,
    ):
        super().__init__(device=device)
        self.image_size = image_size
        self.mtcnn = MLModuleMTCNN(
            thresholds=thresholds,
            device=self.device,
            min_face_size=min_face_size,
            pretrained=False,
        )

    @property
    def state_type(self) -> StateType:
        return StateType(backend="pytorch", architecture="facenet-mtcnn")

    def _array_or(self, arr: Union[np.ndarray, None], other: np.ndarray) -> np.ndarray:
        if arr is None:
            return other
        return arr

    def forward_single(
        self, image: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Runs MTCNN face detection on a single image

        Arguments:
            image (torch.Tensor): A single image with channel dimension last `(height, width, channels)`

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                Returns three values in a tuple:

                - bounding boxes stacked in a `np.ndarray, dtype=float64, shape=(n_boxes, 4,)`
                - bounding boxes probabilities stacked in a `np.ndarray, dtype=float64 ,shape=(n_boxes,)`
                - landmarks stacked in a `np.ndarray, dtype=float64`
        """
        # Getting height and width
        height, width, _channels = image.shape
        if height < self.mtcnn.min_face_size or width < self.mtcnn.min_face_size:
            # Image is too small to detect images
            return (
                np.empty((0, 4), dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )

        raw_probabilities: Union[np.ndarray, None]
        raw_boxes: Union[np.ndarray, None]
        raw_landmarks: Union[np.ndarray, None]
        raw_boxes, raw_probabilities, raw_landmarks = self.mtcnn.detect(
            image, landmarks=True
        )

        # Replacing None values
        boxes = self._array_or(raw_boxes, np.empty((0, 4), dtype=np.float64))
        landmarks = self._array_or(raw_landmarks, np.empty(0, dtype=np.float64))
        probabilities = self._array_or(raw_probabilities, np.empty(0, dtype=np.float64))

        return boxes, cast(np.ndarray, probabilities), landmarks

    def forward(
        self, batch: Sequence[torch.Tensor]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Runs MTCNN face detection

        Arguments:
            batch (Sequence[torch.Tensor]): A sequence of images

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
                Returns three values in a tuple:

                - List of bounding boxes stacked in a `np.ndarray, shape=(n_boxes, 4,)`
                - List of bounding boxes probabilities stacked in a `np.ndarray, shape=(n_boxes,)`
                - List of landmarks stacked in a `np.ndarray`
        """
        results = [self.forward_single(img) for img in batch]

        return_value: Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]] = (
            [],
            [],
            [],
        )
        for boxes, probabilities, landmarks in results:
            return_value[0].append(boxes)
            return_value[1].append(probabilities)
            return_value[2].append(landmarks)
        return return_value

    def forward_predictions(
        self, batch: Sequence[torch.Tensor]
    ) -> BatchModelPrediction[np.ndarray]:
        """Runs MTCNN face detection

        Arguments:
            batch (Sequence[torch.Tensor]): A sequence of images

        Returns:
            BatchModelPrediction[np.ndarray]: A batch prediction with the attribute `bounding_boxes`
        """
        return BatchModelPrediction(
            bounding_boxes=[
                BatchBoundingBoxesPrediction(
                    bounding_boxes=boxes, scores=probabilities, features=landmarks
                )
                for boxes, probabilities, landmarks in zip(*self.forward(batch))
            ]
        )

    def get_dataset_transforms(self):
        return [pil_to_array]
