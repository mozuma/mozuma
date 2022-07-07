from typing import List, Sequence, Tuple, Union

import numpy as np
import torch

from mozuma.models.mtcnn._mtcnn import MoZuMaMTCNN
from mozuma.predictions import BatchBoundingBoxesPrediction, BatchModelPrediction
from mozuma.states import StateType
from mozuma.torch.modules import TorchModel

_SingleBoundingBoxTupleForm = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
_MultiBoundingBoxTupleForm = Tuple[
    List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]
]


def pil_to_array(img):
    return np.array(img, dtype=np.uint8)


def _array_or(arr: Union[np.ndarray, None], other: np.ndarray) -> np.ndarray:
    """If the `arr` argument is None, returns `other`"""
    if arr is None:
        return other
    return arr


class TorchMTCNNModule(TorchModel[Sequence[torch.Tensor], _MultiBoundingBoxTupleForm]):
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
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(device=device, is_trainable=False)
        self.image_size = image_size
        self.mtcnn = MoZuMaMTCNN(
            thresholds=thresholds,
            device=self.device,
            min_face_size=min_face_size,
            pretrained=False,
        )

    def to(self, *args, **kwargs):
        device = self._extract_device_from_args(*args, **kwargs)
        self.mtcnn.device = device
        return super().to(*args, **kwargs)

    @property
    def state_type(self) -> StateType:
        return StateType(backend="pytorch", architecture="facenet-mtcnn")

    def forward_single(self, image: torch.Tensor) -> _SingleBoundingBoxTupleForm:
        """Runs MTCNN face detection on a single image

        Arguments:
            image (torch.Tensor): A single image with channel dimension last `(height, width, channels)`

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                Returns three values in a tuple:

                - bounding boxes stacked in a `torch.Tensor, dtype=float64, shape=(n_boxes, 4,)`
                - bounding boxes probabilities stacked in a `torch.Tensor, dtype=float64 ,shape=(n_boxes,)`
                - landmarks stacked in a `torch.Tensor, dtype=float64`
        """
        # Getting height and width
        height, width, _channels = image.shape
        if height < self.mtcnn.min_face_size or width < self.mtcnn.min_face_size:
            # Image is too small to detect images
            return (
                torch.empty((0, 4), dtype=torch.float64),
                torch.empty(0, dtype=torch.float64),
                torch.empty(0, dtype=torch.float64),
            )

        raw_probabilities: Union[np.ndarray, None]
        raw_boxes: Union[np.ndarray, None]
        raw_landmarks: Union[np.ndarray, None]
        raw_boxes, raw_probabilities, raw_landmarks = self.mtcnn.detect(
            image, landmarks=True
        )

        # Replacing None values
        boxes = _array_or(raw_boxes, np.empty((0, 4), dtype=np.float64))
        landmarks = _array_or(raw_landmarks, np.empty(0, dtype=np.float64))
        probabilities = _array_or(raw_probabilities, np.empty(0, dtype=np.float64))

        return torch.Tensor(boxes), torch.Tensor(probabilities), torch.Tensor(landmarks)

    def forward(self, batch: Sequence[torch.Tensor]) -> _MultiBoundingBoxTupleForm:
        """Runs MTCNN face detection

        Arguments:
            batch (Sequence[torch.Tensor]): A sequence of images

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
                Returns three values in a tuple:

                - List of bounding boxes stacked in a `torch.Tensor, shape=(n_boxes, 4,)`
                - List of bounding boxes probabilities stacked in a `torch.Tensor, shape=(n_boxes,)`
                - List of landmarks stacked in a `torch.Tensor`
        """
        results = [self.forward_single(img) for img in batch]

        return_value: Tuple[
            List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]
        ] = (
            [],
            [],
            [],
        )
        for boxes, probabilities, landmarks in results:
            return_value[0].append(boxes)
            return_value[1].append(probabilities)
            return_value[2].append(landmarks)
        return return_value

    def to_predictions(
        self, forward_output: _MultiBoundingBoxTupleForm
    ) -> BatchModelPrediction[torch.Tensor]:
        """Transforms MTCNN face detection output

        Arguments:
            forward_output (tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]):
                A tuple containing the bounding box, probabilities and landmarks.

        Returns:
            BatchModelPrediction[np.ndarray]: A batch prediction with the attribute `bounding_boxes`
        """
        return BatchModelPrediction(
            bounding_boxes=[
                BatchBoundingBoxesPrediction(
                    bounding_boxes=boxes, scores=probabilities, features=landmarks
                )
                for boxes, probabilities, landmarks in zip(*forward_output)
            ]
        )

    def get_dataset_transforms(self):
        return [pil_to_array]
