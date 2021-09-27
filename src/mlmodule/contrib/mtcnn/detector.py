import dataclasses
from typing import Dict, Tuple, List, Union, TypeVar, Any

import torch
import numpy as np
from PIL.Image import Image

from mlmodule.contrib.mtcnn.mtcnn import MLModuleMTCNN
from mlmodule.box import BBoxOutput, BBoxPoint, BBoxCollection
from mlmodule.torch import BaseTorchMLModule
from mlmodule.torch.data.base import IndexedDataset
from mlmodule.torch.mixins import DownloadPretrainedStateFromProvider, \
    ResizableImageInputMixin
from mlmodule.torch.utils import torch_apply_state_to_partial_model
from mlmodule.torch.data.images import transforms


InputDatasetType = TypeVar('InputDatasetType', bound=IndexedDataset[Any, Any, Union[Image, np.ndarray]])


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
            np.array([x/target for x, target in zip(img.size, (width, height))])
        )


class MTCNNDetector(BaseTorchMLModule[InputDatasetType],
                    DownloadPretrainedStateFromProvider,
                    ResizableImageInputMixin):
    """Face detection module"""

    state_dict_key = "pretrained-models/face-detection/mtcnn.pt"

    def __init__(self, thresholds=None, image_size: Tuple[int, int] = (720, 720), min_face_size=20, device=None):
        super().__init__(device=device)
        thresholds = thresholds or [0.6, 0.7, 0.7]
        self.image_size = image_size
        self.mtcnn = MLModuleMTCNN(
            thresholds=thresholds, device=self.device, min_face_size=min_face_size, pretrained=False
        )

    def shrink_input_image_size(self) -> Tuple[int, int]:
        return self.image_size

    def get_default_pretrained_state_dict_from_provider(self) -> Dict[str, torch.Tensor]:
        pretrained_mtcnn = MLModuleMTCNN(pretrained=True)
        pretrained_dict = {
            f'mtcnn.{key}': value for key, value in pretrained_mtcnn.state_dict().items()
            if key.startswith('onet') or key.startswith('pnet') or key.startswith('rnet')
        }
        return torch_apply_state_to_partial_model(self, pretrained_dict)

    def forward(self, x, aspect_ratios):
        boxes, prob, landmarks = self.mtcnn.detect(x, landmarks=True)
        # Applying aspect ratios
        aspect_ratios = aspect_ratios.numpy()
        boxes = [b*np.hstack((a, a)) if b is not None else b for b, a in zip(boxes, aspect_ratios)]
        landmarks = [land*a if land is not None else land for land, a in zip(landmarks, aspect_ratios)]
        return boxes, prob, landmarks

    def bulk_inference(
            self, data: InputDatasetType, data_loader_options=None, **opts
    ) -> Tuple[List, List[BBoxCollection]]:
        """Runs inference on all images in a ImageFilesDatasets

        :param data_loader_options:
        :param data: A dataset returning tuples of item_index, PIL.Image
        :return:
        """
        # Default batch size
        data_loader_options = data_loader_options or {}
        data_loader_options.setdefault('batch_size', 256)
        return super().bulk_inference(
            data, data_loader_options=data_loader_options,
            **opts
        )

    def get_dataset_transforms(self):
        return [
            ResizeWithAspectRatios(self.image_size)
        ]

    @classmethod
    def results_handler(
            cls, acc_results: Tuple[List, List[BBoxCollection]],
            new_indices: List,
            new_output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[List, List[BBoxCollection]]:
        """Runs after the forward pass at inference

        :param acc_results: Holds a tuple with indices, list of FacesFeatures namedtuple
        :param new_indices: New indices for the current batch
        :param new_output: New inference output for the current batch
        :return:
        """

        # Dealing for the first call where acc_results is None
        output: List[BBoxCollection]
        indices, output = acc_results or ([], [])

        # Converting to list
        new_indices = cls.tensor_to_python_list_safe(new_indices)
        indices += new_indices

        for ind, (boxes, probs, landmarks) in zip(new_indices, zip(*new_output)):
            # Iterating through output for each image
            img_bbox = []

            if boxes is not None:
                # Rescaling
                for b, p, l in zip(boxes.tolist(), probs.tolist(), landmarks):
                    # Iterating through each bounding box
                    if b is not None:
                        # We have detected a face
                        img_bbox.append(BBoxOutput(
                            bounding_box=(BBoxPoint(*b[:2]), BBoxPoint(*b[2:])),  # Extracting two points
                            probability=p,
                            features=l
                        ))
            output.append(img_bbox)

        return indices, output
