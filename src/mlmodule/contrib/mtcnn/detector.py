from typing import Dict, Tuple, List, Union, TypeVar, Any

import torch
import numpy as np
from PIL.Image import Image

from mlmodule.contrib.mtcnn.mtcnn import MLModuleMTCNN
from mlmodule.box import BBoxOutput, BBoxPoint, BBoxCollection
from mlmodule.torch import BaseTorchMLModule
from mlmodule.torch.data.base import IndexedDataset
from mlmodule.torch.mixins import TorchPretrainedModuleMixin, DownloadPretrainedStateFromProvider
from mlmodule.torch.utils import torch_apply_state_to_partial_model
from mlmodule.torch.data.images import transforms


InputDatasetType = TypeVar('InputDatasetType', bound=IndexedDataset[Any, Any, Union[Image, np.ndarray]])


class MTCNNDetector(BaseTorchMLModule[InputDatasetType],
                    TorchPretrainedModuleMixin, DownloadPretrainedStateFromProvider):
    """Face detection module"""

    state_dict_key = "pretrained-models/face-detection/mtcnn.pt"

    def __init__(self, thresholds=None, image_size=720, min_face_size=20, device=None):
        super().__init__(device=device)
        thresholds = thresholds or [0.6, 0.7, 0.7]
        self.image_size = image_size
        self.mtcnn = MLModuleMTCNN(thresholds=thresholds, device=device, min_face_size=min_face_size, pretrained=False)

    def get_default_pretrained_state_dict_from_provider(self) -> Dict[str, torch.Tensor]:
        pretrained_mtcnn = MLModuleMTCNN(pretrained=True)
        pretrained_dict = {
            f'mtcnn.{key}': value for key, value in pretrained_mtcnn.state_dict().items()
            if key.startswith('onet') or key.startswith('pnet') or key.startswith('rnet')
        }
        return torch_apply_state_to_partial_model(self, pretrained_dict)

    def forward(self, x):
        return self.mtcnn.detect(x, landmarks=True)

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
        # Aspect ratios of each image
        aspect_ratios = {idx: [x/self.image_size for x in img.size]
                         for idx, img in data}
        return super().bulk_inference(
            data, data_loader_options=data_loader_options,
            result_handler_options={'aspect_ratios': aspect_ratios},
            **opts
        )

    def get_dataset_transforms(self):
        return [
            transforms.Resize((self.image_size, self.image_size)),
            np.uint8
        ]

    @classmethod
    def results_handler(
            cls, acc_results: Tuple[List, List[BBoxCollection]],
            new_indices: List,
            new_output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            aspect_ratios: Dict = None
    ) -> Tuple[List, List[BBoxCollection]]:
        """Runs after the forward pass at inference

        :param acc_results: Holds a tuple with indices, list of FacesFeatures namedtuple
        :param new_indices: New indices for the current batch
        :param new_output: New inference output for the current batch
        :param aspect_ratios: Used to rescale coordinated of bounding boxes and landmarks
        :return:
        """
        if aspect_ratios is None:
            raise ValueError('aspect_ratios parameter cannot be None')

        # Dealing for the first call where acc_results is None
        output: List[BBoxCollection]
        indices, output = acc_results or ([], [])

        # Converting to list
        new_indices = cls.tensor_to_python_list_safe(new_indices)
        indices += new_indices

        for ind, (boxes, probs, landmarks) in zip(new_indices, zip(*new_output)):
            # Iterating through output for each image
            img_bbox = []
            # Rescaling
            boxes = boxes*(aspect_ratios[ind]*2)
            landmarks = landmarks*aspect_ratios[ind]
            for b, p, l in zip(boxes, probs, landmarks):
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
