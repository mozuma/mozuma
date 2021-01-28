from collections import namedtuple

import torch
from torch.hub import load_state_dict_from_url
import numpy as np
from facenet_pytorch import MTCNN

from mlmodule.torch import BaseTorchMLModule
from mlmodule.torch.mixins import TorchPretrainedModuleMixin
from mlmodule.torch.utils import torch_apply_state_to_partial_model
from mlmodule.torch.data.images import ImageDataset, transforms

MTCNN_WEIGHTS_URL = ''
FacesDetected = namedtuple('FacesDetected', ['boxes', 'probas', 'landmarks'])


class MTCNNDetector(BaseTorchMLModule, TorchPretrainedModuleMixin):

    __result_struct__ = FacesDetected

    def __init__(self, thresholds=None, image_size=720, min_face_size=20, device=None):
        super().__init__(device=device)
        thresholds = thresholds or [0.6, 0.7, 0.7]
        self.image_size = image_size
        self.mtcnn = MTCNN(thresholds=thresholds,
                           device=device, min_face_size=min_face_size)

    def get_default_pretrained_state_dict(self, map_location=None, cache_dir=None, **options):
        """Returns the state dict for a pretrained resnet model

        :param map_location:
        :param cache_dir:
        :param options:
        :return:
        """
        if MTCNN_WEIGHTS_URL:  # TODO: add state dict to S3
            # Downloading state dictionary
            pretrained_state_dict = load_state_dict_from_url(
                MTCNN_WEIGHTS_URL, model_dir=cache_dir, map_location=map_location, **options
            )
        else:
            pretrained_state_dict = self.mtcnn.state_dict()
        # Removing deleted layers from state dict and updating the other with pretrained data
        return torch_apply_state_to_partial_model(self, pretrained_state_dict)

    def forward(self, x):
        return self.mtcnn.detect(x, landmarks=True)

    @staticmethod
    def rescale_coordinates(indices, results, aspect_ratios):
        rescaled_results = []
        for i, (boxes, probs, landmarks) in zip(indices, results):
            rescaled_results.append(
                FacesDetected(
                    boxes*(aspect_ratios[i]*2), probs, landmarks*aspect_ratios[i])
            )
        return indices, rescaled_results

    def bulk_inference(self, data: ImageDataset, batch_size=256, **data_loader_options):
        """Runs inference on all images in a ImageFilesDatasets

        :param data: A dataset returning tuples of item_index, PIL.Image
        :param batch_size:
        :param data_loader_options:
        :return:
        """
        aspect_ratios = {idx: [x/self.image_size for x in img.size]
                         for idx, img in data}
        indices, results = super().bulk_inference(data, batch_size=batch_size, **data_loader_options)
        return self.rescale_coordinates(indices, results, aspect_ratios)

    def get_dataset_transforms(self):
        return [
            transforms.Resize((self.image_size, self.image_size)),
            np.uint8,
            lambda x: torch.as_tensor(x.copy())
        ]

    @classmethod
    def results_handler(cls, acc_results, new_indices, new_output: torch.Tensor):
        """Runs after the forward pass at inference

        :param acc_results: Holds a tuple with indices, list of FacesDetected namedtuple
        :param new_indices: New indices for the current batch
        :param new_output: New inference output for the current batch
        :return:
        """
        # Dealing for the first call where acc_results is None
        indices, output = acc_results or ([], [])

        # Appending new indices
        indices += cls.tensor_to_python_list_safe(new_indices)

        # Appending new output
        new_boxes_list, new_probas_list, new_landmarks_list = new_output
        output += [
            FacesDetected(boxes, probas, landmarks)
            for boxes, probas, landmarks in zip(new_boxes_list, new_probas_list, new_landmarks_list)
        ]

        return indices, output
