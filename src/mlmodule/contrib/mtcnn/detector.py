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

    def __init__(self, thresholds=[0.6, 0.7, 0.7], image_size=720, min_face_size=20, device=torch.device('cpu')):
        super().__init__()
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
    def rescale_coordinates(results, aspect_ratios):
        rescaled_results = []
        for i, (boxes, probs, landmarks) in enumerate(results):
            rescaled_results.append(
                FacesDetected(
                    boxes*(aspect_ratios[i]*2), probs, landmarks*aspect_ratios[i])
            )
        return rescaled_results

    def bulk_inference(self, data: ImageDataset, batch_size=256, **data_loader_options):
        """Runs inference on all images in a ImageFilesDatasets

        :param data: A dataset returning tuples of item_index, PIL.Image
        :param batch_size:
        :param data_loader_options:
        :return:
        """
        aspect_ratios = [[x/self.image_size for x in img.size]
                         for _, img in data]
        results = super().bulk_inference(data, batch_size=256, **data_loader_options)
        return self.rescale_coordinates(results, aspect_ratios)

    def get_dataset_transforms(self):
        return [
            transforms.Resize((self.image_size, self.image_size)),
            np.uint8,
            lambda x: torch.as_tensor(x.copy())
        ]

    @classmethod
    def get_results_handler(cls):
        """Runs after the forward pass at inference

        :return:
        """
        return lambda indices, outputs: zip(indices, [(outputs[0][i], outputs[1][i], outputs[2][i]) for i in range(len(outputs[0]))])
