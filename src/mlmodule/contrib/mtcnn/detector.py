from collections import namedtuple

import torch
from torch.hub import load_state_dict_from_url
from facenet_pytorch import MTCNN

from mlmodule.torch import BaseTorchMLModule
from mlmodule.torch.mixins import TorchPretrainedModuleMixin
from mlmodule.torch.utils import torch_apply_state_to_partial_model
from mlmodule.torch.data.images import ImageDataset

MTCNN_WEIGHTS_URL = ''


class MTCNNDetector(BaseTorchMLModule, TorchPretrainedModuleMixin):

    def __init__(self, thresholds=[0.6, 0.7, 0.7], device=torch.device('cpu')):
        super().__init__()
        self.mtcnn = MTCNN(thresholds=thresholds, device=device)

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

    def bulk_inference(self, data: ImageDataset):
        """Runs inference on all images in a ImageFilesDatasets

        :param data: A dataset returning tuples of item_index, PIL.Image
        :return:
        """
        # Setting model in eval mode
        self.mtcnn.eval()

        # Disabling gradient computation
        results = []
        FacesDetected = namedtuple(
            'FacesDetected', ['boxes', 'probas', 'landmarks'])
        with torch.no_grad():
            for i, img in data:
                boxes, probas, landmarks = self.mtcnn.detect(
                    img, landmarks=True)
                results.append(
                    FacesDetected(boxes, probas, landmarks))
        return results
