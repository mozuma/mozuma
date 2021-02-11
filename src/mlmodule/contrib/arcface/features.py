import torch

from mlmodule.contrib.arcface.base import BaseArcFaceModule
from mlmodule.torch.data.images import transforms


class ArcFaceFeatures(BaseArcFaceModule):

    def __init__(self, device=None):
        super().__init__(device=device)

    @classmethod
    def results_handler(cls, acc_results, new_indices, new_output: torch.Tensor):
        """Runs after the forward pass at inference

        :param acc_results: Holds a tuple with indices, list of FacesFeatures namedtuple
        :param new_indices: New indices for the current batch
        :param new_output: New inference output for the current batch
        :return:
        """
        file_names, face_indices = new_indices
        new_indices = list(zip(file_names, face_indices.tolist()))
        return super().results_handler(acc_results, new_indices, new_output)

    def get_dataset_transforms(self):
        return [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
