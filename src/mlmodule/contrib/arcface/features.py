import torch
import numpy as np

from mlmodule.contrib.arcface.base import BaseArcFaceModule
from mlmodule.contrib.arcface.utils import warp_and_crop_face, get_reference_facial_points
from mlmodule.torch.data.faces import FaceDataset, FacesFeatures, FaceBatch
from mlmodule.torch.data.images import transforms


def collate_wrapper(batch):
    indices, faces = zip(*batch)
    cropped_faces, face_descriptors = zip(*faces)
    return indices, FaceBatch(cropped_faces, face_descriptors)


class FaceAlignment(object):

    def __init__(self, ref_pts):
        self.ref_pts = ref_pts
        self.tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __call__(self, sample):
        image, faces = sample
        return torch.stack([
            self.tfm(warp_and_crop_face(
                np.array(image), src_pts, self.ref_pts, crop_size=(112, 112)))
            for src_pts in faces.landmarks
        ]), faces


class ArcFaceFeatures(BaseArcFaceModule):

    __result_struct__ = FacesFeatures

    def __init__(self, device=None):
        super().__init__(device=device)
        self.ref_pts = get_reference_facial_points(default_square=True)

    def __call__(self, x: FaceBatch):
        features = self.forward(x.cropped_faces, x.num_faces)
        # update face descriptors with features
        return [FacesFeatures(*faces._replace(features=features[i].cpu())) for i, faces in enumerate(x.descriptors)]

    def forward(self, faces, num_faces):
        features = super().forward(faces)
        return torch.split(features, num_faces, dim=0)

    def get_data_loader(self, data, **data_loader_options):
        return super().get_data_loader(data, collate_fn=collate_wrapper, **data_loader_options)

    @classmethod
    def results_handler(cls, acc_results, new_indices, new_output):
        """Runs after the forward pass at inference

            :param acc_results: Holds a tuple with indices, list of FacesFeatures namedtuple
            :param new_indices: New indices for the current batch
            :param new_output: New inference output for the current batch
            :return:
            """
        # Dealing for the first call where acc_results is None
        indices, output = acc_results or ([], [])

        # Appending new indices
        indices += cls.tensor_to_python_list_safe(new_indices)

        # Appending new output
        output += new_output

        return indices, output

    def get_dataset_transforms(self):
        return [FaceAlignment(self.ref_pts)]
