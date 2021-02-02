from collections import namedtuple
from typing import List

import torch
import torchvision.transforms.functional as TF

from mlmodule.torch.data.base import IndexedDataset
from mlmodule.torch.data.images import get_pil_image_from_file, convert_to_rgb

FacesFeatures = namedtuple(
    'FacesFeatures', ['boxes', 'probas', 'landmarks', 'features'])


class ImageFromFile(object):

    def __call__(self, sample):
        image, faces = sample
        return get_pil_image_from_file(image), faces


class ToRGB(object):

    def __call__(self, sample):
        image, faces = sample
        return convert_to_rgb(image), faces


class FaceDataset(IndexedDataset):

    def __init__(self, image_path, face_detected, to_rgb=True):
        super().__init__(image_path, list(zip(image_path, face_detected)))
        self.add_transforms([ImageFromFile()])
        if to_rgb:
            self.add_transforms([ToRGB()])


class FaceBatch(object):

    def __init__(self, cropped_faces: List[torch.tensor], face_descriptors: List[FacesFeatures]):
        self.cropped_faces = torch.cat(cropped_faces)
        self.num_faces = [x.size(0) for x in cropped_faces]
        self.descriptors = face_descriptors

    def to(self, device: torch.device):
        self.cropped_faces = self.cropped_faces.to(device)
        return self

    def cpu(self):
        self.cropped_faces.cpu()

    def __len__(self):
        return len(self.num_faces)

    def size(self):
        return self.cropped_faces.size()
