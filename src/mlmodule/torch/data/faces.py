from typing import Callable
import numpy as np

from mlmodule.torch.data.base import IndexedDataset
from mlmodule.torch.data.images import get_pil_image_from_file, convert_to_rgb


class FaceDataset(IndexedDataset):

    def __init__(self, image_path, face_detected, crop_fn: Callable[[str], np.array], to_rgb=True):
        indices, items = self.get_cropped_faces(
            image_path, face_detected, crop_fn, to_rgb)
        super().__init__(indices, items)

    @ classmethod
    def get_cropped_faces(cls, image_path, face_detected, crop_fn, to_rgb=True):
        cropped_faces = []
        indices = []
        for path, faces in list(zip(image_path, face_detected)):
            img = get_pil_image_from_file(path)
            if to_rgb:
                img = convert_to_rgb(img)
            for i, ldk in enumerate(faces.landmarks):
                cropped_faces.append(crop_fn(img, ldk))
                indices.append((path, i))
        return indices, cropped_faces
