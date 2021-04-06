from typing import Tuple

import numpy as np
from PIL.Image import Image

from mlmodule.contrib.arcface.utils import warp_and_crop_face, get_reference_facial_points
from mlmodule.box import BBoxOutput


class ArcFaceAlignment(object):
    """Faces crop preprocessing"""

    def __init__(
        self,
        ref_pts=get_reference_facial_points(default_square=True),
        face_size=112
    ):
        self.ref_pts = ref_pts
        self.face_size = face_size

    def __call__(self, data: Tuple[Image, BBoxOutput]) -> np.ndarray:
        """Crops and realigns the detected faces"""
        image, bbox = data
        return warp_and_crop_face(
            np.array(image), bbox.features, self.ref_pts, crop_size=(self.face_size, self.face_size))
