from typing import Tuple

import numpy as np
from PIL.Image import Image

from mozuma.models.arcface.utils import get_reference_facial_points, warp_and_crop_face
from mozuma.predictions import BatchBoundingBoxesPrediction


class ArcFaceAlignment(object):
    """Faces crop preprocessing"""

    def __init__(
        self, ref_pts=get_reference_facial_points(default_square=True), face_size=112
    ):
        self.ref_pts = ref_pts
        self.face_size = face_size

    def __call__(
        self, data: Tuple[Image, BatchBoundingBoxesPrediction[np.ndarray]]
    ) -> np.ndarray:
        """Crops and realigns the detected faces"""
        image, bbox = data
        if bbox.features is None:
            raise ValueError(
                "Expected bounding box information to have attribute features != None"
            )
        return warp_and_crop_face(
            np.array(image),
            bbox.features[0],
            self.ref_pts,
            crop_size=(self.face_size, self.face_size),
        )
