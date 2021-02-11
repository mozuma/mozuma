import numpy as np
from mlmodule.contrib.arcface.utils import warp_and_crop_face, get_reference_facial_points


class ArcFaceAlignment(object):

    def __init__(
        self,
        ref_pts=get_reference_facial_points(default_square=True),
        face_size=112
    ):
        self.ref_pts = ref_pts
        self.face_size = face_size

    def __call__(self, image, landmarks):
        return warp_and_crop_face(
            np.array(image), landmarks, self.ref_pts, crop_size=(self.face_size, self.face_size))
