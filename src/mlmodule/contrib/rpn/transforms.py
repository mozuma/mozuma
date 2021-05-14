from typing import Tuple

import numpy as np
import torch
from PIL.Image import Image
from torchvision.transforms import functional as f

from mlmodule.box import BBoxOutput
from mlmodule.torch.data.images import TORCHVISION_STANDARD_IMAGE_TRANSFORMS


class RegionCrop(object):
    """Region Extraction preprocessing"""

    def __call__(self, data: Tuple[Image, BBoxOutput]) -> Image:
        """ Crops a region from an image """
        # NOTE: I am 99% sure the crop isn't done in place and copies the image first
        image, box = data

        x0, y0 = int(box.bounding_box[0].x), int(box.bounding_box[0].y)
        x1, y1 = int(box.bounding_box[1].x), int(box.bounding_box[1].y)
        left, top = min(x0, x1), min(y0, y1)
        w, h = np.abs(x1 - x0), np.abs(y1 - y0)

        return f.crop(image, top, left, h, w)


class StandardTorchvisionRegionTransforms(object):
    """ Region Extraction preprocessing for all regions in an image """

    def __call__(self, data: Image) -> torch.Tensor:
        """ Applies the standard torchvision transforms to each image in a list """
        img = data
        for t in TORCHVISION_STANDARD_IMAGE_TRANSFORMS:
            img = t(img)

        return img
