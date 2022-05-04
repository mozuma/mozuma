import dataclasses
import random
from typing import Tuple

from torchvision.transforms import functional as F


@dataclasses.dataclass
class Resize:
    """
    :param min_size: minimun size (height, width).
    :param max_size: maximum size (height, width).
    """

    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size) -> Tuple[int, int]:
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image):
        original_size = image.size
        size = self.get_size(original_size)
        return (F.resize(image, size), original_size)


@dataclasses.dataclass
class ToTensor:
    def __call__(self, inputs):
        image, sizes = inputs
        return (F.to_tensor(image), sizes)


@dataclasses.dataclass
class Normalize:
    def __init__(self, mean, std, to_bgr255):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, inputs):
        image, sizes = inputs
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return (image, sizes)
