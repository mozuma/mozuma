from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

from mozuma.helpers.images import convert_to_rgb


def get_image_transform(src_pixel_size: int):
    """Image transforms for CLIP Image encoder"""
    return Compose(
        [
            Resize(src_pixel_size, interpolation=Image.BICUBIC),
            CenterCrop(src_pixel_size),
            convert_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
