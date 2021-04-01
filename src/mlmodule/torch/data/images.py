from pathlib import Path
from typing import List, TypeVar, Union, IO

from PIL import Image
from torchvision import transforms

from mlmodule.torch.data.base import IndexedDataset

TORCHVISION_STANDARD_IMAGE_TRANSFORMS = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]


def get_pil_image_from_file(file: Union[str, Path, IO]) -> Image.Image:
    return Image.open(file)


def convert_to_rgb(pil_image: Image.Image) -> Image.Image:
    return pil_image.convert('RGB')


TI = TypeVar('TI')


class BaseImageDataset(IndexedDataset[TI, Image.Image]):

    def __init__(self, indices: List[TI], image_path: List[str], to_rgb=True):
        super().__init__(indices, image_path)
        self.add_transforms([get_pil_image_from_file])
        if to_rgb:
            self.add_transforms([convert_to_rgb])


class ImageDataset(BaseImageDataset[str]):

    def __init__(self, image_path, to_rgb=True):
        super().__init__(image_path, image_path, to_rgb=to_rgb)
