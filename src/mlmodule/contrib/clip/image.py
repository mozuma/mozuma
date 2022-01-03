"""CLIP image encoders"""
__all__ = (
    "CLIPImageEncoder",
    "CLIPResNet50ImageEncoder",
    "CLIPResNet101ImageEncoder",
    "CLIPResNet50x4ImageEncoder",
    "CLIPViTB32ImageEncoder",
)

from typing import Callable, List, Optional, TypeVar, Union

import torch
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from typing_extensions import Literal

from mlmodule.contrib.clip.base import BaseCLIPModule
from mlmodule.torch.data.images import convert_to_rgb
from mlmodule.types import ImageDatasetType

_IndexType = TypeVar("_IndexType", covariant=True)


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


class CLIPImageEncoder(BaseCLIPModule[_IndexType, ImageDatasetType]):
    """Image encoder of the CLIP model"""

    model_type: Union[Literal["image"], Literal["text"]] = "image"

    def __init__(self, device: Optional[torch.device] = None):
        super().__init__(device=device)

        clip_module = self._get_clip_module()

        # Populating image encoder attributes
        self.visual = clip_module.visual

        self.convert_weights()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Apply the image encoder"""
        return self.visual(images.type(self._dtype))

    def get_dataset_transforms(self) -> List[Callable]:
        """Dataset transform to resize and preprocess images"""
        return [get_image_transform(self.visual.input_resolution)]


class CLIPResNet50ImageEncoder(CLIPImageEncoder):
    """CLIP Image encoder - ResNet50"""

    clip_model_name = "RN50"


class CLIPResNet101ImageEncoder(CLIPImageEncoder):
    """CLIP Image encoder - ResNet101"""

    clip_model_name = "RN101"


class CLIPResNet50x4ImageEncoder(CLIPImageEncoder):
    """CLIP Image encoder - ResNet50x4"""

    clip_model_name = "RN50x4"


class CLIPViTB32ImageEncoder(CLIPImageEncoder):
    """CLIP Image encoder - ViT-B/32"""

    clip_model_name = "ViT-B/32"
