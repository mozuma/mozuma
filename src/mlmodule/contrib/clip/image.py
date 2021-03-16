__all__ = (
    'CLIPImageEncoder',
    'CLIPResNet50ImageEncoder',
    'CLIPResNet101ImageEncoder',
    'CLIPResNet50x4ImageEncoder',
    'CLIPViTB32ImageEncoder',
)

from typing import Optional

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from mlmodule.contrib.clip.base import BaseCLIPModule


def get_image_transform(src_pixel_size):
    return Compose([
        Resize(src_pixel_size, interpolation=Image.BICUBIC),
        CenterCrop(src_pixel_size),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class CLIPImageEncoder(BaseCLIPModule):

    def __init__(self, device: Optional[torch.device] = None):
        super().__init__(device=device)

        clip_module = self._get_clip_module()

        # Populating image encoder attributes
        self.visual = clip_module.visual

    def forward(self, images):
        return self.visual(images)

    def get_dataset_transforms(self):
        return [get_image_transform(self.visual.input_resolution)]


class CLIPResNet50ImageEncoder(CLIPImageEncoder):
    clip_model_name = "RN50"


class CLIPResNet101ImageEncoder(CLIPImageEncoder):
    clip_model_name = "RN101"


class CLIPResNet50x4ImageEncoder(CLIPImageEncoder):
    clip_model_name = "RN50x4"


class CLIPViTB32ImageEncoder(CLIPImageEncoder):
    clip_model_name = "ViT-B/32"
