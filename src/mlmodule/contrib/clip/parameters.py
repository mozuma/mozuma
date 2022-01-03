__all__ = ("PARAMETERS",)

from collections import OrderedDict
from typing import Any, Dict
from typing import OrderedDict as OrderedDictType

PARAMETERS: Dict[str, OrderedDictType[str, Any]] = {
    "RN50": OrderedDict(
        embed_dim=1024,
        # Vision
        image_resolution=224,
        vision_layers=[3, 4, 6, 3],
        vision_width=64,
        vision_patch_size=None,
        # Text
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
    ),
    "RN101": OrderedDict(
        embed_dim=512,
        # Vision
        image_resolution=224,
        vision_layers=[3, 4, 23, 3],
        vision_width=64,
        vision_patch_size=None,
        # Text
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
    ),
    "RN50x4": OrderedDict(
        embed_dim=640,
        # Vision
        image_resolution=288,
        vision_layers=[4, 6, 10, 6],
        vision_width=80,
        vision_patch_size=None,
        # Text
        context_length=77,
        vocab_size=49408,
        transformer_width=640,
        transformer_heads=10,
        transformer_layers=12,
    ),
    "ViT-B/32": OrderedDict(
        embed_dim=512,
        # Vision
        image_resolution=224,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=32,
        # Text
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
    ),
}
