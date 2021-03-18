
## Requirements

This needs mlmodule with the clip and torch extra requirements:

```bash
pip install git+ssh://git@github.com/LSIR/mlmodule.git#egg=mlmodule[torch,clip]
# or
pip install mlmodule[torch,clip]
```

## Download CLIP models on PC32

```bash
export PUBLIC_ASSETS=/mnt/storage01/lsir-public-assets/pretrained-models
# For text encoders
python -m mlmodule.cli download clip.CLIPResNet50TextEncoder "$PUBLIC_ASSETS/text-encoder/clip-rn50-text.pt"
python -m mlmodule.cli download clip.CLIPResNet101TextEncoder "$PUBLIC_ASSETS/text-encoder/clip-rn101-text.pt"
python -m mlmodule.cli download clip.CLIPResNet50x4TextEncoder "$PUBLIC_ASSETS/text-encoder/clip-rn50x4-text.pt"
python -m mlmodule.cli download clip.CLIPViTB32TextEncoder "$PUBLIC_ASSETS/text-encoder/clip-vit-b32-text.pt"
# For image encoders
python -m mlmodule.cli download clip.CLIPResNet50ImageEncoder "$PUBLIC_ASSETS/image-encoder/clip-rn50-image.pt"
python -m mlmodule.cli download clip.CLIPResNet101ImageEncoder "$PUBLIC_ASSETS/image-encoder/clip-rn101-image.pt"
python -m mlmodule.cli download clip.CLIPResNet50x4ImageEncoder "$PUBLIC_ASSETS/image-encoder/clip-rn50x4-image.pt"
python -m mlmodule.cli download clip.CLIPViTB32ImageEncoder "$PUBLIC_ASSETS/image-encoder/clip-vit-b32-image.pt"
```


## List models and parameters

There is a command line utility to list all available models from CLIP with their associated parameters in JSON format:

```bash
python -m mlmodule.contrib.clip.list
```

The output is used to fill the mlmodule.contrib.clip.parameters file.
