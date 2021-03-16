
## Download CLIP models on PC32

```bash
export PUBLIC_ASSETS=/mnt/storage01/lsir-public-assets/pretrained-models
# For text encoders
python -m mlmodule.cli download clip.CLIPResNet50TextEncoder "$PUBLIC_ASSETS/text-encoder/clip-rn50-text.pt"
python -m mlmodule.cli download clip.CLIPResNet101TextEncoder "$PUBLIC_ASSETS/text-encoder/clip-rn101-text.pt"
python -m mlmodule.cli download clip.CLIPResNet50x4TextEncoder "$PUBLIC_ASSETS/text-encoder/clip-rn50x4-text.pt"
python -m mlmodule.cli download clip.CLIPViTB32TextEncoder "$PUBLIC_ASSETS/text-encoder/clip-vit-b32-text.pt"
```


## List models and parameters

There is a command line utility to list all available models from CLIP with their associated parameters in JSON format:

```bash
python -m mlmodule.contrib.clip.list
```

The output is used to fill the mlmodule.contrib.clip.parameters file.
