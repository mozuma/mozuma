# CLIP

See [OpenAI/CLIP](https://github.com/openai/CLIP) for the source code and original models.

This model has extra requirements:

```bash
pip install git+ssh://git@github.com/LSIR/mlmodule.git#egg=mlmodule[torch,clip]
# or
pip install mlmodule[torch,clip]
```

## Models

CLIP comes with image ([`CLIPImageModule`][mlmodule.contrib.clip.image.CLIPImageModule])
and a text ([`CLIPTextModule`][mlmodule.contrib.clip.text.CLIPTextModule]) encoders.
These modules are an implementation of [`TorchMlModule`][mlmodule.v2.torch.modules.TorchMlModule].

::: mlmodule.contrib.clip.image.CLIPImageModule
    selection:
        members: none

::: mlmodule.contrib.clip.text.CLIPTextModule
    selection:
        members: none

## Pre-trained states from CLIP

See the [stores documentation](../references/stores.md) for usage.

::: mlmodule.contrib.clip.stores.CLIPStore
    selection:
        members: none

## List models and parameters

There is a command line utility to list all available models from CLIP with their associated parameters in JSON format:

```bash
python -m mlmodule.contrib.clip.list
```

The output is used to fill the mlmodule.contrib.clip.parameters file.
