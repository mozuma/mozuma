# CLIP

CLIP[@clip_radford21a] model encodes text snippets and images into a common embedding space which enables zero-shot retrieval and prediction.
See [OpenAI/CLIP](https://github.com/openai/CLIP) for the source code and original models.


## Pre-trained models

{% for model in models.clip -%}
::: mozuma.models.clip.{{ model.factory }}
    rendering:
        show_signature: False
{% endfor %}


## Models

CLIP comes with image ([`CLIPImageModule`][mozuma.models.clip.image.CLIPImageModule])
and a text ([`CLIPTextModule`][mozuma.models.clip.text.CLIPTextModule]) encoders.
These modules are an implementation of [`TorchMlModule`][mozuma.torch.modules.TorchMlModule].

::: mozuma.models.clip.image.CLIPImageModule
    selection:
        members: none

::: mozuma.models.clip.text.CLIPTextModule
    selection:
        members: none

## Pre-trained states from CLIP

See the [stores documentation](../references/stores.md) for usage.

::: mozuma.models.clip.stores.CLIPStore
    selection:
        members: none

## List models and parameters

There is a command line utility to list all available models from CLIP with their associated parameters in JSON format:

```bash
python -m mozuma.models.clip.list
```

The output is used to fill the mozuma.models.clip.parameters file.
