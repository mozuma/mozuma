# ArcFace

Implementation of ArcFace[@deng2018arcface]
in PyTorch by [InsightFace](https://github.com/TreB1eN/InsightFace_Pytorch).


## Pre-trained models

{% for model in models.arcface -%}
::: mozuma.models.arcface.{{ model.factory }}
    rendering:
        show_signature: False
{% endfor %}


## Base models

The MagFace model is an implementation of a [`TorchModel`][mozuma.torch.modules.TorchModel].

::: mozuma.models.arcface.modules.TorchArcFaceModule
    selection:
        members: none

## Provider store

See the [stores documentation](../references/stores.md) for usage.

::: mozuma.models.arcface.stores.ArcFaceStore
    selection:
        members: none
