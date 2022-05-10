# ArcFace

Implementation of ArcFace[@deng2018arcface]
in PyTorch by [InsightFace](https://github.com/TreB1eN/InsightFace_Pytorch).


## Pre-trained models

{% for model in models.arcface -%}
::: mlmodule.models.arcface.{{ model.factory }}
    rendering:
        show_signature: False
{% endfor %}


## Base models

The MagFace model is an implementation of a [`TorchMlModule`][mlmodule.torch.modules.TorchMlModule].

::: mlmodule.models.arcface.modules.TorchArcFaceModule
    selection:
        members: none

## Provider store

See the [stores documentation](../references/stores.md) for usage.

::: mlmodule.models.arcface.stores.ArcFaceStore
    selection:
        members: none
