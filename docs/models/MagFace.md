# MagFace

We are using the official implementation of MagFace[@Meng_2021_CVPR] in Pytorch. See [https://github.com/IrvingMeng/MagFace](https://github.com/IrvingMeng/MagFace).


## Pre-trained models

{% for model in models.magface -%}
::: mlmodule.models.magface.{{ model.factory }}
    rendering:
        show_signature: False
{% endfor %}


## Base models

The MagFace model is an implementation of a [`TorchMlModule`][mlmodule.torch.modules.TorchMlModule].

::: mlmodule.models.magface.modules.TorchMagFaceModule
    selection:
        members: none

## Provider store

See the [stores documentation](../references/stores.md) for usage.

::: mlmodule.models.magface.stores.MagFaceStore
    selection:
        members: none
