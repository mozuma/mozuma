# ResNet

PyTorch implementation of ResNet[@resnet] as defined in [Torchvision](https://pytorch.org/vision/stable/models.html).

## Pre-trained models

{% for model in models.resnet -%}
::: mlmodule.models.resnet.{{ model.factory }}
    rendering:
        show_signature: False
{% endfor %}

## Base model

The ResNet model is an implementation of a [`TorchMlModule`][mlmodule.torch.modules.TorchMlModule].

::: mlmodule.models.resnet.TorchResNetModule
    selection:
        members: none

## Pre-trained state origins

See the [stores documentation](../references/stores.md) for usage.

::: mlmodule.models.resnet.stores.ResNetTorchVisionStore
    selection:
        members: none
