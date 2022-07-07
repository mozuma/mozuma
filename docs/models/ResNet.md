# ResNet

PyTorch implementation of ResNet[@resnet] as defined in [Torchvision](https://pytorch.org/vision/stable/models.html).

## Pre-trained models

{% for model in models.resnet -%}
::: mozuma.models.resnet.{{ model.factory }}
    rendering:
        show_signature: False
{% endfor %}

## Base model

The ResNet model is an implementation of a [`TorchMlModule`][mozuma.torch.modules.TorchMlModule].

::: mozuma.models.resnet.TorchResNetModule
    selection:
        members: none

## Pre-trained state origins

See the [stores documentation](../references/stores.md) for usage.

::: mozuma.models.resnet.stores.ResNetTorchVisionStore
    selection:
        members: none
