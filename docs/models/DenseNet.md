# DenseNet

PyTorch implementation of DenseNet architecture[@Huang_2017_CVPR] as defined in [Torchvision](https://pytorch.org/vision/stable/models.html).

## Pre-trained models

{% for model in models.densenet -%}
::: mozuma.models.densenet.{{ model.factory }}
    rendering:
        show_signature: False
{% endfor %}


## Base model

The DenseNet model is an implementation of a [`TorchModel`][mozuma.torch.modules.TorchModel].

::: mozuma.models.densenet.TorchDenseNetModule
    selection:
        members: none

## Pre-trained state origins

See the [stores documentation](../references/stores.md) for usage.

::: mozuma.models.densenet.stores.DenseNetTorchVisionStore
    selection:
        members: none

::: mozuma.models.densenet.stores.DenseNetPlaces365Store
    selection:
        members: none
