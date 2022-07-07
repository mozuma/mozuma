# MTCNN

We are using `facenet-pytorch` to load pre-trained MTCNN model[@mtcnn], see <https://github.com/timesler/facenet-pytorch>.

## Pre-trained models

{% for model in models.mtcnn -%}
::: mozuma.models.mtcnn.{{ model.factory }}
    rendering:
        show_signature: False
{% endfor %}


## Base model

The MTCNN model is an implementation of a [`TorchMlModule`][mozuma.torch.modules.TorchMlModule].

::: mozuma.models.mtcnn.modules.TorchMTCNNModule
    selection:
        members: none

## Provider store

See the [stores documentation](../references/stores.md) for usage.

::: mozuma.models.mtcnn.stores.FaceNetMTCNNStore
    selection:
        members: none
