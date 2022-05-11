# MTCNN

We are using `facenet-pytorch` to load pre-trained MTCNN model[@mtcnn], see <https://github.com/timesler/facenet-pytorch>.

## Pre-trained models

{% for model in models.mtcnn -%}
::: mlmodule.models.mtcnn.{{ model.factory }}
    rendering:
        show_signature: False
{% endfor %}


## Base model

The MTCNN model is an implementation of a [`TorchMlModule`][mlmodule.torch.modules.TorchMlModule].

::: mlmodule.models.mtcnn.modules.TorchMTCNNModule
    selection:
        members: none

## Provider store

See the [stores documentation](../references/stores.md) for usage.

::: mlmodule.models.mtcnn.stores.FaceNetMTCNNStore
    selection:
        members: none
