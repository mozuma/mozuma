# MTCNN

We are using `facenet-pytorch` to load pre-trained MTCNN model[@mtcnn], see <https://github.com/timesler/facenet-pytorch>.

## Requirements

This needs mlmodule with the `mtcnn` and `torch` extra requirements:

```bash
pip install git+ssh://git@github.com/LSIR/mlmodule.git#egg=mlmodule[torch,mtcnn]
# or
pip install mlmodule[torch,mtcnn]
```

## Model

The MTCNN model is an implementation of a [`TorchMlModule`][mlmodule.v2.torch.modules.TorchMlModule].

::: mlmodule.models.mtcnn.modules.TorchMTCNNModule
    selection:
        members: none

## Provider store

See the [stores documentation](../references/stores.md) for usage.

::: mlmodule.models.mtcnn.stores.FaceNetMTCNNStore
    selection:
        members: none
