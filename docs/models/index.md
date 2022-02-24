# Models

!!! tip
    See the the menu on the left for a list of available models

The sections below will help with defining a new model.
Each section contains the functions a MLModule model class should define
to implement a feature.


## Models state management

A model with internal state (weights) should at least implement the
[`ModelWithState`][mlmodule.v2.base.models.ModelWithState] protocol.

::: mlmodule.v2.base.models.ModelWithState

## Labels

When a model returns label scores, it must define a
[`LabelSet`][mlmodule.labels.base.LabelSet].
This should be defined by implementing the
[`ModelWithLabels`][mlmodule.v2.base.models.ModelWithLabels]
protocol.

::: mlmodule.v2.base.models.ModelWithLabels

## PyTorch models

PyTorch models should be a subclass of `TorchMlModule`.

!!! note
    PyTorch models already implement the
    [`ModelWithState`][mlmodule.v2.base.models.ModelWithState] protocol
    by default.

::: mlmodule.v2.torch.modules.TorchMlModule
