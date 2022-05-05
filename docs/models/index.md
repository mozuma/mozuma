# Models

!!! tip
    See the the menu on the left for a list of available models

In MLModule, a model is usually implemented as a class.
The model implementation details primarily depend on the type
of [runner](../references/runners.md) used.
For instance, the [`TorchInferenceRunner`][mlmodule.torch.runners.TorchInferenceRunner]
expects to receive a subclass of [`TorchMlModule`][mlmodule.torch.modules.TorchMlModule].

However, there are a few conventions to follow:

- Model predictions should implement the
  [BatchModelPrediction][mlmodule.predictions.BatchModelPrediction] class, this is required for
  [callbacks](../references/callbacks.md) to work properly.
- If the model's state needs to be saved,
  the model should follow the [`ModelWithState`][mlmodule.models.ModelWithState] protocol.
- If the model outputs labels,
  the model should follow the [`ModelWithLabels`][mlmodule.models.ModelWithLabels] protocol.

## Predictions

!!! note
    The `ArrayLike` type is expected to be a `np.ndarray` or a `torch.Tensor`.

::: mlmodule.predictions.BatchModelPrediction
::: mlmodule.predictions.BatchBoundingBoxesPrediction
::: mlmodule.predictions.BatchVideoFramesPrediction


## State management

A model with internal state (weights) should implement the
[`ModelWithState`][mlmodule.models.ModelWithState] protocol
to be compatible with [state stores](../references/stores.md).

::: mlmodule.models.ModelWithState

## Labels

When a model returns label scores, it must define a
[`LabelSet`][mlmodule.labels.base.LabelSet].
This should be defined by implementing the
[`ModelWithLabels`][mlmodule.models.ModelWithLabels]
protocol.

::: mlmodule.models.ModelWithLabels

## PyTorch models

PyTorch models should be a subclass of `TorchMlModule`.

!!! note
    PyTorch models already implement the
    [`ModelWithState`][mlmodule.models.ModelWithState] protocol
    by default.

::: mlmodule.torch.modules.TorchMlModule
    selection:
        members:
            - state_type
            - forward
            - get_dataloader_collate_fn
            - get_dataset_transforms
            - to_predictions
