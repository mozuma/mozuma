# Models

## Pre-trained models

| Reference    | Name   | License |
|---------------|---------------|-----------|
{% for module in models -%}
{% for model in models[module] -%}
| [`{{ model.factory }}`][mozuma.models.{{ module }}.{{ model.factory }}] | {{ model.name }} | [{{ licenses[model.license].name }} :octicons-link-external-16:]({{ licenses[model.license].link }}){:target="_blank"} |
{% endfor -%}
{% endfor %}


## Add a new model

In MoZuMa, a model is usually implemented as a class.
The model implementation details primarily depend on the type
of [runner](../references/runners.md) used.
For instance, the [`TorchInferenceRunner`][mozuma.torch.runners.TorchInferenceRunner]
expects to receive a subclass of [`TorchModel`][mozuma.torch.modules.TorchModel].

However, there are a few conventions to follow:

- Model predictions should implement the
  [BatchModelPrediction][mozuma.predictions.BatchModelPrediction] class, this is required for
  [callbacks](../references/callbacks.md) to work properly.
- If the model's state needs to be saved,
  the model should follow the [`ModelWithState`][mozuma.models.ModelWithState] protocol.
- If the model predicts labels,
  it should follow the [`ModelWithLabels`][mozuma.models.ModelWithLabels] protocol.

## Predictions

!!! note
    The `ArrayLike` type is expected to be a `np.ndarray` or a `torch.Tensor`.

::: mozuma.predictions.BatchModelPrediction
::: mozuma.predictions.BatchBoundingBoxesPrediction
::: mozuma.predictions.BatchVideoFramesPrediction


## State management

A model with internal state (weights) should implement the
[`ModelWithState`][mozuma.models.ModelWithState] protocol
to be compatible with [state stores](../references/stores.md).

::: mozuma.models.ModelWithState

## Labels

When a model returns label scores, it must define a
[`LabelSet`][mozuma.labels.base.LabelSet].
This should be defined by implementing the
[`ModelWithLabels`][mozuma.models.ModelWithLabels]
protocol.

::: mozuma.models.ModelWithLabels

## PyTorch models

PyTorch models should be a subclass of `TorchModel`.

!!! note
    PyTorch models already implement the
    [`ModelWithState`][mozuma.models.ModelWithState] protocol
    by default.

::: mozuma.torch.modules.TorchModel
    selection:
        members:
            - state_type
            - forward
            - get_dataloader_collate_fn
            - get_dataset_transforms
            - to_predictions
