# Callbacks

Callbacks are used to control to control how a [runner](runners.md)
handles the results (features, labels or model weights).

They are classes implementing a pre-defined set of functions:

- [`save_features`][mozuma.callbacks.base.BaseSaveFeaturesCallback.save_features]
- [`save_label_scores`][mozuma.callbacks.base.BaseSaveLabelsCallback.save_label_scores]
- [`save_bounding_boxes`][mozuma.callbacks.base.BaseSaveBoundingBoxCallback.save_bounding_boxes]
- [`save_frames`][mozuma.callbacks.base.BaseSaveVideoFramesCallback.save_frames]

## In memory callbacks

These callbacks accumulate results in-memory.
They expose their results via object attributes.

::: mozuma.callbacks.CollectFeaturesInMemory

::: mozuma.callbacks.CollectLabelsInMemory

::: mozuma.callbacks.CollectBoundingBoxesInMemory

::: mozuma.callbacks.CollectVideoFramesInMemory

## Callbacks for training

::: mozuma.callbacks.states.SaveModelState

## Write your own callbacks

::: mozuma.callbacks.base.BaseSaveFeaturesCallback

::: mozuma.callbacks.base.BaseSaveLabelsCallback

::: mozuma.callbacks.base.BaseSaveBoundingBoxCallback

::: mozuma.callbacks.base.BaseSaveVideoFramesCallback

::: mozuma.callbacks.base.BaseRunnerEndCallback
