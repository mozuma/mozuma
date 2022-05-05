# Callbacks

Callbacks are used to control to control how a [runner](runners.md)
handles the results (features, labels or model weights).

They are classes implementing a pre-defined set of functions:

- [`save_features`][mlmodule.callbacks.base.BaseSaveFeaturesCallback.save_features]
- [`save_label_scores`][mlmodule.callbacks.base.BaseSaveLabelsCallback.save_label_scores]
- [`save_bounding_boxes`][mlmodule.callbacks.base.BaseSaveBoundingBoxCallback.save_bounding_boxes]
- [`save_frames`][mlmodule.callbacks.base.BaseSaveVideoFramesCallback.save_frames]

## In memory callbacks

These callbacks accumulate results in-memory.
They expose their results via object attributes.

::: mlmodule.callbacks.CollectFeaturesInMemory

::: mlmodule.callbacks.CollectLabelsInMemory

::: mlmodule.callbacks.CollectBoundingBoxesInMemory

::: mlmodule.callbacks.CollectVideoFramesInMemory

## Callbacks for training

::: mlmodule.callbacks.states.SaveModelState

## Write your own callbacks

::: mlmodule.callbacks.base.BaseSaveFeaturesCallback

::: mlmodule.callbacks.base.BaseSaveLabelsCallback

::: mlmodule.callbacks.base.BaseSaveBoundingBoxCallback

::: mlmodule.callbacks.base.BaseSaveVideoFramesCallback

::: mlmodule.callbacks.base.BaseRunnerEndCallback
