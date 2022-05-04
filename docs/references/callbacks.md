# Callbacks

Callbacks are used to control to control how a [runner](runners.md)
handles the results (features, labels or model weights).

They are classes implementing a pre-defined set of functions:

- [`save_features`][mlmodule.v2.base.callbacks.BaseSaveFeaturesCallback.save_features]
- [`save_label_scores`][mlmodule.v2.base.callbacks.BaseSaveLabelsCallback.save_label_scores]
- [`save_bounding_boxes`][mlmodule.v2.base.callbacks.BaseSaveBoundingBoxCallback.save_bounding_boxes]
- [`save_frames`][mlmodule.v2.base.callbacks.BaseSaveVideoFramesCallback.save_frames]

## In memory callbacks

These callbacks accumulate results in-memory.
They expose their results via object attributes.

::: mlmodule.v2.helpers.callbacks.CollectFeaturesInMemory

::: mlmodule.v2.helpers.callbacks.CollectLabelsInMemory

::: mlmodule.v2.helpers.callbacks.CollectBoundingBoxesInMemory

::: mlmodule.v2.helpers.callbacks.CollectVideoFramesInMemory

## Callbacks for training

::: mlmodule.v2.helpers.callbacks.SaveModelState

## Write your own callbacks

::: mlmodule.v2.base.callbacks.BaseSaveFeaturesCallback

::: mlmodule.v2.base.callbacks.BaseSaveLabelsCallback

::: mlmodule.v2.base.callbacks.BaseSaveBoundingBoxCallback

::: mlmodule.v2.base.callbacks.BaseSaveVideoFramesCallback

::: mlmodule.v2.base.callbacks.BaseRunnerEndCallback
