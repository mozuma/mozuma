# Video key-frames extractor

This model implements two types of modules: a video frames encoder and the key-frames module.
These models are an implementation of a [`TorchModel`][mozuma.torch.modules.TorchModel].

## Pre-trained models

{% for model in models.keyframes -%}
::: mozuma.models.keyframes.{{ model.factory }}
    rendering:
        show_signature: False
{% endfor %}


## Base key-frames selector model

These models allow to extract key-frames from a video.

::: mozuma.models.keyframes.selectors.KeyFrameSelector
    selection:
        members: none

## Base video frames encoder model

::: mozuma.models.keyframes.encoders.VideoFramesEncoder
    selection:
        members: none
