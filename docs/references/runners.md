
# Runners

Runners are used to execute inference or training of a model.
They are initialised with a `model`, a `dataset`, `callbacks` and `options`.
They are executed with the `run` function which takes no arguments.

## Inference

::: mlmodule.v2.torch.runners.TorchInferenceRunner
    selection:
      members:
        - run

::: mlmodule.v2.torch.runners.TorchInferenceMultiGPURunner
    selection:
      members:
        - run

## Training

::: mlmodule.v2.torch.runners.TorchTrainingRunner
    selection:
      members:
        - run

## Write your own runner

::: mlmodule.v2.base.runners.BaseRunner
