
# Runners

Runners are used to execute inference or training of a model.
They are initialised with a `model`, a `dataset`, `callbacks` and `options`.
They are executed with the `run` function which takes no arguments.

## Inference

::: mlmodule.torch.runners.TorchInferenceRunner
    selection:
      members:
        - run

::: mlmodule.torch.runners.TorchInferenceMultiGPURunner
    selection:
      members:
        - run

## Training

::: mlmodule.torch.runners.TorchTrainingRunner
    selection:
      members:
        - run

## Write your own runner

::: mlmodule.runners.BaseRunner
