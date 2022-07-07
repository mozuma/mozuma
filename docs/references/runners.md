
# Runners

Runners are used to execute inference or training of a model.
They are initialised with a `model`, a `dataset`, `callbacks` and `options`.
They are executed with the `run` function which takes no arguments.

## Inference

::: mozuma.torch.runners.TorchInferenceRunner
    selection:
      members:
        - run

::: mozuma.torch.runners.TorchInferenceMultiGPURunner
    selection:
      members:
        - run

## Training

::: mozuma.torch.runners.TorchTrainingRunner
    selection:
      members:
        - run

## Write your own runner

::: mozuma.runners.BaseRunner
