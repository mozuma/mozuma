
# Getting started

## Code-based usage

This guide runs through the inference of a PyTorch ResNet model pre-trained on imagenet.

First, we need to create a dataset of images, for this we will be using the `OpenFileDataset`.

```python
from mlmodule.torch.datasets import OpenImageFileDataset

# Getting a dataset of images (1)
dataset = OpenImageFileDataset(
    paths=[
        "tests/fixtures/cats_dogs/cat_0.jpg",
        "tests/fixtures/cats_dogs/cat_90.jpg"
    ]
)
```

1.  See [Datasets](references/datasets.md) for a list of available datasets.

Next, we need to load the ResNet PyTorch module specifying the `resnet18` architecture.
The model is initialised with weights provided by the `MLModuleModelStore`.

```python
from mlmodule.models.resnet import TorchResNetModule
from mlmodule.states import StateKey
from mlmodule.stores import MLModuleModelStore

# Model definition (1)
resnet = TorchResNetModule("resnet18")

# Getting pre-trained model (2)
store = MLModuleModelStore()
# Getting the state pre-trained on ImageNet (3)
store.load(
    resnet,
    StateKey(state_type=resnet.state_type, training_id="imagenet")
)
```

1. List of all [models](models/index.md)
2. List of all [stores](references/stores.md)
3. Description of how states are handled is available is [state's reference](references/states.md)

Once we have a model initialized, we need to define what we want to do with it.
In this case, we'll run an inference loop using the `TorchInferenceRunner`.

Note that we pass two callbacks to the runner: `CollectFeaturesInMemory` and `CollectLabelsInMemory`.
They will be called to collect the resulting features and labels for each batch.

```python
from mlmodule.callbacks import (
    CollectFeaturesInMemory, CollectLabelsInMemory
)
from mlmodule.torch.options import TorchRunnerOptions
from mlmodule.torch.runners import TorchInferenceRunner

# Creating the callback to collect data (1)
features = CollectFeaturesInMemory()
labels = CollectLabelsInMemory()

# Getting the torch runner for inference (2)
runner = TorchInferenceRunner(
    model=resnet,
    dataset=dataset,
    callbacks=[features, labels],
    options=TorchRunnerOptions(tqdm_enabled=True),
)
```

1. List of available [callbacks](references/callbacks.md).
2. List of available [runners](references/runners.md)

When defined, the runner can be run and the callback objects will contain the features and labels.

```python
# Executing inference
runner.run()

# Printing the features
print(features.indices, features.features)

# Printing labels
print(labels.indices, labels.labels)
```

## Command-line interface

MLModule exposes a command-line interface. See `python -m mlmodule -h` for a list of available commands.

For instance, one can run a ResNet model against a list of local images with the following command:

```shell
python -m mlmodule run ".resnet.modules.TorchResNetModule(resnet18)" *.jpg
```

It prints the results (features and labels) in JSON format.

Similarly, we can extract the key-frames from videos:

```shell
python -m mlmodule run ".keyframes.selectors.resnet_key_frame_selector(resnet18, 10)" *.mp4 --file-type vi --batch-size 1
```
