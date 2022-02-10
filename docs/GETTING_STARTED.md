
# Getting started

This guide runs through the inference of a PyTorch ResNet model pretrained on imagenet.

First, we need to create a dataset of images, for this we will be using the `OpenFileDataset`.

```python
from mlmodule.v2.torch.datasets import OpenImageFileDataset

# Getting a dataset of images
dataset = OpenImageFileDataset(
    paths=["tests/fixtures/cats_dogs/cat_0.jpg", "tests/fixtures/cats_dogs/cat_90.jpg"]
)
```

Next, we need to load the ResNet PyTorch module specifying the `resnet18` architecture.
The model is initialised with weights provided by the `MLModuleModelStore`.

```python
from mlmodule.contrib.resnet import TorchResNetModule
from mlmodule.v2.stores import MLModuleModelStore

# Model definition
resnet = TorchResNetModule("resnet18")

# Getting pretrained model
store = MLModuleModelStore()
store.load(resnet)
```

Once we have a model initialized, we need to define what we want to do with it.
In this case, we'll run an inference loop using the `TorchInferenceRunner`.

Note that we pass two callbacks to the runner: `CollectFeaturesInMemory` and `CollectLabelsInMemory`.
They will be called to collect the resulting features and labels for each batch.

```python
from mlmodule.v2.helpers.callbacks import CollectFeaturesInMemory, CollectLabelsInMemory
from mlmodule.v2.torch.options import TorchRunnerOptions
from mlmodule.v2.torch.runners import TorchInferenceRunner

# Creating the callback to collect data
features = CollectFeaturesInMemory()
labels = CollectLabelsInMemory()

# Getting the torch runner for inference
runner = TorchInferenceRunner(
    model=resnet,
    dataset=dataset,
    callbacks=[features, labels],
    options=TorchRunnerOptions(tqdm_enabled=True),
)
```

When defined, the runner can be run and the callback objects will contain the features and labels.

```python
# Executing inference
runner.run()

# Printing the features
print(features.indices, features.features)

# Printing labels
print(labels.indices, labels.labels)
```
