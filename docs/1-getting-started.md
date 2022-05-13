---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3.7.10 ('mlmodule')
    language: python
    name: python3
---


# Getting started

## Code-based usage

This guide runs through the inference of a PyTorch ResNet model pre-trained on imagenet.

First, we need to create a dataset of images, for this we will be using an `ImageDataset`.

```python
from mlmodule.torch.datasets import LocalBinaryFilesDataset, ImageDataset

# Getting a dataset of images (1)
dataset = ImageDataset(
    LocalBinaryFilesDataset(
        paths=[
            "../tests/fixtures/cats_dogs/cat_0.jpg",
            "../tests/fixtures/cats_dogs/cat_90.jpg",
        ]
    )
)
```


1.  See [Datasets](references/datasets.md) for a list of available datasets.

Next, we need to load the ResNet PyTorch module specifying the `resnet18` architecture.
The model is initialised with weights pre-trained on ImageNet[@deng2009imagenet].

```python
from mlmodule.models.resnet import torch_resnet_imagenet

# Model definition (1)
resnet = torch_resnet_imagenet("resnet18")
```


1. List of all [models](models/index.md)

Once we have the model initialized, we need to define what we want to do with it.
In this case, we'll run an inference loop using the `TorchInferenceRunner`.

Note that we pass two callbacks to the runner: `CollectFeaturesInMemory` and `CollectLabelsInMemory`.
They will be called to save the features and labels in-memory.

```python
from mlmodule.callbacks import CollectFeaturesInMemory, CollectLabelsInMemory
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

Now that the runner is initialised, we run it with the method `run`.

The callbacks have accumulated the features and labels in memory and we print their content.

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
