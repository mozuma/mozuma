# MoZuMa

MoZuMa is a model zoo for multimedia search application. It provides an easy to use interface to run models for:

- **Text to image retrieval**: Rank images by their similarity to a text query.
- **Image similarity search**: Rank images by their similarity to query image.
- **Image classification**: Add labels to images.
- **Face detection**: Detect and retrieve images with similar faces.
- **Object detection**: Detect and retrieve images with similar objects.
- **Video keyframes extraction**: Retrieve the important frames of a video.
  Key-frames are used to apply all the other queries on videos.
- **Multilingual text search**: Rank similar sentences from a text query in multiple languages.

## Installation

We support `Python >= {{develop.python_version}}` and `PyTorch >= {{develop.pytorch_version}}`.
However, it is likely that MoZuMa can be run on previous versions of PyTorch, we are simply not testing it
for versions before `{{develop.pytorch_version}}`.

!!! info

    `mozuma` requires `PyTorch` and we recommend to follow the [PyTorch's documentation](https://pytorch.org/get-started/locally/#start-locally) for the installation procedure.

```shell
pip install mozuma
```

## How to search multimedia collections ?

Running a model with MoZuMa will always have the same structure:

```python
# Getting a dataset of images (1)
dataset = ImageDataset(LocalBinaryFilesDataset(paths))

# Model definition (2)
model = torch_resnet_imagenet("resnet18")

# Creating the callback to collect data (3)
features = CollectFeaturesInMemory()
labels = CollectLabelsInMemory()
callbacks = [features, labels]

# Getting the torch runner for inference (4)
runner = TorchInferenceRunner(
    model=model,
    dataset=dataset,
    callbacks=callbacks,
    options=TorchRunnerOptions(tqdm_enabled=True),
)
runner.run()
```

1. See [Datasets](references/datasets.md) for a list of available datasets.
2. List of all [models](models/index.md)
3. List of available [callbacks](references/callbacks.md).
4. List of available [runners](references/runners.md)

The following sections are discussing different search scenario
for which we will be changing the `model` and `callbacks` variables

## Search with labels

- **Models**: See [models](models/index.md) marked with purpose *labeling*
- **Callbacks**: [CollectLabelsInMemory][mozuma.callbacks.CollectLabelsInMemory]
- **Examples**: [Getting started](examples/0-getting-started.md)

Some models have been trained to do classification, they produce labels to described an image.
See [`mozuma.labels`](https://github.com/mozuma/mozuma/tree/master/src/mozuma/labels)
for a list of label sets. The models supporting labels are:

- [DenseNet](models/DenseNet.md)
- [ResNet](models/ResNet.md)

??? tip "Work with custom labels"

    We also provide a way to add custom labels on top of existing features.
    See the [classification module](models/Classification.md) or
    the example [CIFAR10 Image Classification](examples/train_cifar10.md).

## Search with features

- **Models**: See [models](models/index.md) marked with purpose *Text-to-image*,
  *Image similarity*, *Object similarity*, *Sentence similarity*, *Face recognition*
- **Callbacks**: [CollectFeaturesInMemory][mozuma.callbacks.CollectFeaturesInMemory]
  [CollectBoundingBoxesInMemory][mozuma.callbacks.CollectBoundingBoxesInMemory]
- **Examples**: [Text-to-image with CLIP](examples/text_to_retrieval_with_clip.md),
  [Face Similarity Search](examples/arcface.md),
  [Multilingual text search](examples/test_distiluse_multilingual.md)

Sometimes what we are looking for is not included in the labels.
In this case, we can use similarity search to find images from a query images.
This can be done by comparing embeddings (or features) of images.

??? question "What are embeddings?"

    Embeddings (or features) are dense vectors that contain a lot of information on the image and they have interesting properties. Usually embeddings that are close together will represent entities that are similar.

For instance, [Densenet pre-trained on Places365][mozuma.models.densenet.torch_densenet_places365] embeddings will contain information on the context of the image, where it is taking place, the scene.
Therefore, searching similar images will return images taken in a similar scene.

On the other hand the version [pre-trained on ImageNet][mozuma.models.densenet.torch_densenet_imagenet] will tend to focus on the main subject.

Similarly, embeddings can also be used to find similar faces or objects with [ArcFace][mozuma.models.arcface.torch_arcface_insightface] and [VinVL][mozuma.models.vinvl.torch_vinvl_detector] respectively.

Finally, embeddings are also used by [CLIP](models/CLIP.md) to find images similar to a text description.

## Handling videos

- **Model**: See [models](models/index.md) marked with purpose *Key-frames*
- **Callbacks**: [CollectVideoFramesInMemory][mozuma.callbacks.CollectVideoFramesInMemory]
- **Examples**: [Video Key-Frames extraction model](examples/keyframes.md)

Searching videos can be inefficient if we have to apply models to all frames.
Here, we provide [Video key-frames extractors](models/keyframes.md) to select only
a few representative frames for a video. We then apply the other models on these
key-frames.

## Going further

- [Understand MoZuMa's search capabilities with code examples](examples/1-overview.md)
- [Add a model to MoZuMa](contributing/1-add-a-model.md)
