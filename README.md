# MLModule

The goal of this library is to standardise the definition of models
used in the AI Data platform.

## Requirements

* Python 3.7 or newer

## Installation

From the git repository:

```bash
pip install git+ssh://git@github.com/LSIR/mlmodule.git
```

For convenience, we ship a base Docker image https://hub.docker.com/repository/docker/lsirepfl/mlmodulekit which contains dependencies that can be hard to install (for instance PyTorch or MMCV). See [MLModuleKit](mlmodulekit/README.md).

## Getting started code

In this example we run a ResNet trained on ImageNet against an image

TODO

## Run a model from CLI

This only works for models accepting images for now.
For instance, to run CLIP on all images in a folder:

```bash
python -m mlmodule.cli run clip.CLIPViTB32ImageEncoder folder/* --batch-size 256 --num-workers 12
```

## Models

Using the pretrained models requires access to the `lsir-public-assets` S3 bucket.
Follow the [dedicated guide](https://github.com/LSIR/dataplatform-infra/tree/main/lsir-public-assets#read-bucket-content)
to get access to the pretrained models.

Available models

* [ResNet](docs/models/ResNet.md): ImageNet
* Face detection with [MTCNN](docs/models/MTCNN.md) and [ArcFace](docs/models/ArcFace.md) or [MagFace](docs/models/MagFace.md).
* [DenseNet](docs/models/DenseNet.md): ImageNet and Places365
* [VinVL](docs/models/VinVL.md): Object detection


## Development

See the [development guide](docs/DEVELOP.md) and the
[model contribution guide](docs/CONTRIBUTE.md).
