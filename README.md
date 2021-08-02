# MLModule

The goal of this library is to standardise the definition of models
used in the AI Data platform.

## Installation

From the git repository:

```bash
pip install git+ssh://git@github.com/LSIR/mlmodule.git
```

As a docker image:

```shell
docker pull lsirepfl/mlmodule:<version>
```

## Models

Using the pretrained models requires access to the `lsir-public-assets` S3 bucket.
Follow the [dedicated guide](https://github.com/LSIR/dataplatform-infra/tree/main/lsir-public-assets#read-bucket-content)
to get access to the pretrained models.

Available models

* [ResNet](docs/ResNet.md): ImageNet
* Face detection with [MTCNN](docs/MTCNN.md) and [ArcFace](docs/ArcFace.md)
* [DenseNet](docs/DenseNet.md): ImageNet and Places365


## Run a model from CLI

This only works for models accepting images for now. 
For instance, to run CLIP on all images in a folder:

```bash
python -m mlmodule.cli run clip.CLIPViTB32ImageEncoder folder/* --batch-size 256 --num-workers 12
```


## Installation for development


```bash
# For an installation with all model dependencies
make install

# For a minimal installation with just MLModule dependencies
make install-minimal

# For development
make develop
```

## Build the Docker image

The image can be built tested and pushed with one command

```shell
make release-docker-image
```

Alternatively, the image can be build from a different base image (here to add support for PC32):

```shell
make release-docker-image IMAGE_TAG_PREFIX=pc32-v BASE_IMAGE=lsirepfl/pytorch:pc32-v1.7.1-py3.7.10-cu110
```

## Requirements management

Updating requirements should be done in `setup.cfg`. 
The update the `requirement.txt` file run:

```bash
make requirements
```

To install the requirements

```bash
make env-install
```

## Tests

Install package for development

```bash
make develop
```

Run tests

```bash
make test
```

## Publish a new version

* Push the new version to the `master` branch
* Add a tag on the branch with the format `vX.Y.Z`. For instance, `v0.1.1`.

## Contribute

See [Contribute guide](CONTRIBUTE.md)
