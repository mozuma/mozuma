# MLModule

The goal of this library is to standardise the definition of models
used in the AI Data platform.

## Installation

```bash
pip install git+ssh://git@github.com/LSIR/mlmodule.git
```

## Models

Using the pretrained models requires access to the `lsir-public-assets` S3 bucket.
Follow the [dedicated guide](https://github.com/LSIR/dataplatform-infra/tree/main/lsir-public-assets#read-bucket-content)
to get access to the pretrained models.

Available models

* [ResNet](docs/ResNet.md): ImageNet
* Face detection with [MTCNN](docs/MTCNN.md) and [ArcFace](docs/ArcFace.md)
* [DenseNet](docs/DenseNet.md): ImageNet and Places365


## Installation for development

Installing requirements

```bash
# With pip
pip install -r requirements.txt
# Or with pip-tools to install only the required dependencies
pip-sync
```

Updating requirements should be done in `requirements.in` and `dev-requirements.in` files.
The actual requirements files should be updated with 
(see [pip-tools](https://github.com/jazzband/pip-tools) for more documentation):

```bash
pip-compile
```

## Tests

Install package for development

```bash
pip install -e .
```

Run tests

```bash
pytest
```

## Publish a new version

* Update the version number in `setup.py`
* Push a new commit to the `master` branch with the new version
* Add a tag on the branch with the format `vX.Y.Z`. For instance, `v0.1.1`.

## Contribute

See [Contribute guide](CONTRIBUTE.md)
