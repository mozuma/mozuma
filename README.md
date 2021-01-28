# MLModule

The goal of this library is to standardise the definition of models
used in the AI Data platform.

## Installation

```bash
pip install git+ssh://git@github.com/LSIR/mlmodule.git
```

## Models

* [ResNet](docs/ResNet.md)
* Face detection with [MTCNN](docs/MTCNN.md)


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

## Contribute

See [Contribute guide](CONTRIBUTE.md)
