# MLModule

The goal of this library is to standardise the definition of models
used in the AI Data platform.

## Installation

```bash
pip install git+ssh://git@github.com/LSIR/mlmodule.git
```

## Models

* [ResNet](docs/ResNet.md)


## Installation for development

Installing requirements

```bash
pip install -r requirement.txt dev-requirements.txt
```

Updating requirements should be done in `requirements.in` and `dev-requirements.in` files.
The actual requirements files should be updated with 
(see [pip-tools](https://github.com/jazzband/pip-tools) for more documentation):

```bash
pip-compile requirements.in
pip-compile dev-requirements.in
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
