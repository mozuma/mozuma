# Developer guide

## Installation

### Using `tox`

This method requires [`conda`](https://docs.conda.io/en/latest/) and [`tox`](https://tox.readthedocs.io/en/latest/) to be installed.

Create a development environment:

```shell
# CPU development
tox --devenv venv -e py39
# or CUDA 11.1
tox --devenv venv -e cuda111-py39
```

The environment can be activated with:

```shell
conda activate ./venv
```

### Using `pip`

This method requires `pip` to be installed

```bash
pip install -r requirements.txt
# To install MLModule in development mode with all dependencies
pip install -e .
```

## Testing

Testing can be done using `tox`

```shell
# CPU testing
tox -e py39
# or CUDA 11.1
tox -e cuda111-py39
```

or with directly using `pytest` on an environment with all dependencies installed

```shell
pytest
```

## Code quality

We use `black` as formatter. Install pre commit hooks for systematic styling on commits:

```shell
pip install pre-commit
pre-commit install
```

## Requirements

Updating requirements should be done in `setup.cfg`.
To update the `requirement.txt` file run:

```bash
pip-compile --extra full --upgrade
```

## Publish a new version

* Push the new version to the `master` branch
* Create a GitHUB release on the branch `master` with format `vX.Y.Z`. For instance, `v0.1.1`.

## Upload new model weights

!!! note
    You will need to set the `GH_TOKEN` environment variable to the token given by
    `gh auth status -t` of the [GitHUB CLI](https://cli.github.com/manual/gh_auth_status).

* Add the model and provider stores to the `scripts/update_public_store.py`.
  They should be added in a function that returns a list of tuple with model and provider store.
* Update the `get_all_models_stores` to iterate over your new function.
* Run the `update_public_store` scripts. This script accepts a `--dry-run` argument
  to see changes before actually uploading models.
