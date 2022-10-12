# Developer setup

## Installation

This section covers the installation of a development environment for contributing to MoZuMa.
If you want to use MoZuMa in your project, see [installation instructions](../index.md#installation) instead.

We are developing on Python {{develop.python_version}} and PyTorch {{develop.pytorch_version}}.

### Using `tox` (recommended)

This method is the recommended method as it will install a complete environment with PyTorch.
It requires [`conda`](https://docs.conda.io/en/latest/) and
[`tox`](https://tox.readthedocs.io/en/latest/) to be installed.

Create a development environment:

```shell
# CPU development
tox --devenv venv -e {{develop.tox_env_version}}
# or with GPU support
tox --devenv venv -e {{develop.tox_env_version_cuda}}
```

The environment can be activated with:

```shell
conda activate ./venv
```

### Using `pip`

This method requires you to install PyTorch and TorchVision
(see [PyTorch documentation](https://pytorch.org/)).

```bash
pip install -r requirements.txt
# To install MoZuMa in development mode with all dependencies
pip install -e .
```

## Testing

Testing can be done using `tox`:

```shell
# CPU testing
tox -e {{develop.tox_env_version}}
# or with GPU support
tox -e {{develop.tox_env_version_cuda}}
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

## Publish a new version

* Push the new version to the `master` branch
* Test the release on all supported Python / PyTorch versions with the command.
  ```shell
  tox -e "{{release.tox_env_version}}"
  ```
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

## Update documentation

Install requirements to build the documentation:

```shell
pip install -r docs/requirements.txt
```

We are using [`mkdocs-material`](https://squidfunk.github.io/mkdocs-material/).
It allows for previewing the changes with hot-reload:

```shell
mkdocs serve
```

The documentation can then be deployed to GitHUB pages manually with:

```shell
mkdocs gh-deploy
```

Or automatically when merging a Pull Request into the main branch with GitHUB actions.

## Adding notebooks to the documentation

When adding new notebooks in the documentation, the bash script `docs/pair-notebooks.sh` should be executed to create the notebooks in Markdown version.

Once the notebooks are paired to a markdown version, they can be updated with the `docs/sync-notebooks.sh` script.
