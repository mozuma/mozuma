# Introduction

MoZuma is a library containing a collection of machine learning models
with standardised interface to run inference, train and manage model state files.

It provides high-level abstractions called [runners](references/runners.md)
to run inference or training against a model.
Saving the predictions or the model's state is controlled
via easily extensible [callbacks](references/callbacks.md).

MoZuMa should also be easy to install,
therefore, we try to keep as few dependencies as possible.

To start using MoZuMa, go ahead to the [installation](0-installation.md)
and [getting started](1-getting-started.md) guides !

## Features

- [x] Model zoo
- [x] PyTorch inference
- [x] PyTorch training
- [x] Multi-GPU support
