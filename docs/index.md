# Home

## Overview

MLModule is a library containing a collection of machine learning models
with standardised interface to load/save weights, run inference and training.

It aims at providing high-level abstractions called [runners](references/runners.md)
on top of inference and training loops
while allowing extensions via [callbacks](references/callbacks.md).
These callbacks control the way the output of a runner is handled
(i.e. features, labels, model weights...).

We also try to keep as few dependencies as possible.
Meaning models will be mostly implemented from
modules available in deep learning frameworks (such as `PyTorch` or `torchvision`).

Go ahead to the [getting started](1-getting-started.md) guide for an overview of MLModule.

## Features

- [x] Model zoo
- [x] PyTorch inference
- [ ] PyTorch training
- [ ] Multi-GPU support

## Installation

From the git repository:

```bash
pip install git+ssh://git@github.com/LSIR/mlmodule.git
```

For convenience, we ship a base Docker image
[https://hub.docker.com/repository/docker/lsirepfl/mlmodulekit](https://hub.docker.com/repository/docker/lsirepfl/mlmodulekit)
which contains dependencies that can be hard to install (for instance PyTorch or MMCV). See [MLModuleKit](3-mlmodulekit.md).
