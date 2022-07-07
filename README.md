# MoZuMa

MoZuMa is a library containing a collection of machine learning models
with standardised interface to run inference, train and manage model state files.

It aims at providing high-level abstractions called [runners](references/runners.md)
on top of inference and training loops
while allowing extensions via [callbacks](references/callbacks.md).
These callbacks control the way the output of a runner is handled
(i.e. features, labels, model weights...).

We also try to keep as few dependencies as possible.
Meaning models will be mostly implemented from
modules available in deep learning frameworks (such as `PyTorch` or `torchvision`).

See the  for more information.

## Quick links

- [Documentation](https://mozuma.github.io/mozuma/)
- [Installation](https://mozuma.github.io/mozuma/0-installation)
- [Getting started](https://mozuma.github.io/mozuma/1-getting-starteg)
- [Models](https://mozuma.github.io/mozuma/models/)


## Example gallery

See `docs/examples/` for a collection of ready to use notebooks.
