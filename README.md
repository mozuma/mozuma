# MLModule

MLModule is a library containing a collection of machine learning models
with standardised interface to load/save weights, run inference and training.

It aims at providing high-level abstractions called [runners](references/runners.md)
on top of inference and training loops
while allowing extensions via [callbacks](references/callbacks.md).
These callbacks control the way the output of a runner is handled
(e.g. features, labels, model weights...).

We also try to keep as few dependencies as possible.
Meaning models will be mostly implemented from
modules available in deep learning frameworks (such as `PyTorch` or `torchvision`).

See the [documentation](https://lsir.github.io/mlmodule/) for more information.
