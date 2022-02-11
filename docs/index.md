# Home

MLModule is a library containing a collection of machine learning models
with standardised interface to load/save weights, run inference and training.

It aims at providing high-level abstractions on top of inference and training loops
while allowing extensions via callbacks classes.
These callbacks control the way the output of a runner is handled
(i.e. features, labels, model weigths...)

Features:

- [x] Model zoo
- [x] PyTorch inference
- [ ] PyTorch training
- [ ] Multi-GPU support
