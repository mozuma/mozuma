# Base PyTorch docker images

This folder contains Dockerfiles to create a conda environment with PyTorch installed.
The image can be pulled with:

```shell
docker pull lsirepfl/pytorch:<version>
```

## Tags

### `v1.7.1-py3.7.10-cu110`

Standard release of PyTorch from the official PyTorch distribution:

* Python 3.7.10
* Cuda 11.0
* PyTorch 1.7.1
* TorchVision 0.8.2

### `pc32-v1.7.1-py3.7.10-cu110`

Compiled PyTorch for LSIRPC32 GPU.

* Python 3.7.10
* Cuda 11.0
* PyTorch 1.7.1
* TorchVision 0.8.2

## Build images

There are two different Dockerfiles:

* `Dockerfile.from-binary` builds an image with official binaries distributed by PyTorch.
* `Dockerfile.from-source` builds PyTorch from source.

These images accept the build arguments:

* `PYTHON_VERSION` (default=`3.7.10`)
* `CUDA_VERSION` (default=`11.0`)
* `CUDA_MAGMA_VERSION` (default=`110`, only used in `Dockerfile.from-source`)
* `TORCH_VERSION` (default=`1.7.1`)
* `TORCHVISION_VERSION` (default=`0.8.2`)
