# MLModule Kit

Docker images with PyTorch related libraries pre-installed to use MLModule.

```
docker pull lsirepfl/mlmodulekit:<version>
```

## Versions

List of image tags and installed libraries.

All versions are available with the suffix `-ca35`. 
These images are compiled for the CUDA compute ability version 3.5 (see https://en.wikipedia.org/wiki/CUDA), 
they must be used when using the LSIR PC32 machine.

### `1`, `1-ca35`

* Python 3.7
* CUDA 11.0
* PyTorch 1.7.1
* TorchVision 0.8.2
* MMCV full 1.3.11

### `2`

* Python 3.7
* CUDA 11.1
* PyTorch 1.9.1
* TorchVision 0.10.1
* MMCV full 1.3.14

### `3`

* Python 3.9
* CUDA 11.1
* PyTorch 1.9.1
* TorchVision 0.10.1
