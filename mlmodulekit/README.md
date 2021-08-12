# MLModule Kit

Docker images with PyTorch related libraries pre-installed to use MLModule

## Versions

All versions are available with the suffix `-ca35`. 
These images are compiled for the CUDA compute ability version 3.5 (see https://en.wikipedia.org/wiki/CUDA), 
they must be used when using the LSIR PC32 machine.

### `1`, `1-ca35`

* Python 3.7
* CUDA 11.0
* PyTorch 1.7.1
* TorchVision 0.8.2
* MMCV full 1.3.11
* MMDet at [`2894516`](https://github.com/open-mmlab/mmdetection.git@2894516bacf9ff82c3bc6d6970019d0890a993aa)
* CLIP at [`8cad3a7`](https://github.com/openai/CLIP.git@8cad3a736a833bc4c9b4dd34ef12b52ec0e68856)
* Facenet Pytorch facenet-pytorch==2.5.2
