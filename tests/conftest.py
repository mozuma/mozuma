import os
import random

import numpy as np
import pytest
import torch
from _pytest.fixtures import SubRequest

from mlmodule.contrib.arcface import ArcFaceFeatures
from mlmodule.contrib.clip import CLIPViTB32ImageEncoder
from mlmodule.contrib.densenet import DenseNet161ImageNetFeatures, DenseNet161ImageNetClassifier, \
    DenseNet161PlacesFeatures, DenseNet161PlacesClassifier
from mlmodule.contrib.mtcnn import MTCNNDetector
from mlmodule.contrib.resnet import ResNet18ImageNetFeatures, ResNet18ImageNetClassifier


@pytest.fixture(scope='session', params=["cpu", "cuda"])
def torch_device(request: SubRequest) -> torch.device:
    """Fixture for the PyTorch device, run GPU only when CUDA is available

    :param request:
    :return:
    """
    if request.param != 'cpu' and not torch.cuda.is_available():
        pytest.skip(f"Skipping device {request.param}, CUDA not available")
    if request.param != 'cpu' and os.environ.get('CPU_ONLY_TESTS') == 'y':
        pytest.skip(f"Skipping device {request.param}, tests are running for CPU only")
    return torch.device(request.param)


@pytest.fixture(scope='session')
def gpu_torch_device() -> torch.device:
    """Fixture to get a GPU torch device. The test will be skipped if not GPU is available"""
    if not torch.cuda.is_available():
        pytest.skip(f"Skipping test as is CUDA not available")
    if os.environ.get('CPU_ONLY_TESTS') == 'y':
        pytest.skip(f"Skipping as tests are running for CPU only")
    return torch.device('cuda')


@pytest.fixture(scope='session')
def set_seeds():
    def _set_seeds(val=123):
        torch.manual_seed(val)
        torch.cuda.manual_seed(val)
        np.random.seed(val)
        random.seed(val)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
    return _set_seeds


@pytest.fixture(params=[
    ResNet18ImageNetFeatures,
    ResNet18ImageNetClassifier,
    DenseNet161ImageNetFeatures,
    DenseNet161ImageNetClassifier,
    DenseNet161PlacesFeatures,
    DenseNet161PlacesClassifier,
    CLIPViTB32ImageEncoder,
    MTCNNDetector,
    ArcFaceFeatures
])
def data_platform_scanner(request: SubRequest):
    """Fixture for generic tests of Modules to be used in the data platform

    :param request:
    :return:
    """
    return request.param
