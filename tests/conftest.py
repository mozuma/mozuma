import os
from typing import Callable, Set, Type

import numpy as np
import pytest
import torch
from _pytest.fixtures import SubRequest

from mlmodule.contrib.arcface import ArcFaceFeatures
from mlmodule.contrib.clip import CLIPViTB32ImageEncoder
from mlmodule.contrib.densenet import (
    DenseNet161ImageNetClassifier,
    DenseNet161ImageNetFeatures,
    DenseNet161PlacesClassifier,
    DenseNet161PlacesFeatures,
)

# from mlmodule.contrib.keyframes.v1 import TorchMLModuleKeyFrames
from mlmodule.contrib.magface.features import MagFaceFeatures
from mlmodule.contrib.mtcnn import MTCNNDetector
from mlmodule.contrib.mtcnn.detector_ori import MTCNNDetectorOriginal
from mlmodule.contrib.resnet import ResNet18ImageNetClassifier, ResNet18ImageNetFeatures
from mlmodule.contrib.resnet.modules import TorchResNetModule
from mlmodule.contrib.vinvl import VinVLDetector
from mlmodule.torch.base import BaseTorchMLModule
from mlmodule.torch.data.images import ImageDataset
from mlmodule.types import StateDict
from mlmodule.utils import list_files_in_dir
from mlmodule.v2.base.models import ModelWithStateFromProvider


@pytest.fixture(scope="session", params=["cpu", "cuda"])
def torch_device(request: SubRequest) -> torch.device:
    """Fixture for the PyTorch device, run GPU only when CUDA is available

    :param request:
    :return:
    """
    if request.param != "cpu" and not torch.cuda.is_available():
        pytest.skip(f"Skipping device {request.param}, CUDA not available")
    if request.param != "cpu" and os.environ.get("CPU_ONLY_TESTS") == "y":
        pytest.skip(f"Skipping device {request.param}, tests are running for CPU only")
    return torch.device(request.param)


@pytest.fixture(scope="session")
def gpu_torch_device() -> torch.device:
    """Fixture to get a GPU torch device. The test will be skipped if not GPU is available"""
    if not torch.cuda.is_available():
        pytest.skip("Skipping test as is CUDA not available")
    if os.environ.get("CPU_ONLY_TESTS") == "y":
        pytest.skip("Skipping as tests are running for CPU only")
    return torch.device("cuda")


@pytest.fixture(
    params=[
        lambda: TorchResNetModule("resnet18")
        # CLIPViTB32ImageEncoder,
        # MTCNNDetector,
        # ArcFaceFeatures,
        # MagFaceFeatures,
        # TorchMLModuleKeyFrames,
        # VinVLDetector - too slow to download
    ]
)
def module_pretrained_by_provider(
    request: SubRequest,
) -> ModelWithStateFromProvider:
    """Returns a module that implements DownloadPretrainedStateFromProvider"""
    return request.param


@pytest.fixture(
    params=[
        lambda: TorchResNetModule("resnet18")
        # CLIPViTB32ImageEncoder,
        # MTCNNDetector,
        # ArcFaceFeatures,
        # MagFaceFeatures,
        # TorchMLModuleKeyFrames,
        # VinVLDetector - too slow to download
    ]
)
def module_pretrained_mlmodule_store(
    request: SubRequest,
) -> ModelWithStateFromProvider:
    """Returns a module that implements DownloadPretrainedStateFromProvider"""
    return request.param


@pytest.fixture(
    params=[
        ResNet18ImageNetFeatures,
        ResNet18ImageNetClassifier,
        DenseNet161ImageNetFeatures,
        DenseNet161ImageNetClassifier,
        DenseNet161PlacesFeatures,
        DenseNet161PlacesClassifier,
        CLIPViTB32ImageEncoder,
        MTCNNDetector,
        MTCNNDetectorOriginal,
        ArcFaceFeatures,
        MagFaceFeatures,
        # TorchMLModuleKeyFrames,
        VinVLDetector,
    ]
)
def data_platform_scanner(request: SubRequest):
    """Fixture for generic tests of Modules to be used in the data platform

    :param request:
    :return:
    """
    return request.param


@pytest.fixture(
    params=[
        ResNet18ImageNetFeatures,
        DenseNet161ImageNetFeatures,
        DenseNet161PlacesFeatures,
        CLIPViTB32ImageEncoder,
        MTCNNDetector,
        MTCNNDetectorOriginal,
        VinVLDetector,
    ]
)
def image_module(request: SubRequest) -> Type[BaseTorchMLModule]:
    """MLModules operating on images"""
    return request.param


@pytest.fixture(scope="session")
def gpu_only_modules() -> Set[Type[BaseTorchMLModule]]:
    """MLModules operating on images"""
    return set()


@pytest.fixture
def assert_state_dict_equals() -> Callable[[StateDict, StateDict], None]:
    def _assert_state_dict_equals(sd1: StateDict, sd2: StateDict) -> None:
        for key in sd1:
            np.testing.assert_array_equal(sd1[key], sd2[key])

    return _assert_state_dict_equals


@pytest.fixture(scope="session")
def image_dataset() -> ImageDataset:
    """Sample image dataset"""
    base_path = os.path.join("tests", "fixtures", "faces")
    file_names = list_files_in_dir(base_path, allowed_extensions=("jpg",))
    return ImageDataset(file_names)


@pytest.fixture(scope="session")
def video_file_path() -> str:
    return os.path.join("tests", "fixtures", "video", "test.mp4")
