import os
from typing import Callable, List, Set, Type

import numpy as np
import pytest
import torch
from _pytest.fixtures import SubRequest

from mlmodule.contrib.arcface import ArcFaceFeatures
from mlmodule.contrib.clip.image import CLIPImageModule
from mlmodule.contrib.clip.stores import CLIPStore
from mlmodule.contrib.clip.text import CLIPTextModule
from mlmodule.contrib.densenet import (
    DenseNet161ImageNetClassifier,
    DenseNet161ImageNetFeatures,
    DenseNet161PlacesClassifier,
    DenseNet161PlacesFeatures,
)
from mlmodule.contrib.keyframes.encoders import VideoFramesEncoder
from mlmodule.contrib.keyframes.selectors import KeyFrameSelector
from mlmodule.contrib.magface.modules import TorchMagFaceModule
from mlmodule.contrib.magface.stores import MagFaceStore
from mlmodule.contrib.mtcnn.modules import TorchMTCNNModule
from mlmodule.contrib.mtcnn.stores import FaceNetMTCNNStore
from mlmodule.contrib.resnet.modules import TorchResNetModule
from mlmodule.contrib.resnet.stores import ResNetTorchVisionStore
from mlmodule.contrib.vinvl import VinVLDetector
from mlmodule.torch.base import BaseTorchMLModule
from mlmodule.torch.data.images import ImageDataset
from mlmodule.types import StateDict
from mlmodule.utils import list_files_in_dir
from mlmodule.v2.testing import ModuleTestConfiguration
from mlmodule.v2.torch.modules import TorchMlModule

MODULE_TO_TEST: List[ModuleTestConfiguration] = [
    # ResNet
    ModuleTestConfiguration(
        "torchresnet18",
        lambda: TorchResNetModule("resnet18"),
        batch_factory=lambda: torch.rand(
            [2, 3, 224, 224]
        ),  # batch, channels, width, height
        provider_store=ResNetTorchVisionStore(),
        provider_store_training_ids={"imagenet"},
    ),
    # CLIP
    ModuleTestConfiguration(
        "clip-image-rn50",
        lambda: CLIPImageModule("RN50"),
        batch_factory=lambda: torch.rand(
            [2, 3, 224, 224]
        ),  # batch, channels, width, height
        provider_store=CLIPStore(),
        provider_store_training_ids={"clip"},
    ),
    ModuleTestConfiguration(
        "clip-text-rn50",
        lambda: CLIPTextModule("RN50"),
        batch_factory=lambda: torch.randint(10, size=(2, 77)),  # batch, ctx_len
        provider_store=CLIPStore(),
        provider_store_training_ids={"clip"},
    ),
    # MTCNN
    ModuleTestConfiguration(
        "mtcnn",
        lambda: TorchMTCNNModule(),
        batch_factory=lambda: [torch.rand([720 + i * 10, 720, 3]) for i in range(5)],
        provider_store=FaceNetMTCNNStore(),
        provider_store_training_ids={"facenet"},
    ),
    # MagFace
    ModuleTestConfiguration(
        "magface",
        lambda: TorchMagFaceModule(),
        batch_factory=lambda: torch.rand(
            (2, 3, 112, 112)
        ),  # batch, channels, width, height
        provider_store=MagFaceStore(),
        provider_store_training_ids={"magface"},
    ),
    # Key-frames
    ModuleTestConfiguration(
        "frames-encoder-rn18",
        lambda: VideoFramesEncoder(TorchResNetModule("resnet18")),
        batch_factory=lambda: (
            [torch.range(0, 1), torch.range(0, 3)],  # Frame indices
            [
                torch.rand([1, 3, 224, 224]),  # frame_idx, channels, width, height
                torch.rand([3, 3, 224, 224]),  # frame_idx, channels, width, height
            ],
        ),
    ),
    ModuleTestConfiguration(
        "frames-selector-rn18",
        lambda: KeyFrameSelector(TorchResNetModule("resnet18")),
        batch_factory=lambda: (
            [torch.range(0, 1), torch.range(0, 3)],  # Frame indices
            [
                torch.rand([1, 3, 224, 224]),  # frame_idx, channels, width, height
                torch.rand([3, 3, 224, 224]),  # frame_idx, channels, width, height
            ],
        ),
    ),
]


@pytest.fixture(params=MODULE_TO_TEST, ids=[str(m) for m in MODULE_TO_TEST])
def ml_module(request: SubRequest) -> ModuleTestConfiguration:
    """All modules that are part of the MLModule library"""
    return request.param


@pytest.fixture
def torch_ml_module(
    ml_module: ModuleTestConfiguration,
) -> ModuleTestConfiguration[TorchMlModule]:
    """All modules implemented in Torch"""
    if not ml_module.is_pytorch:
        pytest.skip(f"Skipping {ml_module} as it is not a PyTorch module")
    return ml_module


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


@pytest.fixture
def cats_and_dogs_images() -> List[str]:
    base_path = os.path.join("tests", "fixtures", "cats_dogs")
    return list_files_in_dir(base_path, allowed_extensions=("jpg",))[:50]


# OLD


@pytest.fixture(
    params=[
        lambda: TorchResNetModule("resnet18"),
        lambda: CLIPImageModule("ViT-B/32"),
        lambda: CLIPTextModule("ViT-B/32"),
        # CLIPViTB32ImageEncoder,
        # ArcFaceFeatures,
        # MagFaceFeatures,
        # TorchMLModuleKeyFrames,
        # VinVLDetector - too slow to download
    ],
    ids=[
        "torch-resnet-18",
        "torch-keyframes",
        "torch-clip-vit-image",
        "torch-clip-vit-text",
    ],
)
def module_pretrained_by_provider(
    request: SubRequest,
):
    """Returns a module that implements DownloadPretrainedStateFromProvider"""
    return request.param


@pytest.fixture(
    params=[
        lambda: TorchResNetModule("resnet18"),
        lambda: CLIPImageModule("ViT-B/32"),
        lambda: CLIPTextModule("ViT-B/32"),
        # CLIPViTB32ImageEncoder,
        # ArcFaceFeatures,
        # MagFaceFeatures,
        # TorchMLModuleKeyFrames,
        # VinVLDetector - too slow to download
    ],
    ids=[
        "torch-resnet-18",
        "torch-keyframes",
        "torch-clip-vit-image",
        "torch-clip-vit-text",
    ],
)
def module_pretrained_mlmodule_store(
    request: SubRequest,
):
    return request.param


@pytest.fixture(
    params=[
        DenseNet161ImageNetFeatures,
        DenseNet161ImageNetClassifier,
        DenseNet161PlacesFeatures,
        DenseNet161PlacesClassifier,
        ArcFaceFeatures,
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
        DenseNet161ImageNetFeatures,
        DenseNet161PlacesFeatures,
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
