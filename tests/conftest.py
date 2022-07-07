import os
from typing import List

import pytest
import torch
from _pytest.fixtures import SubRequest

from mozuma.helpers.files import list_files_in_dir
from mozuma.labels.imagenet import IMAGENET_LABELS
from mozuma.models.arcface.modules import TorchArcFaceModule
from mozuma.models.arcface.stores import ArcFaceStore
from mozuma.models.classification.modules import (
    LinearClassifierTorchModule,
    MLPClassifierTorchModule,
)
from mozuma.models.clip.image import CLIPImageModule
from mozuma.models.clip.stores import CLIPStore
from mozuma.models.clip.text import CLIPTextModule
from mozuma.models.densenet.modules import TorchDenseNetModule, torch_densenet_places365
from mozuma.models.densenet.stores import (
    DenseNetPlaces365Store,
    DenseNetTorchVisionStore,
)
from mozuma.models.keyframes.encoders import VideoFramesEncoder
from mozuma.models.keyframes.selectors import KeyFrameSelector
from mozuma.models.magface.modules import TorchMagFaceModule
from mozuma.models.magface.stores import MagFaceStore
from mozuma.models.mtcnn.modules import TorchMTCNNModule
from mozuma.models.mtcnn.stores import FaceNetMTCNNStore
from mozuma.models.resnet.modules import TorchResNetModule
from mozuma.models.resnet.stores import ResNetTorchVisionStore
from mozuma.models.sentences.distilbert.modules import (
    DistilUseBaseMultilingualCasedV2Module,
)
from mozuma.models.sentences.distilbert.stores import (
    SBERTDistiluseBaseMultilingualCasedV2Store,
)
from mozuma.models.vinvl.modules import TorchVinVLDetectorModule
from mozuma.models.vinvl.stores import VinVLStore
from mozuma.testing import ModuleTestConfiguration
from mozuma.torch.modules import TorchModel

MODULE_TO_TEST: List[ModuleTestConfiguration] = [
    # ResNet
    ModuleTestConfiguration(
        "torchresnet18",
        lambda: TorchResNetModule("resnet18"),
        batch_factory=lambda: torch.rand(
            [2, 3, 224, 224]
        ),  # batch, channels, width, height
        provider_store=ResNetTorchVisionStore(),
        training_id="imagenet",
    ),
    # DenseNet
    ModuleTestConfiguration(
        "torchdensenet161",
        lambda: TorchDenseNetModule("densenet161"),
        batch_factory=lambda: torch.rand(
            [2, 3, 224, 224]
        ),  # batch, channels, width, height
        provider_store=DenseNetTorchVisionStore(),
        training_id="imagenet",
    ),
    ModuleTestConfiguration(
        "torchdensenet161places",
        lambda: torch_densenet_places365("densenet161"),
        batch_factory=lambda: torch.rand(
            [2, 3, 224, 224]
        ),  # batch, channels, width, height
        provider_store=DenseNetPlaces365Store(),
        training_id="places365",
    ),
    # CLIP
    ModuleTestConfiguration(
        "clip-image-rn50",
        lambda: CLIPImageModule("RN50"),
        batch_factory=lambda: torch.rand(
            [2, 3, 224, 224]
        ),  # batch, channels, width, height
        provider_store=CLIPStore(),
        training_id="clip",
    ),
    ModuleTestConfiguration(
        "clip-text-rn50",
        lambda: CLIPTextModule("RN50"),
        batch_factory=lambda: torch.randint(10, size=(2, 77)),  # batch, ctx_len
        provider_store=CLIPStore(),
        training_id="clip",
    ),
    # MTCNN
    ModuleTestConfiguration(
        "mtcnn",
        lambda: TorchMTCNNModule(),
        batch_factory=lambda: [torch.rand([720 + i * 10, 720, 3]) for i in range(5)],
        provider_store=FaceNetMTCNNStore(),
        training_id="facenet",
    ),
    # ArcFace
    ModuleTestConfiguration(
        "arcface",
        lambda: TorchArcFaceModule(),
        batch_factory=lambda: torch.rand(
            (2, 3, 112, 112)
        ),  # batch, channels, width, height
        provider_store=ArcFaceStore(),
        training_id="insightface",
    ),
    # MagFace
    ModuleTestConfiguration(
        "magface",
        lambda: TorchMagFaceModule(),
        batch_factory=lambda: torch.rand(
            (2, 3, 112, 112)
        ),  # batch, channels, width, height
        provider_store=MagFaceStore(),
        training_id="magface",
    ),
    # Key-frames
    ModuleTestConfiguration(
        "frames-encoder-rn18",
        lambda: VideoFramesEncoder(TorchResNetModule("resnet18")),
        training_id="imagenet",
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
        training_id="imagenet",
        batch_factory=lambda: (
            [torch.range(0, 1), torch.range(0, 3)],  # Frame indices
            [
                torch.rand([1, 3, 224, 224]),  # frame_idx, channels, width, height
                torch.rand([3, 3, 224, 224]),  # frame_idx, channels, width, height
            ],
        ),
    ),
    # VinVL
    ModuleTestConfiguration(
        "vinvl",
        lambda: TorchVinVLDetectorModule(),
        batch_factory=lambda: (
            torch.rand(5, 3, 60, 56),  # batch, channels, width, height
            [(56, 56)] * 5,  # width, height
        ),
        provider_store=VinVLStore(),
        training_id="vinvl",
    ),
    # S-BERT
    ModuleTestConfiguration(
        "distiluse-base-multilingual-cased-v2",
        lambda: DistilUseBaseMultilingualCasedV2Module(),
        training_id="cased-v2",
        batch_factory=lambda: (
            torch.LongTensor([[2, 3, 4]]),  # token ids
            torch.FloatTensor([[1, 1, 1]]),  # attention mask
        ),
        provider_store=SBERTDistiluseBaseMultilingualCasedV2Store(),
    ),
    # Classifiers
    ModuleTestConfiguration(
        "linear-classifier",
        lambda: LinearClassifierTorchModule(in_features=10, label_set=IMAGENET_LABELS),
        batch_factory=lambda: torch.rand(3, 10),  # batch, input dimension
    ),
    ModuleTestConfiguration(
        "mlp-classifier",
        lambda: MLPClassifierTorchModule(
            in_features=10,
            hidden_layers=(5, 3),
            label_set=IMAGENET_LABELS,
            activation="ReLU",
        ),
        batch_factory=lambda: torch.rand(3, 10),  # batch, input dimension
    ),
]


@pytest.fixture(params=MODULE_TO_TEST, ids=[str(m) for m in MODULE_TO_TEST])
def ml_module(request: SubRequest) -> ModuleTestConfiguration:
    """All modules that are part of the MLModule library"""
    return request.param


@pytest.fixture
def torch_ml_module(
    ml_module: ModuleTestConfiguration,
) -> ModuleTestConfiguration[TorchModel]:
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


@pytest.fixture(scope="session")
def cats_and_dogs_images() -> List[str]:
    base_path = os.path.join("tests", "fixtures", "cats_dogs")
    return list_files_in_dir(base_path, allowed_extensions=("jpg",))[:50]


@pytest.fixture(scope="session")
def video_file_path() -> str:
    return os.path.join("tests", "fixtures", "video", "test.mp4")
