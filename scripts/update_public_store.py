"""This script will take model states from mlmodule.models and publish them to the MLModule store"""
import argparse
import dataclasses
import itertools
import logging
from typing import Iterable, List, Optional, Tuple, Union

from mlmodule.helpers.torchvision import DenseNetArch, ResNetArch
from mlmodule.labels.places import PLACES_LABELS
from mlmodule.models.clip.base import BaseCLIPModule
from mlmodule.models.clip.image import CLIPImageModule
from mlmodule.models.clip.parameters import PARAMETERS
from mlmodule.models.clip.stores import CLIPStore
from mlmodule.models.clip.text import CLIPTextModule
from mlmodule.models.densenet.modules import TorchDenseNetModule
from mlmodule.models.densenet.stores import (
    DenseNetPlaces365Store,
    DenseNetTorchVisionStore,
)
from mlmodule.models.magface.modules import TorchMagFaceModule
from mlmodule.models.magface.stores import MagFaceStore
from mlmodule.models.mtcnn.modules import TorchMTCNNModule
from mlmodule.models.mtcnn.stores import FaceNetMTCNNStore
from mlmodule.models.resnet.modules import TorchResNetModule
from mlmodule.models.resnet.stores import ResNetTorchVisionStore
from mlmodule.models.sentences.distilbert.modules import (
    DistilUseBaseMultilingualCasedV2Module,
)
from mlmodule.models.sentences.distilbert.stores import (
    SBERTDistiluseBaseMultilingualCasedV2Store,
)
from mlmodule.models.vinvl.modules import TorchVinVLDetectorModule
from mlmodule.models.vinvl.stores import VinVLStore
from mlmodule.v2.base.models import ModelWithState
from mlmodule.v2.states import StateKey, StateType
from mlmodule.v2.stores.abstract import AbstractStateStore
from mlmodule.v2.stores.github import GitHUBReleaseStore

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class UpdatePublicStoreOptions:
    backend: Optional[str] = None
    architecture: Optional[str] = None
    dry_run: bool = False


def get_mlmodule_store() -> GitHUBReleaseStore:
    return GitHUBReleaseStore("LSIR", "mlmodule")


def get_resnet_stores() -> List[Tuple[TorchResNetModule, ResNetTorchVisionStore]]:
    """ResNet models and store"""
    resnet_arch: List[ResNetArch] = [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "resnext50_32x4d",
        "resnext101_32x8d",
        "wide_resnet50_2",
        "wide_resnet101_2",
    ]
    store = ResNetTorchVisionStore()
    ret: List[Tuple[TorchResNetModule, ResNetTorchVisionStore]] = []
    for a in resnet_arch:
        ret.append((TorchResNetModule(a), store))

    return ret


def get_clip_stores() -> List[Tuple[BaseCLIPModule, CLIPStore]]:
    """CLIP models and stores"""
    store = CLIPStore()
    ret: List[Tuple[BaseCLIPModule, CLIPStore]] = []
    for model_name in PARAMETERS:
        ret.append((CLIPImageModule(model_name), store))
        ret.append((CLIPTextModule(model_name), store))
    return ret


def get_densenet_stores() -> List[
    Tuple[TorchDenseNetModule, Union[DenseNetTorchVisionStore, DenseNetPlaces365Store]]
]:
    """DenseNet models and stores"""
    tv_store = DenseNetTorchVisionStore()
    p_store = DenseNetPlaces365Store()

    densenet_archs: List[DenseNetArch] = [
        "densenet121",
        "densenet161",
        "densenet169",
        "densenet201",
    ]
    ret: List[
        Tuple[
            TorchDenseNetModule, Union[DenseNetTorchVisionStore, DenseNetPlaces365Store]
        ]
    ] = []
    for d in densenet_archs:
        ret.append((TorchDenseNetModule(d), tv_store))
        ret.append((TorchDenseNetModule(d, label_set=PLACES_LABELS), p_store))

    return ret


def get_magface_stores() -> List[Tuple[TorchMagFaceModule, MagFaceStore]]:
    return [(TorchMagFaceModule(), MagFaceStore())]


def get_mtcnn_stores() -> List[Tuple[TorchMTCNNModule, FaceNetMTCNNStore]]:
    return [(TorchMTCNNModule(), FaceNetMTCNNStore())]


def get_distiluse_stores() -> List[
    Tuple[
        DistilUseBaseMultilingualCasedV2Module,
        SBERTDistiluseBaseMultilingualCasedV2Store,
    ]
]:
    return [
        (
            DistilUseBaseMultilingualCasedV2Module(),
            SBERTDistiluseBaseMultilingualCasedV2Store(),
        )
    ]


def get_vinvl_stores() -> List[Tuple[TorchVinVLDetectorModule, VinVLStore]]:
    return [(TorchVinVLDetectorModule(), VinVLStore())]


def get_all_models_stores() -> Iterable[Tuple[ModelWithState, AbstractStateStore]]:
    """List of all models with associated store in the contrib module"""
    return itertools.chain(
        get_resnet_stores(),
        get_clip_stores(),
        get_densenet_stores(),
        get_magface_stores(),
        get_mtcnn_stores(),
        get_distiluse_stores(),
        get_vinvl_stores(),
    )


def state_type_match(
    state_type: StateType,
    backend: Optional[str] = None,
    architecture: Optional[str] = None,
) -> bool:
    backend_match = backend is None or state_type.backend == backend
    architecture_match = architecture is None or state_type.architecture == architecture
    return backend_match and architecture_match


def iterate_state_keys_to_upload(
    mlmodule_store: AbstractStateStore,
    backend: Optional[str] = None,
    architecture: Optional[str] = None,
) -> Iterable[Tuple[ModelWithState, AbstractStateStore, StateKey]]:
    """Iterates over the missing model states in MLModule store"""
    ret: List[Tuple[ModelWithState, AbstractStateStore, StateKey]] = []
    for model, provider_store in get_all_models_stores():
        # If the state type does not match the backend and architecture filters
        # We skip this loop
        if not state_type_match(
            model.state_type, backend=backend, architecture=architecture
        ):
            continue

        # Getting available state keys in the provider store and not available in mlmodule
        already_uploaded_state_keys = set(
            mlmodule_store.get_state_keys(model.state_type)
        )

        for sk in provider_store.get_state_keys(model.state_type):
            if sk in already_uploaded_state_keys:
                logger.info(f"Already in MLModule store, skipping {sk}")
                continue
            ret.append((model, provider_store, sk))

    return ret


def main(options: UpdatePublicStoreOptions):
    mlmodule_store = get_mlmodule_store()
    for item in iterate_state_keys_to_upload(
        mlmodule_store, backend=options.backend, architecture=options.architecture
    ):
        model, provider_store, state_key = item
        logger.info(f"Saving {state_key} to MLModule store")

        # If dry run skipping loading and saving the model
        if options.dry_run:
            continue

        provider_store.load(model, state_key)
        mlmodule_store.save(model, training_id=state_key.training_id)


def parse_arguments() -> UpdatePublicStoreOptions:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", default=None, help="Filter model with the given backend"
    )
    parser.add_argument(
        "--architecture", default=None, help="Filter model with the given architecture"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Dry run, does not execute anything."
    )

    return UpdatePublicStoreOptions(**vars(parser.parse_args()))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(parse_arguments())
