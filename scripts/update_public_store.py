"""This script will take model states from mlmodule.contrib and publish them to the MLModule store"""
import itertools
import logging
from typing import Iterable, List, Tuple

from mlmodule.contrib.clip.base import BaseCLIPModule
from mlmodule.contrib.clip.image import CLIPImageModule
from mlmodule.contrib.clip.parameters import PARAMETERS
from mlmodule.contrib.clip.stores import CLIPStore
from mlmodule.contrib.clip.text import CLIPTextModule
from mlmodule.contrib.resnet.modules import TorchResNetModule
from mlmodule.contrib.resnet.stores import ResNetTorchVisionStore
from mlmodule.helpers.torchvision import ResNetArch
from mlmodule.v2.base.models import ModelWithState
from mlmodule.v2.stores.abstract import AbstractStateStore
from mlmodule.v2.stores.github import GitHUBReleaseStore

logger = logging.getLogger(__name__)


def get_mlmodule_store() -> GitHUBReleaseStore:
    return GitHUBReleaseStore("LSIR", "mlmodule")


def get_contrib_resnet_stores() -> List[
    Tuple[TorchResNetModule, ResNetTorchVisionStore]
]:
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


def get_contrib_clip_stores() -> List[Tuple[BaseCLIPModule, CLIPStore]]:
    """CLIP models and stores"""
    store = CLIPStore()
    ret: List[Tuple[BaseCLIPModule, CLIPStore]] = []
    for model_name in PARAMETERS:
        ret.append((CLIPImageModule(model_name), store))
        ret.append((CLIPTextModule(model_name), store))
    return ret


def get_contrib_model_stores() -> Iterable[Tuple[ModelWithState, AbstractStateStore]]:
    """List of all models with associated store in the contrib module"""
    return itertools.chain(get_contrib_resnet_stores(), get_contrib_clip_stores())


def main():
    mlmodule_store = get_mlmodule_store()
    for model, provider_store in get_contrib_model_stores():
        # Getting available state keys in the provider store and not available in mlmodule
        state_keys = set(provider_store.get_state_keys(model.state_type)) - set(
            mlmodule_store.get_state_keys(model.state_type)
        )

        for sk in state_keys:
            logger.info(f"Saving {sk} to MLModule store")
            provider_store.load(model, sk)
            mlmodule_store.save(model, training_id=sk.training_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
