from typing import Callable

from mlmodule.v2.base.models import ModelWithState, ModelWithStateFromProvider
from mlmodule.v2.stores import LocalFileModelStore, MLModuleModelStore


def test_pretrained_download_from_provider_consistency(
    module_pretrained_by_provider: Callable[[], ModelWithStateFromProvider],
) -> None:
    # Making sure the set_state_from_provider changes internal state
    model1 = module_pretrained_by_provider()
    initial_state = model1.get_state()
    model1.set_state_from_provider()
    state_from_provider1 = model1.get_state()
    assert initial_state != state_from_provider1

    # Making sure that setting the state on a second model yields the same state
    model2 = module_pretrained_by_provider()
    model2.set_state_from_provider()
    state_from_provider2 = model2.get_state()
    assert state_from_provider1 == state_from_provider2

    store = MLModuleModelStore()
    # Making sure that getting the weight from MLModule return the same state
    mlmodule_model = module_pretrained_by_provider()
    store.load(mlmodule_model)
    state_from_mlmodule = mlmodule_model.get_state()
    assert state_from_provider1 == state_from_mlmodule


def test_pretrained_download_from_mlmodule_consistency(
    module_pretrained_mlmodule_store: Callable[[], ModelWithState]
) -> None:
    store = MLModuleModelStore()
    # store = LocalFileModelStore("models")     # For testing new models

    # Making sure the set_state_from_provider changes internal state
    model1 = module_pretrained_mlmodule_store()
    initial_state = model1.get_state()
    # Loading weights from MLModule
    store.load(model1)
    state_from_provider1 = model1.get_state()
    assert initial_state != state_from_provider1

    # Making sure that setting the state on a second model yields the same state
    model2 = module_pretrained_mlmodule_store()
    store.load(model2)
    state_from_provider2 = model2.get_state()
    assert state_from_provider1 == state_from_provider2
