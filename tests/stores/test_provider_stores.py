import pytest

from mlmodule.v2.base.models import ModelWithState
from mlmodule.v2.stores import Store
from mlmodule.v2.testing import ModuleTestConfiguration


def test_provider_stores_load(ml_module: ModuleTestConfiguration[ModelWithState]):
    if ml_module.provider_store is None:
        pytest.skip("Model test config does not provide a store")
    store = ml_module.provider_store
    model = ml_module.get_module()

    # Initial random state
    random_state = model.get_state()

    # Getting the first state key for testing
    state_key = store.get_state_keys(model.state_type)[0]

    # Loading weights
    store.load(model, state_key)
    new_state = model.get_state()

    # It should have loaded a different state
    assert random_state != new_state

    # Loading the state from MLModule
    new_model = ml_module.get_module()
    Store().load(new_model, state_key)
    mlmodule_state = new_model.get_state()

    # It should be the same
    assert mlmodule_state == new_state


def test_provider_store_get_state_keys(
    ml_module: ModuleTestConfiguration[ModelWithState],
):
    if ml_module.provider_store is None:
        pytest.skip("Model test config does not provide a store")
    store = ml_module.provider_store
    model = ml_module.get_module()

    # Getting available state keys
    state_keys = store.get_state_keys(model.state_type)
    # They should all be compatible with the model
    assert all(model.state_type.is_compatible_with(s.state_type) for s in state_keys)
    # Make sure the returned training ids are the ones expected
    training_ids = {s.training_id for s in state_keys}
    assert ml_module.provider_store_training_ids == training_ids
