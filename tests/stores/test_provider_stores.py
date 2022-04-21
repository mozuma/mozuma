import dataclasses

import pytest

from mlmodule.v2.base.models import ModelWithState
from mlmodule.v2.states import StateKey
from mlmodule.v2.stores import Store
from mlmodule.v2.testing import ModuleTestConfiguration


@pytest.mark.slow
def test_provider_stores_load(ml_module: ModuleTestConfiguration[ModelWithState]):
    if ml_module.provider_store is None:
        pytest.skip(
            f"Model test config {ml_module.name} does not provide a store to test"
        )
    store = ml_module.provider_store
    model = ml_module.get_module()

    # Initial random state
    random_state = model.get_state()

    # Create the state key
    state_key = StateKey(model.state_type, training_id=ml_module.training_id)

    # Loading weights
    store.load(model, state_key)
    new_state = model.get_state()

    # It should have loaded a different state
    assert (
        random_state != new_state
    ), f"Binary state of model hasn't changed after loading from provider store: {store}"

    # Loading the state from MLModule
    new_model = ml_module.get_module()
    Store().load(new_model, state_key)
    mlmodule_state = new_model.get_state()

    # It should be the same
    assert (
        mlmodule_state == new_state
    ), f"Binary state of model from MLModule and provider store are different for {store}"


def test_provider_store_get_state_keys(
    ml_module: ModuleTestConfiguration[ModelWithState],
):
    if ml_module.provider_store is None:
        pytest.skip(
            f"Model test config {ml_module.name} does not provide a store to test"
        )
    store = ml_module.provider_store
    model = ml_module.get_module()

    # Getting available state keys
    state_keys = store.get_state_keys(model.state_type)
    # They should all be compatible with the model
    assert all(model.state_type.is_compatible_with(s.state_type) for s in state_keys)
    # Make sure the returned training ids are the ones expected
    training_ids = {s.training_id for s in state_keys}
    assert ml_module.training_id in training_ids


def test_provider_unknown_state_type(
    ml_module: ModuleTestConfiguration[ModelWithState],
):
    if ml_module.provider_store is None:
        pytest.skip(
            f"Model test config {ml_module.name} does not provide a store to test"
        )
    store = ml_module.provider_store
    model = ml_module.get_module()

    # Getting state keys for an unknown type
    state_keys = store.get_state_keys(
        dataclasses.replace(model.state_type, architecture="unknown")
    )
    # Should be empty
    assert len(state_keys) == 0
