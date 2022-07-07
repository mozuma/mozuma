import pytest

from mozuma.models.types import ModelWithState
from mozuma.states import StateKey
from mozuma.stores import Store
from mozuma.testing import ModuleTestConfiguration


def test_mozuma_store(ml_module: ModuleTestConfiguration):
    """Test that a state can be loaded from MoZuMa"""
    if ml_module.training_id is None:
        pytest.skip("Module not available in MoZuMa store")

    store = Store()
    model: ModelWithState = ml_module.get_module()
    state_key = StateKey(state_type=model.state_type, training_id=ml_module.training_id)

    # Loading
    store.load(model, state_key)
