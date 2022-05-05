import pytest

from mlmodule.models.types import ModelWithState
from mlmodule.states import StateKey
from mlmodule.v2.stores import Store
from mlmodule.v2.testing import ModuleTestConfiguration


def test_mlmodule_store(ml_module: ModuleTestConfiguration):
    """Test that a state can be loaded from MLModule"""
    if ml_module.training_id is None:
        pytest.skip("Module not available in MlModule store")

    store = Store()
    model: ModelWithState = ml_module.get_module()
    state_key = StateKey(state_type=model.state_type, training_id=ml_module.training_id)

    # Loading
    store.load(model, state_key)
