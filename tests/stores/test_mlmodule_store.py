from mlmodule.v2.base.models import ModelWithState
from mlmodule.v2.stores import Store
from mlmodule.v2.testing import ModuleTestConfiguration


def test_mlmodule_store(ml_module: ModuleTestConfiguration):
    """Test that a state can be loaded from MLModule"""
    store = Store()
    model: ModelWithState = ml_module.get_module()

    # Choosing the first available state
    state_key = store.get_state_keys(model.state_type)[0]

    # Loading
    store.load(model, state_key)
