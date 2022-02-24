import re
from typing import Callable

from mlmodule.v2.base.models import ModelWithState, ModelWithStateFromProvider

VALID_STATE_ARCH_NAMES = re.compile(r"^[\w\-\d]+$")


def test_state_architecture_consistency_provider(
    module_pretrained_by_provider: Callable[[], ModelWithStateFromProvider]
):
    model = module_pretrained_by_provider()

    # Testing to load all available states
    provider_states = model.provider_state_architectures()
    for state_arch in provider_states:
        # Loading the state from provider
        state_id = model.set_state_from_provider(state_arch)
        # Makes sure the returned state is consistent with the requested arch
        assert state_id.state_type == state_arch
        # The training ID must be set
        assert state_id.training_id is not None


def test_architecture_valid_name(
    module_pretrained_mlmodule_store: Callable[[], ModelWithState]
):
    model = module_pretrained_mlmodule_store()
    state_arch = model.state_type

    # Currently we only implement PyTorch models
    assert state_arch.backend in ("pytorch",)
    assert VALID_STATE_ARCH_NAMES.match(state_arch.architecture)


def test_is_state_compatible_consistency(
    module_pretrained_mlmodule_store: Callable[[], ModelWithState]
):
    """Tests whether the state architecture of a model passes the is compatible function"""
    model = module_pretrained_mlmodule_store()

    assert model.is_state_compatible(model.state_type())
