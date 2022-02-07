from typing import Callable

from mlmodule.v2.base.models import ModelWithStateFromProvider


def test_pretrained_download_from_provider_consistency(
    module_pretrained_by_provider: Callable[[], ModelWithStateFromProvider],
) -> None:
    # Making sure the set_state_from_provider changes internal state
    model1 = module_pretrained_by_provider()
    inital_state = model1.get_state()
    model1.set_state_from_provider()
    state_from_provider1 = model1.get_state()
    assert inital_state != state_from_provider1

    # Making sure that setting the state on a second model yields the same state
    model2 = module_pretrained_by_provider()
    model2.set_state_from_provider()
    state_from_provider2 = model2.get_state()
    assert state_from_provider1 == state_from_provider2
